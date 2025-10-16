import os,sys
import math
import time
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from torch import optim
from torch.utils.data import DistributedSampler
from evaluation.surrogate_loss2 import SurrogateLossForRankingLoss
from models.linear_model import Linear
from evaluation.metrics import RankingLoss,MicroAUC
from benchmarks.tabular_data_benchmark import NFoldTabularDataset
from utils.tools import init_random_seed,generate_default_config
from benchmarks.tabular_data_benchmark import *
import numpy as np
from utils.deconstruct_yaml import get_config_from_yaml,mix_config_parser
from utils.get_path import get_project_path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,default="tabular_config.yaml")
# dataset can change e.g 'yeast','scene'
parser.add_argument('--dataset', '-dataset', type=str, default="delicious",
                    help='dataset on which experiment is conducted')
parser.add_argument('--ckpt_path', '-ckpt_path', type=str, default="checkpoint",
                    help='dataset on which experiment is conducted')
parser.add_argument('--log_path', '-log_path', type=str, default="tabular",
                    help='dataset on which experiment is conducted')
parser.add_argument('--mode', '-mode', type=str, default="pa",
                    help='loss on which experiment is conducted')
parser.add_argument('--model', '-model', type=str, default="Linear",
                    help='loss on which experiment is conducted')
parser.add_argument('--base_loss', '-base_loss', type=str, default="hinge",
                    help='base_loss on which experiment is conducted')
parser.add_argument('--batch_size', '-bs', type=int, default=512,
                    help='batch size for one iteration during training')
parser.add_argument('--lr', '-lr', type=float, default=0.1,
                    help='learning rate parameter')
parser.add_argument('--weight_decay', '-weight_decay', type=float, default=1e-5,
                    help='learning rate parameter')
parser.add_argument('--max_epoch', '-max_epoch', type=int, default=300,
                    help='maximal training epochs')
parser.add_argument('--local_rank', '-local_rank', type=int, default=0)
parser.add_argument('--seed', '-seed', type=int, default=0)
parser.add_argument('--n_hidden', type=int, default=50)
parser.add_argument('--cuda', type=int, default=2)
parser.add_argument('--reuse', '-reuse', action='store_true', help='parameter reuse')
parser.add_argument('--split', '-split', action='store_true', help='parameter reuse')
parser.add_argument('--nfold', '-nfold', type=int, default=3,
                    help='the rate of cross validation')


def main(args):
    log_path = os.path.join(get_project_path(), args.log_root, args.log_path + "-" + args.model + os.sep)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    log_dataset = os.path.join(log_path, args.dataset, str(args.n_hidden), args.mode, str(args.weight_decay))
    if not os.path.isdir(log_dataset):
        os.makedirs(log_dataset)

    # check = get_project_path() + args.log_root + args.log_path + args.ckpt_path
    check = os.path.join(log_dataset, args.ckpt_path)
    if not os.path.isdir(check):
        os.makedirs(check)
    save_name = args.dataset + '_' + str(args.n_hidden) + '_' + args.mode + '_' + str(args.weight_decay)

    # Setting random seeds
    init_random_seed(args.seed)
    configs = generate_default_config()
    configs['rand_seed'] = args.seed
    configs['device'] = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
#    configs['device'] = torch.device('cpu')
    configs['train_batch_size'] = args.batch_size
    configs['test_batch_size'] = 2 * configs['train_batch_size']
    configs['max_epoch'] = args.max_epoch
    configs['data_standardizing'] = True

    configs['split'] = args.split
    nfold = args.nfold
    dataset = NFoldTabularDataset(args.dataset, configs=configs, nfold=nfold)
    configs['dataset_name'] = args.dataset
    configs['in_features'] = dataset.feat_dim
    configs['num_classes'] = dataset.num_class
    configs['n_hidden'] = args.n_hidden
    configs['lr'] = args.lr
   
    test_performances = []
    test_performances_losses = []
    for count in range(1, nfold + 1):
        # with open(os.path.join(log_dataset, "train_value_bound_u1_300.txt"), mode="a") as f1:
        # print('Cross-validation: [{}/{}].'.format(count, nfold))
        dataset.cv(test_num=count, nfold=nfold)
        train_dataset = dataset.train_dataset
        test_dataset = dataset.test_dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=configs['train_batch_size'],
                                                       shuffle=True, num_workers=configs['num_workers'],
                                                       pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=configs['test_batch_size'],
                                                      shuffle=False, pin_memory=True,
                                                      num_workers=configs['num_workers'])

        if args.model == 'Linear':
            model = Linear(configs).to(configs['device'])
        else:
            print("unknown model type")
            sys.exit(0)

        # state optim
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=args.weight_decay)
        lr = args.lr
        crition = SurrogateLossForRankingLoss(args.mode)

        # training
        print('Training Beginning.')
        max_epoch = args.max_epoch
        train_loss = []
        train_log_name = 'train_loss_' + str(args.n_hidden) + '_' + str(count) + '.txt'
        test_log_name = 'test_loss_' + str(args.n_hidden) + '_' + str(count) + '.txt'
        finall_train_loss = 0.0
        for epoch in range(max_epoch):

            epoch_loss = 0.0
            epoch_auc = 0.0
            exit_model = False
            for iteratrion, (inputs, targets) in enumerate(train_dataloader):
                # output preds
                inputs, targets = inputs.to(configs['device']), targets.to(configs['device'])
                outputs = model(inputs)
                # compute loss
                loss = crition(outputs, targets).to(configs['device'])
                # backward()
                optimizer.zero_grad()
                loss.backward()
                # SGD 
                optimizer.step()
                train_aucloss = MicroAUC(outputs.detach(),targets)
                epoch_loss = epoch_loss + loss
                epoch_auc = epoch_auc + train_aucloss
            # finall_train_loss = (epoch_loss / len(train_dataloader)).cpu().item()
            finall_train_loss = (epoch_loss / len(train_dataloader)).item()
            finall_train_auc = (epoch_auc / len(train_dataloader))
            print("epoch: ", epoch, ", training macro-auc : ", finall_train_auc, ", train_loss", finall_train_loss)


                   
       
            train_loss.append((epoch, (epoch_loss / len(train_dataloader)).item(), train_aucloss))

        print('Training Finish.')
        state = {
            'net': model.state_dict()
        }
        torch.save(state, os.path.join(check, '%s.pth' % save_name))

        # print('Testing Beginning.')

        finall_test_loss = 0.0
        finall_test_auc = 0.0
        for iter_test, (test_inputs, test_targets) in enumerate(test_dataloader):
            # output preds
            test_inputs, test_targets = test_inputs.to(configs['device']), test_targets.to(configs['device'])
            test_outputs = model(test_inputs)
            # compute loss
            test_loss = crition(test_outputs, test_targets).to(configs['device'])
            test_auc = MicroAUC(test_outputs.detach(),test_targets)
    #        test_rankingloss = RankingLoss(test_outputs.detach(), test_targets)
            finall_test_loss = finall_test_loss + test_loss
            finall_test_auc = finall_test_auc + test_auc
        # finall_test_loss = (finall_test_loss / len(test_dataloader)).cpu().item()
        finall_test_loss = (finall_test_loss / len(test_dataloader)).item()
        finall_test_auc = (finall_test_auc / len(test_dataloader))

        print('{}_{} test_loss/test_auc in {}:{}/{}'.format(args.dataset, count, args.mode, finall_test_loss,
                                                            finall_test_auc))
        test_meature = []
        test_meature.append((finall_test_loss, finall_test_auc))
        test_performances.append(finall_test_auc)
        test_performances_losses.append(finall_test_loss)
     #   np.savetxt(os.path.join(log_dataset, test_log_name), test_meature, fmt=['%.4f', '%.4f'])

        del model

    # test_performances = np.stack(test_performances)
    test_performances_losses = np.stack(test_performances_losses)
    # mean, std = np.mean(test_performances), np.std(test_performances)
    mean, std = np.mean(test_performances_losses), np.std(test_performances_losses)
    #with open(os.path.join(log_dataset, "overall_test_performance_u1_300.txt"), mode="w+") as f:
    with open(os.path.join(log_dataset, "test_loss_average.txt"), mode="w+") as f:
        f.write("mean:" + str(np.round(mean, 4)) + "\n")
        f.write("std: " + str(np.round(std, 4)))

def time_function(run_time):
    if run_time < 60:
        return f"{run_time:.6f} s\n"
    elif run_time < 3600:  # 60秒*60=3600秒=1小时
        minutes = run_time / 60
        return f"{minutes:.6f} minutes\n"
    elif run_time < 86400:  # 3600秒*24=86400秒=1天
        hours = run_time / 3600
        return f"{hours:.6f} hours\n"
    else:
        days = run_time / 86400
        return f"{days:.6f} days\n"

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config_from_yaml(args.config)
    args_list = mix_config_parser(args, config)
    print(args.dataset)

    start_time = time.time()
    for args in args_list:
        main(args)

    end_time = time.time()

    run_time = end_time - start_time

    time_info = f"\nrunning time:{run_time:.6f}s\n"

    time_format = time_function(run_time)

    with open("/data/shaoxiao/Desktop/Multi-graph concentration resources/macro-auc-pytorch-main/running_time_new.txt", mode = "a") as f:
        f.write(args.dataset)
        f.write(time_info)
        f.write(time_format)



