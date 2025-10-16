import os,sys
import math
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#from gesvd.gesvd import GESVD, GESVDFunction
#svd = GESVD()
#svd = GESVDFunction()
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
parser.add_argument('--dataset', '-dataset', type=str, default="CAL500",
                    help='dataset on which experiment is conducted')
parser.add_argument('--ckpt_path', '-ckpt_path', type=str, default="checkpoint",
                    help='dataset on which experiment is conducted')
parser.add_argument('--log_path', '-log_path', type=str, default="tabular",
                    help='dataset on which experiment is conducted')
parser.add_argument('--mode', '-mode', type=str, default="u1",
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

# a function of svd
def regularize_weights_svd (model,svd_theta,svd_lambda,i_step,configs) :
    W = model.classifier.weight.data
#    U,S,Vh = svd.apply(W)
    U,S ,Vh = torch.linalg.svd(W,full_matrices=False)
#    U,S,Vh = svd(W)
#    S = np.diag(S)

    S = torch.diag(S)
#    print(svd_theta.item())

    if svd_theta > S.size(1):
        svd_theta = S.size(1)
#    print(svd_theta.item())
#    S = S.to(configs['device'])
#    U = U.to(configs['device'])
#    Vh = Vh.to(configs['device'])

    for i in range(int(svd_theta)):
        S[i,i] = max(0,S[i,i]-i_step*svd_lambda)
    W = (U @ S ) @ Vh

#    W = torch.nan_to_num(W,nan=0.0)



#    W = W.to('cuda')
    model.classifier.weight.data = W

# solve ill-conditions W 

#compute bound all
def compute_bound_value(empirical_loss,dataset,model,svd_theta,new_delta,bound_type):
    bound = 0
    main_r = 0
    if bound_type == "pa_new":
        main_r,bound = compute_bound_value_pa(empirical_loss,dataset,model,svd_theta,new_delta)
    elif bound_type == "u1_new":
        bound = compute_bound_value_u1(empirical_loss,dataset,model,svd_theta,new_delta)
    elif bound_type == "u2_new":
        bound = compute_bound_value_u2(empirical_loss,dataset,model,svd_theta,new_delta)

    elif bound_type == "pa_old":
        bound = compute_bound_value_initial_pa(empirical_loss,dataset,model,new_delta)
    elif bound_type == "u1_old":
        bound = compute_bound_value_initial_u1(empirical_loss,dataset,model,new_delta)
    elif bound_type == "u2_old":
        bound = compute_bound_value_initial_u2(empirical_loss,dataset,model,new_delta)


    return main_r,bound


# compute 
def compute_bound_value_pa (empirical_loss,dataset,model,svd_theta,new_delta):
    W = model.classifier.weight.data
    X = dataset.X
    Y = dataset.Y
    num_instance = X.size(0)
    num_label = Y.size(1)
    a = 0
    for i_label in range(num_label):
        p_list = torch.where(Y[:,i_label]==1)[0]
#        q = torch.where(Y==1)
        num_pos = len(p_list)
#        m = num_pos / len(q[0])
        m = num_pos / num_instance
        if m >0.5:
            m = 1-m
        if m <1e-5:
            m = 1/num_instance

#         p_list = torch.where(Y[:,i_label]==1)[0]
#         q_list = torch.where(Y[:,i_label] == 0)[0]
# #        q = torch.where(Y==1)
#         num_pos = len(p_list)
#         num_neg = len(q_list)
# #        m = num_pos / len(q[0])
#         if num_pos < num_neg:
#             m = num_pos / num_instance
#         else:
#             m = num_neg / num_instance
#         # m = num_pos / num_instance
#         # if m >0.5:
#         #     m = 1-m
#         if m <1e-5:
#             m = 1/num_instance
#         p_list = torch.where(Y[:,i_label]==1)[0]
# #        q = torch.where(Y==1)
#         num_pos = len(p_list)
# #        m = num_pos / len(q[0])
#         m = num_pos / num_instance
#         if m >0.5:
#             m = 1-m
#         if m <1e-5:
#             m = 1/num_instance
        a = a+ 1/m
    
#    U,W_a,Vh = torch.linalg.svd(W)
    # r = 0
    # W += r* torch.eye(min(W.size()))
    W_a = torch.linalg.svdvals(W)

    M_b = torch.norm(X,p=2) 
    # M_b = 0
    # for i in range(num_instance):
    #     temp = torch.norm(X[i,:],p=2)
    #     if temp >M_b:
    #         M_b = temp
    M_a = 0
    for i in range(num_label):
        temp = torch.norm(W[i,:],p=2)
        if temp > M_a:
            M_a = temp
#    M_a = torch.norm(W,2)
    W_l = 0
#    print(W_a.size(0),svd_theta)
    for i in range(int(svd_theta),W_a.size(0)):
        W_l = W_l + W_a[i] * W_a[i]
    delta = 75 * a * math.log(1/new_delta) / (num_label * num_instance)
    a1 = svd_theta  * a / (M_b*M_b * num_instance)
    a2 = M_a * math.sqrt(a * W_l /num_instance)
    main_r = 704 * (a1+a2)
    bound = empirical_loss * 0 + main_r + delta

    return main_r,bound

def compute_bound_value_u1 (empirical_loss,dataset,model,svd_theta,new_delta):
    W = model.classifier.weight.data
    X = dataset.X
    Y = dataset.Y
    num_instance = X.size(0)
    num_label = Y.size(1)
    a = 0
    tau_s = 1
    for i_label in range(num_label):
        p_list = torch.where(Y[:,i_label]==1)[0]
#        q = torch.where(Y==1)
        num_pos = len(p_list)
#        m = num_pos / len(q[0])
        m = num_pos / num_instance
        if m >0.5:
            m = 1-m
        if m <1e-5:
            m = 1/num_instance
        if tau_s > m :
            tau_s = m
        a = a+ 1/m
    
#    U,W_a,Vh = torch.linalg.svd(W)
    # r = 0
    # W += r* torch.eye(min(W.size()))
    W_a = torch.linalg.svdvals(W)

    M_b = torch.norm(X,p=2) 
    # M_b = 0
    # for i in range(num_instance):
    #     temp = torch.norm(X[i,:],p=2)
    #     if temp >M_b:
    #         M_b = temp    
    M_a = 0
    for i in range(num_label):
        temp = torch.norm(W[i,:],p=2)
        if temp > M_a:
            M_a = temp
#    M_a = torch.norm(W,2)
    W_l = 0
#    print(W_a.size(0),svd_theta)
    for i in range(int(svd_theta),W_a.size(0)):
        W_l = W_l + W_a[i] * W_a[i]
    delta = 75 * a * math.log(1/new_delta) / (num_label * num_instance)
    a1 = svd_theta  * a / (M_b*M_b * num_instance)
    a2 = M_a * math.sqrt(a * W_l /num_instance)
    main_r = 704 * (a1+a2)
    bound = empirical_loss * 0 + main_r + delta
    bound = bound / tau_s

    return bound

def compute_bound_value_u2 (empirical_loss,dataset,model,svd_theta,new_delta):
    W = model.classifier.weight.data
    X = dataset.X
    Y = dataset.Y
    num_instance = X.size(0)
    num_label = Y.size(1)
    a = 0
#    tau_s = 1
    for i_label in range(num_label):
        p_list = torch.where(Y[:,i_label]==1)[0]
#        q = torch.where(Y==1)
        num_pos = len(p_list)
#        m = num_pos / len(q[0])
        m = num_pos / num_instance
        if m >0.5:
            m = 1-m
        if m <1e-5:
            m = 1/num_instance
        # if tau_s > m :
        #     tau_s = m
        a = a+ 1/m
    
#    U,W_a,Vh = torch.linalg.svd(W)
    # r = 0
    # W += r* torch.eye(min(W.size()))
    W_a = torch.linalg.svdvals(W)

    M_b = torch.norm(X,p=2) 
    M_b = 0
    for i in range(num_instance):
        temp = torch.norm(X[i,:],p=2)
        if temp >M_b:
            M_b = temp
    M_a = 0
    for i in range(num_label):
        temp = torch.norm(W[i,:],p=2)
        if temp > M_a:
            M_a = temp
#    M_a = torch.norm(W,2)
    W_l = 0
#    print(W_a.size(0),svd_theta)
    for i in range(int(svd_theta),W_a.size(0)):
        W_l = W_l + W_a[i] * W_a[i]
    delta = 115.625 * a * math.log(1/new_delta) / (num_label * num_instance)
    a1 = svd_theta  * a / (M_b*M_b * num_instance)
    a2 = M_a * math.sqrt(a * W_l /num_instance)
    main_r = 704 * (a1+a2) *2
    bound = empirical_loss * 0 + main_r + delta

    return bound

def compute_bound_value_initial_pa (empirical_loss,dataset,model,new_delta):
    W = model.classifier.weight.data
    X = dataset.X
    Y = dataset.Y
    num_instance = X.size(0)
    num_label = Y.size(1)
#    tau_s = 0
    a = 0
    b = 0

    for i_label in range(num_label):
        p_list = torch.where(Y[:,i_label]==1)[0]
        q_list = torch.where(Y[:,i_label] == 0)[0]
#        q = torch.where(Y==1)
        num_pos = len(p_list)
        num_neg = len(q_list)
#        m = num_pos / len(q[0])
        if num_pos < num_neg:
            m = num_pos / num_instance
        else:
            m = num_neg / num_instance
        # m = num_pos / num_instance
        # if m >0.5:
        #     m = 1-m
        if m <1e-5:
            m = 1/num_instance
        # if m < tau_s:
        #     tau_s = m
        a = a+ 1/m
        b = b + math.sqrt(1/m)

#    r = torch.norm(X,p=2)
    r = 0
    for i in range(num_instance):
        temp = torch.norm(X[i,:],p=2)
        if temp > r:
            r = temp

    M_a = 0
    for i in range(num_label):
        temp = torch.norm(W[i,:],p=2)
        if temp > M_a:
            M_a = temp 
    M_b = 1

    A = empirical_loss * 0
    B = 4 * r * M_a / math.sqrt(num_instance) * (b / num_label) * 2
    C = 3 * M_b * math.sqrt(math.log(2/new_delta) / 2/ num_instance) * math.sqrt(a / num_label) * 2



    bound = A + B + C

    return bound

def compute_bound_value_initial_u1 (empirical_loss,dataset,model,new_delta):
    W = model.classifier.weight.data
    X = dataset.X
    Y = dataset.Y
    num_instance = X.size(0)
    num_label = Y.size(1)
    tau_s = 1
    a = 0
    b = 0
#    r = torch.norm(X,p=2)
    r=0
    for i in range(num_instance):
        temp = torch.norm(X[i,:],p=2)
        if temp > r:
            r = temp
    for i_label in range(num_label):
        p_list = torch.where(Y[:,i_label]==1)[0]
        q_list = torch.where(Y[:,i_label] == 0)[0]
#        q = torch.where(Y==1)
        num_pos = len(p_list)
        num_neg = len(q_list)
#        m = num_pos / len(q[0])
        if num_pos < num_neg:
            m = num_pos / num_instance
        else:
            m = num_neg / num_instance
        if m <1e-5:
            m = 1/num_instance
        if tau_s > m :
            tau_s = m
        # if m < tau_s:
        #     tau_s = m
        a = a+ 1/m
        b = b + math.sqrt(1/m)
    M_a = 0
    M_b = 1
    for i in range(num_label):
        temp = torch.norm(W[i,:],p=2)
        if temp > M_a:
            M_a = temp 
    A = empirical_loss / tau_s * 0
    B = 4 * r * M_a / tau_s / math.sqrt(num_instance) * (b / num_label) *2
    C = 3 *M_b / tau_s * math.sqrt(math.log(2/new_delta) /2 / num_instance) * math.sqrt(a/num_label) *2 

    bound = A + B + C
    return bound

def compute_bound_value_initial_u2 (empirical_loss,dataset,model,new_delta):
    W = model.classifier.weight.data
    X = dataset.X
    Y = dataset.Y
    num_instance = X.size(0)
    num_label = Y.size(1)
    # tau_s = 1
    a = 0
    b = 0
#    r = torch.norm(X,p=2)
    r=0
    for i in range(num_instance):
        temp = torch.norm(X[i,:],p=2)
        if temp > r:
            r = temp
    for i_label in range(num_label):
        p_list = torch.where(Y[:,i_label]==1)[0]
        q_list = torch.where(Y[:,i_label] == 0)[0]
#        q = torch.where(Y==1)
        num_pos = len(p_list)
        num_neg = len(q_list)
#        m = num_pos / len(q[0])
        if num_pos < num_neg:
            m = num_pos / num_instance
        else:
            m = num_neg / num_instance
        if m <1e-5:
            m = 1/num_instance
        # if tau_s > m :
        #     tau_s = m
        # if m < tau_s:
        #     tau_s = m
        a = a+ 1/m
        b = b + math.sqrt(1/m)
    M_a = 0
    M_b = 1
    for i in range(num_label):
        temp = torch.norm(W[i,:],p=2)
        if temp > M_a:
            M_a = temp 
    A = empirical_loss  * 0
    B = 4 * r * M_a  / math.sqrt(num_instance) * (b / num_label) *2 *2
    C = 3 *M_b  * math.sqrt(math.log(2/new_delta) /2 / num_instance) * math.sqrt(a/num_label) *2 *2

    bound = A + B + C
    return bound

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
    # generate log

    # some parameter
    svd_rate = 80    # emotion
    svd_rate = 80    # image
    svd_rate = 80    #delicious
    svd_rate = 10    # cal500
    svd_rate = 80    #scene
    svd_rate = 60    # yeast
    svd_rate = 70    # corel5k
    svd_rate = 80    # rcv1subset1
    svd_rate = 80    # bibtex
#    svd_rate = 60
    svd_theta = svd_rate * min(dataset.num_class,args.batch_size)/100
#    svd_lambda = 1e-2
    svd_lambda = 1e-3  # emotion
    svd_lambda = 1e-3  # image
    svd_lambda = 1e-3  # delicious
    svd_lambda = 1e-3  # cal500
    svd_lambda = 1e-3  # scene
    svd_lambda = 1e-3  # yeast
    svd_lambda = 1e-1  # corel5k
    svd_lambda = 1e-3  # rcv1subset1
    svd_lambda = 1e-3  # bibtex
 #   svd_theta = torch.tensor(svd_theta)
    svd_theta = math.floor(svd_theta)
    new_delta = 0.01
    test_performances = []
    for count in range(1, nfold + 1):
        # with open(os.path.join(log_dataset, "train_value_bound_u1_300.txt"), mode="a") as f1:
        with open(os.path.join(log_dataset, "train_value_bound_new_pa.txt"), mode="a") as f1:
            f1.write("svd_rate: "+str(svd_rate)+"\n"+"count n-fold : "+ str(count)+ "\n")
    
        print('Cross-validation: [{}/{}].'.format(count, nfold))
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
        # elif args.model == 'MLP':
        #     model = MLP(configs).to(configs['device'])
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
                #... add SVD to linear.classifier.weights
 #               regularize_weights_svd(model,svd_theta,svd_lambda,i_step=epoch,configs=configs)

                #train_aucloss = RankingLoss(outputs.detach(), targets)
                train_aucloss = MicroAUC(outputs.detach(),targets)
                epoch_loss = epoch_loss + loss
                epoch_auc = epoch_auc + train_aucloss
            # finall_train_loss = (epoch_loss / len(train_dataloader)).cpu().item()
            finall_train_loss = (epoch_loss / len(train_dataloader)).item()
            finall_train_auc = (epoch_auc / len(train_dataloader))
            print("epoch: ", epoch, ", training macro-auc : ", finall_train_auc, ", train_loss", finall_train_loss)
            #svd decomposition
            regularize_weights_svd(model,svd_theta,svd_lambda,epoch,configs)

            # compute bound
            #if epoch > 150  and epoch % 10 ==0 :
            if epoch >= 0 and epoch % 10 == 0:
                main_r,bound = compute_bound_value(finall_train_loss,dataset,model,svd_theta,new_delta,bound_type="pa_new")
                #bound = compute_bound_value_initial_pa(empirical_loss=finall_train_loss, dataset=dataset,model=model,new_delta=0.01)
                #bound = compute_bound_value_u1 (empirical_loss = finall_train_loss, dataset=dataset,model=model, svd_theta=svd_theta, new_delta=0.01)
                print("epoch:",epoch,",the value of bound is ", bound.item())
                #with open(os.path.join(log_dataset, "train_value_bound_u1_300.txt"), mode="a") as f1:
                with open(os.path.join(log_dataset, "train_value_bound_new_pa.txt"), mode="a") as f1:
                    f1.write("epoch :"+str(epoch) +", the value of bound is "+  str(bound.item())+ "  main_r: "+str(main_r.item())+"\n")
                   

            # train_loss.append((epoch, (epoch_loss / len(train_dataloader)).cpu().item(), train_rankingloss))
            train_loss.append((epoch, (epoch_loss / len(train_dataloader)).item(), train_aucloss))
 #           np.savetxt(os.path.join(log_dataset, train_log_name), train_loss, fmt=['%d', '%.4f', '%.4f'])
 #       print('{}_{} train_loss/RankingLoss in {}:{}'.format(args.dataset, count, args.mode, finall_train_loss))
        print('Training Finish.')
        state = {
            'net': model.state_dict()
        }
        torch.save(state, os.path.join(check, '%s.pth' % save_name))

        print('Testing Beginning.')

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
     #   np.savetxt(os.path.join(log_dataset, test_log_name), test_meature, fmt=['%.4f', '%.4f'])

        del model

    test_performances = np.stack(test_performances)
    mean, std = np.mean(test_performances), np.std(test_performances)
    #with open(os.path.join(log_dataset, "overall_test_performance_u1_300.txt"), mode="w+") as f:
    with open(os.path.join(log_dataset, "overall_test_performance_pa_new.txt"), mode="w+") as f:
        f.write("mean:" + str(np.round(mean, 4)) + "\n")
        f.write("std: " + str(np.round(std, 4)))

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config_from_yaml(args.config)
    args_list = mix_config_parser(args, config)
    
    for args in args_list:
        main(args)

