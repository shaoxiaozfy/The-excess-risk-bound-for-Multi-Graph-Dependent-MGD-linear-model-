import os,sys
import math
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from torch import optim
from torch.utils.data import DistributedSampler
from evaluation.surrogate_loss2 import SurrogateLossForRankingLoss
from models.linear_model import Linear
from evaluation.metrics import RankingLoss,MicroAUC
from benchmarks.tabular_data_benchmark4 import NFoldTabularDataset
from utils.tools import init_random_seed,generate_default_config
from benchmarks.tabular_data_benchmark4 import *
import numpy as np
from utils.deconstruct_yaml import get_config_from_yaml,mix_config_parser
from utils.get_path import get_project_path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,default="tabular_config.yaml")
# dataset can change e.g 'yeast','scene'
parser.add_argument('--dataset', '-dataset', type=str, default="goemotions",
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
parser.add_argument('--weight_decay', '-weight_decay', type=float, default=0.01,
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
parser.add_argument('--svd_lambda', type=float, default=0.1)
parser.add_argument('--svd_rate', type=int, default=100,
                    help='batch size for one iteration during training')

#pa_rc
def compute_bound_value_initial_pa (dataset,model,new_delta):
    W = model.classifier.weight.data
    X = dataset.X
    Y = dataset.y

    # Y = Y[:,1].reshape(-1,1)

    # print(Y)
    num_instance = X.size(0)
    num_label = Y.size(1)
    # num_label = 1
#    tau_s = 0
    print(num_label,num_instance)
    print(X.size(1))
    # print(X.size(1))
    # num_label = 18
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

    B = 4 * r * M_a / math.sqrt(num_instance) * (b / num_label) * 2
    C = 3 * M_b * math.sqrt(math.log(2/new_delta) / 2/ num_instance) * math.sqrt(a / num_label) * 2

    bound = B + C

    return bound



#pa_lrc
def compute_bound_value_pa (dataset,model,svd_theta,new_delta):
    W = model.classifier.weight.data
    # print(W)
    X = dataset.X
    Y = dataset.y

    # Y = Y[:,1].reshape(-1,1)

    # print(Y)
    num_instance = X.size(0)
    num_label = Y.size(1)
    # num_label = 18
    # num_label = 1
    a = 0
    b = 0
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
        a = a+ 1/m
        b = b+ math.sqrt(1/m)
    
#    U,W_a,Vh = torch.linalg.svd(W)
    # r = 0
    # W += r* torch.eye(min(W.size()))
    W_a = torch.linalg.svdvals(W)

    # M_b = torch.norm(X,p=2) 
    # M_b = 0
    # for i in range(num_instance):
    #     temp = torch.norm(X[i,:],p=2)
    #     if temp >M_b:
    #         M_b = temp
    M_b = torch.max(torch.norm(X,dim=-1,p=2))
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
    a1 = svd_theta  * a / (M_b*M_b * num_instance * num_label)
    a2 = M_a * b * math.sqrt(W_l /num_instance / num_label)
    r = (a1 + a2 ) *2
    main_r = 704 * r
    bound = main_r + delta
    return a1,a2,W_l,r,bound

    # return r,bound




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
    # save_name = "/data/shaoxiao/Desktop/Multi-graph concentration resources/macro-auc-pytorch-main/logs/tabular-Linear/emotions/256/pa/0.01/0.1/checkpoint/emotions_256_pa_0.01_lrc_0.1_10"
    save_name = "/data/shaoxiao/Desktop/Multi-graph concentration resources/macro-auc-pytorch-main/logs/tabular-Linear/" + args.dataset + "/256/pa/" + str(args.weight_decay) + "/checkpoint/"+ args.dataset +"_256_pa_" + str(args.weight_decay)
    # save_name = "/data/shaoxiao/Desktop/Multi-graph concentration resources/macro-auc-pytorch-main/goemotions_bound.txt"
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

    # some parameter
    # svd_rate = 100 # cal500
    # svd_rate = 100 # emotion
    # svd_rate = 100 # image
    # svd_rate = 100  # scene
    # svd_rate = 100 # yeast
    # svd_rate = 100 # corel5k
    # svd_rate = 100 # rcv1subset1_top944
    # svd_rate = 100  # bibtex

#    svd_rate = 60
    # svd_theta = svd_rate * min(dataset.num_class,dataset.feat_dim)/100
   
    if args.model == 'Linear':
        model = Linear(configs).to(configs['device'])

    # checkpoint = torch.load(os.path.join(check, '%s.pth' % save_name),map_location=configs['device'])
    checkpoint = torch.load("/data/shaoxiao/Desktop/Multi-graph concentration resources/macro-auc-pytorch-main/logs/tabular-Linear/goemotions/256/pa/0.001/checkpoint/goemotions_256_pa_0.001.pth",map_location=configs['device'])
    model.load_state_dict(checkpoint["net"])


    # compute bound
    # bound = compute_bound_value_initial_pa(dataset=dataset,model=model,new_delta=0.01)
    # for svd_rate in range(0,110,10):
    #     svd_theta = svd_rate * min(dataset.num_class,dataset.feat_dim)/100
    #     a1,a2,wl,r,bound2 = compute_bound_value_pa(dataset=dataset,model=model,svd_theta=svd_theta,new_delta=0.01)
    #     with open(os.path.join(log_dataset, "pa_bound.txt"), mode="a") as f1:
    #         f1.write("the bound of pa_lrc: "+str(bound2.item())+"\n")
    #         f1.write("the value of r: "+str(r.item())+"\n")
    #         f1.write("the value of al: "+str(a1.item())+"\n")
    #         f1.write("the value of a2: "+str(a2.item())+"\n")
    #         f1.write("the value of wl: "+str(wl)+"\n")
    # with open(os.path.join(log_dataset, "pa_bound.txt"), mode="a") as f1:
    #         f1.write("the bound of pa_rc: "+str(bound.item())+ "\n")
    #         f1.write("the bound of pa_lrc: "+str(bound2.item())+"\n")
    #         f1.write("the value of r: "+str(r.item())+"\n")
    bound = torch.zeros(3)
    a1 = torch.zeros(3)
    bound2 = torch.zeros(3)
    std_max = torch.zeros(3)
    r = torch.zeros(3)
    for count in range(3):
        dataset.cv(test_num=count+1, nfold=nfold)
        bound[count] = compute_bound_value_initial_pa(dataset=dataset.train_dataset,model=model,new_delta=0.01)
        svd_theta = args.svd_rate * min(dataset.num_class,dataset.feat_dim)/100
        a1[count],a2,wl,r[count],bound2[count] = compute_bound_value_pa(dataset=dataset.train_dataset,model=model,svd_theta=svd_theta,new_delta=0.01)
        num_instance = dataset.train_dataset.X.to(configs['device']).size(0)
        num_class = dataset.train_dataset.y.to(configs['device']).size(1)
        outputs = model(dataset.train_dataset.X.to(configs['device']))
        mu = torch.mean(outputs, dim=0)
        mu_repeat = mu.reshape(1, -1).repeat(num_instance, 1)
        std = torch.sum((outputs - mu_repeat)**2,dim=0)/(num_instance-1)
        std_max[count] = torch.max(std)

    with open("/data/shaoxiao/Desktop/Multi-graph concentration resources/macro-auc-pytorch-main/pa_lrc_bound.txt", mode="a") as f1:
        bound_mean = torch.mean(bound)
        bound_std = torch.std(bound)
        a1_mean = torch.mean(a1)
        a1_std = torch.std(a1)
        r_mean = torch.mean(r)
        r_std = torch.std(r)
        bound2_mean = torch.mean(bound2)
        bound2_std = torch.std(bound2)
        std_max_mean = torch.mean(std_max)
        std_max_std = torch.std(std_max)

        f1.write(args.dataset + "\n")
        f1.write("the bound of pa_rc: "+str(bound_mean.item())+"+-"+ str(bound_std.item())+ "\n")
        f1.write("the bound of pa_lrc: "+str(bound2_mean.item())+"+-"+ str(bound2_std.item())+ "\n")
        f1.write("the bound of r: "+str(r_mean.item())+"+-"+ str(r_std.item())+ "\n")
        f1.write("the bound of variance: "+str(std_max_mean.item())+"+-"+ str(std_max_std.item())+ "\n")

        print(str(r_mean.item()) + str(r_std.item()))
        print(a1_mean,a1_std)


    
    # bound = compute_bound_value_initial_pa(dataset=dataset,model=model,new_delta=0.01) # .train_dataset
    # svd_theta = args.svd_rate * min(dataset.num_class,dataset.feat_dim)/100
    # a1,a2,wl,r,bound2 = compute_bound_value_pa(dataset=dataset,model=model,svd_theta=svd_theta,new_delta=0.01)
    # with open(os.path.join(log_dataset, "pa_bound.txt"), mode="a") as f1:
    #     f1.write("the bound of pa_rc: "+str(bound.item())+"\n")
    #     f1.write("the bound of pa_lrc: "+str(bound2.item())+"\n")
    #     f1.write("the value of r: "+str(r.item())+"\n")
    #     f1.write("the value of al: "+str(a1.item())+"\n")
    #     f1.write("the value of a2: "+str(a2.item())+"\n")
    #     f1.write("the value of wl: "+str(wl)+"\n")
    # W = model.classifier.weight.data
    # num_instance = dataset.train_dataset.X.to(configs['device']).size(0)
    # num_class = dataset.train_dataset.y.to(configs['device']).size(1)
    # outputs = model(dataset.train_dataset.X.to(configs['device']))
    # mu = torch.mean(outputs, dim=0)
    # mu_repeat = mu.reshape(1, -1).repeat(num_instance, 1)
    # std = torch.sum((outputs - mu_repeat)**2,dim=0)/(num_instance-1)
    # std_max = torch.max(std)
    # with open(os.path.join(log_dataset, "pa_variance.txt"), mode="a") as f1:
    #     f1.write("the variance of pa_lrc: "+str(std_max.item())+"\n")

    



    








if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config_from_yaml(args.config)
    args_list = mix_config_parser(args, config)
    
    for args in args_list:
        main(args)

