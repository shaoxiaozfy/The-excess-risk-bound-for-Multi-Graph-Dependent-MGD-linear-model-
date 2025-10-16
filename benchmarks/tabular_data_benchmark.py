import torch
import torch.utils.data as data
import scipy.io as scio
import os
from sklearn.model_selection import train_test_split
from utils.get_path import get_project_path

dataset_names = ["bibtex","CAL500","corel5k","delicious","emotions","enron","Image","rcv1subset1","scene","yeast"]

class NFoldTabularDataset:
    def __init__(self, dataset_name='' , datadir= get_project_path() + "benchmarks/tabular_datasets/tabular_data_files", configs={}, nfold=3):
        self.datadir = datadir
        self.dataset_name = dataset_name
        self.datafile = os.path.join(datadir, self.dataset_name, self.dataset_name + '.mat')

        # Load data
        self.dtype = configs['dtype']
        self.data_standardizing = configs['data_standardizing']
        self.eps = configs['eps']
        self._data(configs['split'], configs['shuffle'], configs['rand_seed'], nfold)
        self.rand_seed = configs['rand_seed']
        self.feat_dim = self.X.size(1)
        self.num_class = self.Y.size(1)


    def _data(self, dataSplit, shuffle=False, random_state=None, nfold=3):
        self._data_loading()
        self._data_preprocess()
        self._data_split(dataSplit, shuffle, random_state, nfold)

    def _data_loading(self):
        data = scio.loadmat(self.datafile)
        self.X = torch.from_numpy(data['data'].astype(float)).type(self.dtype)
        self.Y = torch.from_numpy(data['target']).type(self.dtype)
        self.Y[self.Y < 0] = 0
        # self.Y[self.Y < 1] = -1
        # self.Y_test[self.Y_test < 1] = -1

    def _data_preprocess(self):
        # max_X = torch.max(self.X, dim=0, keepdim=True)[0]
        # min_X = torch.min(self.X, dim=0, keepdim=True)[0]
        # self.X = (self.X - min_X) / (max_X - min_X + self.eps)
        mu_train = torch.mean(self.X, dim=0)
        std_train = torch.std(self.X, dim=0)
        self.X = (self.X - mu_train) / std_train

        # mu_test = torch.mean(self.X_test, dim=0)
        # std_test = torch.std(self.X_test, dim=0)
        # self.X_test = (self.X_test - mu_test) / std_test

    def _data_split(self, dataSplit, shuffle=False, random_state=None, nfold=3):
        file_random = '{:d}'.format(random_state)
        for count in range(1, nfold + 1):
            sub_file = os.path.join(self.datadir, self.dataset_name,
                                    self.dataset_name + '_' + file_random + '_' + str(count) + '.mat')
            if not os.path.exists(sub_file):
                if count < nfold:
                    self.sub_x, self.re_x, self.sub_y, self.re_y = train_test_split(self.X, self.Y,
                                                                                    train_size=1 / (nfold - count + 1),
                                                                                    random_state=random_state,
                                                                                    shuffle=shuffle)
                    scio.savemat(sub_file, {'X': self.sub_x.numpy(), 'Y': self.sub_y.numpy()})
                    self.X = self.re_x
                    self.Y = self.re_y
                else:
                    scio.savemat(sub_file, {'X': self.X.numpy(), 'Y': self.Y.numpy()})

            else:
                break

    def cv(self, test_num, nfold):
        print(nfold)
        file_random = '{:d}'.format(self.rand_seed)
        first_train = False
        for count in range(1, nfold + 1):
            sub_file = os.path.join(self.datadir, self.dataset_name,
                                    self.dataset_name + '_' + file_random + '_' + str(count) + '.mat')
            if count == test_num:
                test_data = scio.loadmat(sub_file)
                self.X_test = torch.from_numpy(test_data['X'].astype(float)).type(self.dtype)
                self.Y_test = torch.from_numpy(test_data['Y']).type(self.dtype)
            else:
                train_data = scio.loadmat(sub_file)
                if first_train == False:
                    self.X_train = torch.from_numpy(train_data['X'].astype(float)).type(self.dtype)
                    self.Y_train = torch.from_numpy(train_data['Y']).type(self.dtype)
                    first_train = True
                else:
                    self.X_sub = torch.from_numpy(train_data['X'].astype(float)).type(self.dtype)
                    self.Y_sub = torch.from_numpy(train_data['Y']).type(self.dtype)
                    self.X_train = torch.cat((self.X_train, self.X_sub), dim=0)
                    self.Y_train = torch.cat((self.Y_train, self.Y_sub), dim=0)

        self.test_dataset = sub_dataset(self.X_test, self.Y_test)
        self.train_dataset = sub_dataset(self.X_train, self.Y_train)


class sub_dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        each_x = self.X[index]
        each_y = self.y[index]
        return each_x, each_y

