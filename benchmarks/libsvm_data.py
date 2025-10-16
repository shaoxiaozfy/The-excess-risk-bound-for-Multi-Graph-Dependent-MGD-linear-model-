import numpy as np
import torch
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class LIBSVM_Dataset(Dataset):
    def __init__(self, file_path, scaler=None):
        # 读取 LIBSVM 数据
        self.X, self.y = load_svmlight_file(file_path, n_features=186104, multilabel=True)
        # self.X, self.y = load_svmlight_file(file_path, n_features=5000, multilabel=True)
        # 转换为 NumPy 数组
        self.X = self.X.toarray()  # 转换为稠密数组
        # print(self.X.shape)
        # print(self.y)
        # self.y = np.array(self.y)

        label_num = 3956
        tmp_y = []
        for labels_tuple in self.y:
            labels = [0] * label_num
            for idx in labels_tuple:
                labels[int(idx) - 1] = 1
            tmp_y.append(labels)
        self.y = np.array(tmp_y)


        # StandardNormalization
        if scaler is None:
            # for train data
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            # for test data
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
    def __len__(self):
        # 返回数据集大小
        return len(self.y)

    def __getitem__(self, idx):
        # 获取特定索引的数据和标签
        feature = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return feature, label

    def get_scaler(self):
        return self.scaler




if __name__ == '__main__':
    # 使用示例
    file_path = '/home/wuguoqiang/projects/dismec-code/eurlex/test-remapped-tfidf-relabeled.txt'  # 替换为你的 LIBSVM 文件路径
    dataset = LIBSVM_Dataset(file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 遍历数据集
    # for features, labels in dataloader:
    #     print(features, labels)