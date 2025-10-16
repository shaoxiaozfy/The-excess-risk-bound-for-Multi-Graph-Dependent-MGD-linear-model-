import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

class TabularDataset(Dataset):
    def __init__(self, csv_file_dir, data_name):
        # 读取 CSV 文件

        csv_file_path = os.path.join(csv_file_dir, data_name)

        print('111')
        self.data = pd.read_csv(csv_file_path)
        # self.label = pd.read_csv(csv_labels)
        print('222')
        # 假设最后一列是标签，其余是特征
        # self.features = self.data.iloc[:, :-1].values
        # self.labels = self.data.iloc[:, -1].values
        # self.features = self.data.values
        # self.labels = self.label.values

        # self.feature_num = 0  # 根据实际数据调整
        self.label_num = 0

        if 'eurlex' in csv_file_path:
            # self.feature_num = 186104  # 根据实际数据调整
            self.label_num = 3956
        else:
            print('Not known data name')
        print('.....')
        self.features = self.data.iloc[:, :-self.label_num].values
        self.labels = self.data.iloc[:, -self.label_num:].values


    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引的数据
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label




if __name__ == "__main__":
    # 使用示例
    csv_file_dir = '/home/wuguoqiang/projects/dismec-code/data_after_process/eurlex'
    data_name = 'train.csv'

    # csv_file = 'data.csv'  # 替换为你的 CSV 文件路径
    dataset = TabularDataset(csv_file_dir, data_name)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 遍历数据集
    # for features, labels in dataloader:
        # print(features, labels)

