import torch

from avalanche.benchmarks.datasets.multi_label_dataset.voc import MultiLabelVOC
from avalanche.benchmarks.datasets.multi_label_dataset.nus_wide import NUS_WIDE
from avalanche.benchmarks.datasets.multi_label_dataset.coco.mycoco import MultiLabelCOCO
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

usr_root = os.path.expanduser("~")
dataset_root = usr_root + "/data/Datasets/coco2017/"

traindata_root = dataset_root + 'train/data/'
train_annFile = dataset_root + 'annotations/instances_train2017.json'

valdata_root = dataset_root + 'validation/data/'
val_annFile = dataset_root + 'annotations/instances_val2017.json'

def cal_imbalance(dataset_name):
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([300, 300]),
    ])

    if dataset_name == "voc":
        train_dataset = MultiLabelVOC(
            root=usr_root + "/data/Datasets/VOC/",
            year="2012",
            image_set="train",
            transform=train_trans
        )
        val_dataset = MultiLabelVOC(
            root=usr_root + "/data/Datasets/VOC/",
            year="2012",
            image_set="test",
            transform=train_trans
        )
        test_dataset = MultiLabelVOC(
            root=usr_root + "/data/Datasets/VOC/",
            year="2012",
            image_set="val",
            transform=train_trans
        )
    elif dataset_name == "nus-wide":
        train_dataset = NUS_WIDE(imageset="train", transforms=train_trans)
        val_dataset = NUS_WIDE(imageset="val", transforms=train_trans)
        test_dataset = NUS_WIDE(imageset="test", transforms=train_trans)
    elif dataset_name == "coco":
        train_dataset = MultiLabelCOCO(image_set="train", transform=train_trans)
        val_dataset = MultiLabelCOCO(image_set="val", transform=train_trans)
        test_dataset = MultiLabelCOCO(image_set="test", transform=train_trans)
    else:
        raise NotImplementedError
    print("length: ", len(train_dataset)+len(val_dataset)+len(test_dataset))

    true_y = []
    for dataset in [train_dataset,val_dataset,test_dataset]:
        loader = DataLoader(dataset,batch_size=256,num_workers=2)
        for x,y in loader:
            true_y.append(y)
            print(len(true_y))
    true_y = torch.concat(true_y)
    s_pos = torch.count_nonzero(true_y)
    s_neg = true_y.shape[0]*true_y.shape[1] - s_pos

    to = min(s_pos,s_neg)/(true_y.shape[0]*true_y.shape[1])
    imb2 = 1 / to
    imb1 = torch.sqrt(imb2)
    imb3 = imb1*imb2

    print(imb1,imb2,imb3)

if __name__ == '__main__':
    cal_imbalance("nus-wide")