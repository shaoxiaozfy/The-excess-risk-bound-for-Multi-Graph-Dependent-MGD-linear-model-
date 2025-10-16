import os

import torch
from torchvision.datasets import CIFAR10
from benchmarks.image_datasets.voc.voc import MultiLabelVOC
from benchmarks.image_datasets.nus_wide.nus_wide import NUS_WIDE
from benchmarks.image_datasets.coco.mycoco import MultiLabelCOCO
from torchvision.transforms import transforms

def multi_label_batchlearning_benchmark(args):
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize([300,300]),
        transforms.Resize([224, 224]),
        ]
    )
    if args.dataset_name == "voc":
        train_dataset = MultiLabelVOC(
            root= args.default_dataset_location + "/Datasets/VOC/",
            year="2012",
            image_set="train",
            transform=train_trans
        )
        val_dataset = MultiLabelVOC(
            root=args.default_dataset_location + "/Datasets/VOC/",
            year="2012",
            image_set="test",
            transform=train_trans
        )
        test_dataset = MultiLabelVOC(
            root= args.default_dataset_location + "/Datasets/VOC/",
            year="2012",
            image_set="val",
            transform=train_trans
        )
    elif args.dataset_name == "nus-wide":
        train_dataset = NUS_WIDE(root=args.default_dataset_location + "/Datasets/NUS-WIDE/", imageset="train",
                                 transforms=train_trans)
        val_dataset = NUS_WIDE(root=args.default_dataset_location + "/Datasets/NUS-WIDE/", imageset="val",
                               transforms=train_trans)
        test_dataset = NUS_WIDE(root=args.default_dataset_location + "/Datasets/NUS-WIDE/", imageset="test",
                                transforms=train_trans)
    elif args.dataset_name == "coco":
        train_dataset = MultiLabelCOCO(root=args.default_dataset_location + "/Datasets/coco2017/", image_set="train",
                                       transform=train_trans)
        val_dataset = MultiLabelCOCO(root=args.default_dataset_location + "/Datasets/coco2017/", image_set="val",
                                     transform=train_trans)
        test_dataset = MultiLabelCOCO(root=args.default_dataset_location + "/Datasets/coco2017/", image_set="test",
                                      transform=train_trans)
    else:
        raise NotImplementedError

    print(len(train_dataset),len(val_dataset),len(test_dataset))
    # benchmark = create_multi_dataset_generic_benchmark(
    #     train_datasets=(train_dataset,),
    #     test_datasets=(test_dataset,),
    #     other_streams_datasets={"val":(val_dataset,)},
    #     task_labels=[0,]
    # )
    #
    # return benchmark
    return train_dataset,val_dataset,test_dataset


def get_onehot(target):
    one_hot = torch.zeros(10)
    one_hot[target] = 1
    return one_hot


class OneCIFAR10(CIFAR10):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 ):
        super().__init__(root, train, transform, target_transform, download)
        self.targets = [get_onehot(e) for e in self.targets]


def cifar10_batchlearning_benchmark(args):
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        ])

    train_dataset = OneCIFAR10(args.default_dataset_location+"cifar10",train=True, transform=train_trans,download=True)
    test_dataset = OneCIFAR10(args.default_dataset_location+"cifar10",train=False, transform=train_trans,download=True)


    print(len(train_dataset),len(test_dataset))

    return train_dataset,test_dataset
