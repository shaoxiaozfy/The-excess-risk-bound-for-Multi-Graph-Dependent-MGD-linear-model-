################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco, Antonio Carta                                  #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34,resnet101,resnet50,vgg11
class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network

    **Example**::

        >>> from avalanche.models import SimpleCNN
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleCNN(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.net = resnet18()
        self.extractor = nn.Sequential(*list(self.net.children())[:-1])

        self.num_feature = self.net.fc.in_features
        # self.net.fc = nn.Linear(self.num_feature, num_classes)
        self.classifier = nn.Linear(self.num_feature, num_classes)

    def forward(self, x):
        # x = self.net(x)
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        # weights = ResNet34_Weights.IMAGENET1K_V1
        ori_net = resnet34()
        self.extractor = nn.Sequential(*list(ori_net.children())[:-1])

        self.num_feature = ori_net.fc.in_features
        # self.net.fc = nn.Linear(self.num_feature, num_classes)
        self.classifier = nn.Linear(self.num_feature, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = self.act(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        ori_net = resnet50()
        self.extractor = nn.Sequential(*list(ori_net.children())[:-1])

        self.num_feature = ori_net.fc.in_features
        # self.net.fc = nn.Linear(self.num_feature, num_classes)
        self.classifier = nn.Linear(self.num_feature, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = self.act(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        ori_net = resnet101()
        self.extractor = nn.Sequential(*list(ori_net.children())[:-1])

        self.num_feature = ori_net.fc.in_features
        # self.net.fc = nn.Linear(self.num_feature, num_classes)
        self.classifier = nn.Linear(self.num_feature, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = self.act(x)
        return x

class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        ori_net = vgg11()
        a = list(ori_net.children())
        self.features = a[0]
        self.avgpool = a[1]
        self.classifier = a[2][:-1]

        self.num_feature = ori_net.classifier[-1].in_features
        # self.net.fc = nn.Linear(self.num_feature, num_classes)
        self.linear = nn.Linear(self.num_feature, num_classes,bias=True)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.linear(x)
        # x = self.act(x)
        return x

__all__ = ["SimpleCNN", "ResNet18","ResNet34","ResNet50","ResNet101","VGG11"]
