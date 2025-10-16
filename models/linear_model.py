import torch
from torch import nn

from utils.tools import init_random_seed


class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.configs = configs
        self.features = nn.Linear(configs['in_features'], configs['n_hidden'])
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(configs['n_hidden'], configs['num_classes'])

    def forward(self, x):
        out = self.features(x)
        out = self.relu(out)
        out = self.classifier(out)

        return out

    def reset_parameters(self):
        init_random_seed(self.configs['rand_seed'])
        torch.nn.init.normal_(self.features.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.1)


class Linear(nn.Module):
    def __init__(self, configs):
        super(Linear, self).__init__()
        self.configs = configs
        self.classifier = nn.Linear(configs['in_features'], configs['num_classes'])

    def forward(self, x):
        out = self.classifier(x)

        return out

    def reset_parameters(self):
        init_random_seed(self.configs['rand_seed'])
        torch.nn.init.normal_(self.features.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.1)
