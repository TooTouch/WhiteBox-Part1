import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset

import sys
import os 
import json
import numpy as np

from model import SimpleCNNCIFAR10

from collections import OrderedDict
from functools import partial

# Model Load
cifar10_simplecnn = torch.load('../checkpoint/simple_cnn_cifar10.pth')
model = SimpleCNNCIFAR10()
model.load_state_dict(cifar10_simplecnn['model'])

# Transforms
cifar10_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Data Load
cifar10_test = dset.CIFAR10(root='../dataset/CIFAR10',
                          train=False,
                          transform=cifar10_transform_test,
                          download=True)
# idx2class
cifar10_class2idx = cifar10_test.class_to_idx
cifar10_idx2class = dict(zip(list(cifar10_class2idx.values()), list(cifar10_class2idx.keys())))


def get_features_hook(self, input, output):
    _max = output.data.cpu().numpy().max()
    _min = output.data.cpu().numpy().min()
    np.save(str(self),output.data.cpu().numpy())
    print("self: ",str(self),'max: ',_max,'\nmin: ',_min)

for i,_ in model.named_children():
    if 'fc' not in i:
        for k in range(len(model._modules.get(i))):
            model._modules.get(i)[k].register_forward_hook(get_features_hook)
