import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from Paper_global_vars import global_vars

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])
 
#  选择数据集:
root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")
testset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
# 加载数据:
valid_data = DataLoader(dataset=testset, batch_size=global_vars.batch_size, shuffle=False)