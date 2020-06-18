import os
import sys
import numpy as np
import torch
import argparse

from tqdm import tqdm
from torchvision import transforms

sys.path.append("..")
from utils.dataset import *

parser = argparse.ArgumentParser(description="Calculates mean and standard deviation of CheXpert.")
parser.add_argument("path", help="Path to CheXpert data.")
args = parser.parse_args()
path = args.path

# compute stats for train set
trainset = CheXpertDataset(path, img_size=256, train=True,
                           transforms=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=1)
train_mean = 0.0
train_std = 0.0
for img,label in tqdm(trainloader):
    img = img.numpy()
    train_mean += np.mean(img)
    train_std += np.std(img)
train_mean /= len(trainset)
train_std /= len(trainset)
print(f'Training mean: {train_mean}')
print(f'Training std: {train_std}')

# compute stats for test set
testset = trainset = CheXpertDataset(path, img_size=256, train=False,
                                     transforms=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=1)
test_mean = 0.0
test_std = 0.0
for img,label in tqdm(testloader):
    img = img.numpy()
    test_mean += np.mean(img)
    test_std += np.std(img)
test_mean /= len(testset)
test_std /= len(testset)
print(f'Testing mean: {test_mean}')
print(f'Testing std: {test_std}')

# Outputs for 256x256
# Training mean: 0.5028450971783546
# Training std: 0.2900499572707849
# Testing mean: 0.5027450731931589
# Testing std: 0.2901882898603749