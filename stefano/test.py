# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import KFold

from tqdm import tqdm

import load_script
from params import *
from custom_layers import *
from training import *
from networks import *

# Parameters
cuda=False

# Dataset
train_dataset,train_target=load_script.load_dataset(train=True)
test_dataset,test_target=load_script.load_dataset(train=False)

mean=train_dataset.mean(0).view(1,28,50)
std=train_dataset.std(0).view(1,28,50)
train_dataset=(train_dataset-mean)/std
test_dataset=(test_dataset-mean)/std


# Models to train
net_type=ThreeCNNLayers
optimizer_type=optim.Adam
criterion_type=nn.CrossEntropyLoss
network_params=NetworkParams(conv_filters=[28,28,28],conv_kernels=[3,3,3],
                             linear_filters=[200,2],
                             dropout_rate=[0.8,0.8,0.8],batch_norm=True,conv1D=True)
optimizer_params=OptimizerParams()
train_params=TrainParams(max_epoch=300,mini_batch_size=79)


three_layers=Params(net_type,optimizer_type,criterion_type,network_params=network_params,
                  optimizer_params=optimizer_params,train_params=train_params,cuda=cuda,
                  plot=False)


models=[three_layers]


# Training all the models
for model in models:
    cv=CrossValidation(k=4,train_dataset=train_dataset,test_dataset=test_dataset,
                   train_target=train_target,test_target=test_target,cuda=cuda)
    cv(model,repetitions=1,cross_validation=False,repetitions_test=4)
