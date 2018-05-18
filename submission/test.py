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

import load_script
from params import *
from custom_layers import *
from training import *
from networks import *

###### Parameters  ######
cuda=False
add_noise=False

#########################
##     IMPORTANT       ##
# To run all method:    #
#########################
run_all_methods=False

######  Dataset  ######
train_dataset,train_target=load_script.load_dataset(train=True)
test_dataset,test_target=load_script.load_dataset(train=False)

# Add noise
if add_noise:
    train_ds = []
    train_ds.append(train_dataset.data.numpy())
    for i in range(4):
        train_ds.append(train_ds[0] + np.random.normal(scale = 0.5, size = train_ds[0].shape))

    train_ds = np.asarray(train_ds).reshape(316 * 5, 28, 50)

    train_dataset = Variable(torch.FloatTensor(train_ds))
    train_target = Variable(torch.LongTensor(np.tile(train_target.data.numpy().T, 5).T)) 

# Normalize
mean=train_dataset.mean(0).view(1,28,50)
std=train_dataset.std(0).view(1,28,50)
train_dataset=(train_dataset-mean)/std
test_dataset=(test_dataset-mean)/std


######  Models to train  #######

# Three CNN Layers - 2 Linears #
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
three_layers_label='Three CNN Layers - 2 Linears'
model1=(three_layers,three_layers_label)

# Two CNN Layers - 2 Linears #
net_type=DoubleCNNLayers
optimizer_type=optim.Adam
criterion_type=nn.CrossEntropyLoss
network_params=NetworkParams(conv_filters=[28,28],conv_kernels=[3,3],
                             linear_filters=[200,2],
                             dropout_rate=[0.8,0.8],batch_norm=True,conv1D=True)
optimizer_params=OptimizerParams()
train_params=TrainParams(max_epoch=300,mini_batch_size=79)


two_layers=Params(net_type,optimizer_type,criterion_type,network_params=network_params,
                  optimizer_params=optimizer_params,train_params=train_params,cuda=cuda,
                  plot=False)
two_layers_label='Two CNN Layers - 2 Linears'
model2=(two_layers,two_layers_label)


# Fully convolutional network - 4 CNN layers #
net_type=FullConv
optimizer_type=optim.Adam
criterion_type=nn.CrossEntropyLoss
network_params=NetworkParams(conv_filters=[28,14,7,2],conv_kernels=[5,9,11,20],dropout_rate=[0.8,0.8,0.5,0],\
                             stride=1,dilation=[3,1,1,1]) 
optimizer_params=OptimizerParams()
train_params=TrainParams(max_epoch=1500,mini_batch_size=2*79)


full_conv=Params(net_type,optimizer_type,criterion_type,network_params=network_params,\
              optimizer_params=optimizer_params,train_params=train_params, cuda=cuda, plot=False)
full_conv_label='Four CNN Layers - Fully Convolutional Network'
model3=(full_conv,full_conv_label)

# Fully connected network - 4 linear layers #
net_type=FullConnect
optimizer_type=optim.Adam
criterion_type=nn.CrossEntropyLoss
network_params=NetworkParams(linear_filters=[500,200,50,2],conv1D=False)
optimizer_params=OptimizerParams()
train_params=TrainParams(max_epoch=1000,mini_batch_size=79*2)

full_conn=Params(net_type,optimizer_type,criterion_type,network_params=network_params,\
              optimizer_params=optimizer_params,train_params=train_params,cuda=cuda,plot=False)
full_conn_label='Four Linear Layers - Fully Connected Network'
model4=(full_conn,full_conn_label)

# Three CNN 2D layers - 3 linear layers #
net_type=ThreeLayers2D
optimizer_type=optim.Adam
criterion_type=nn.CrossEntropyLoss
network_params=NetworkParams(conv_filters=[5,10], conv_kernels=[(1,5),(28,3)],
                             linear_filters=[100, 20, 2],
                             dropout_rate=0.8,batch_norm=True,conv1D=False)
optimizer_params=OptimizerParams()
train_params=TrainParams(max_epoch=1000,mini_batch_size=79*2)

cnn2D=Params(net_type,optimizer_type,criterion_type,network_params=network_params,
              optimizer_params=optimizer_params,train_params=train_params,cuda=cuda,plot=False)
cnn2D_label='Three CNN2D Layers - 3 Linear Layers'
model5=(cnn2D,cnn2D_label)

# Declaring the list with all the models
if run_all_methods:
    models=[model1,model2,model3,model4,model5]
else:
    models=[model1,model2,model4]

#######  Training all the models  #######
for model,label in models:
    print("\n\nTraining with:",label,'\n')
    cv=CrossValidation(k=4,train_dataset=train_dataset,test_dataset=test_dataset,
                   train_target=train_target,test_target=test_target,cuda=cuda)
    cv(model,repetitions=1,cross_validation=False,repetitions_test=4)
