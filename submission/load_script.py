import torch
import numpy as np
from torch.autograd import Variable

import dlc_bci

def load_dataset(train=True):
    ''' Return the standard version of the dataset
    '''
    dataset,target=dlc_bci.load('../data',train=train)

    dataset=Variable(dataset)
    target_dataset=Variable(target)

    return dataset,target_dataset

def load_dataset_1000hz(train=True):
    ''' Return the 1000hz version of the dataset

    The dataset is downsampled. The original size of each sample is 28x500.
    The dataset that is returned is sampled with dilation 10 and stride 1 from the original dataset.
    The new size of each sample is 28x50 and there will be 10 times more data.
    '''
    dataset,target=dlc_bci.load('../data',train=train,one_khz=True)


    downsampled_dataset=[]
    downsampled_target=[]
    for i in range(10):
        indexes=range(i,dataset.shape[2],10)
        downsampled_dataset.append(dataset[:,:,indexes])
        downsampled_target.append(target)

    downsampled_dataset=torch.cat(downsampled_dataset)
    downsampled_target=torch.cat(downsampled_target)

    downsampled_dataset=Variable(downsampled_dataset)
    downsampled_target=Variable(downsampled_target)

    return downsampled_dataset,downsampled_target
