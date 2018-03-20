import torch
import numpy as np
from torch.autograd import Variable


# Import default load script from parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import dlc_bci

def load_dataset(train=True):
    dataset,target=dlc_bci.load('../data',train=train)

    new_target=torch.ones(target.shape[0],2)
    new_target[:,0][target==1]=0
    new_target[:,1][target==0]=0
    dataset=Variable(dataset)
    target_dataset=Variable(new_target)

    return dataset,target_dataset
