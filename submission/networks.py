import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import KFold
from custom_layers import *

from tqdm import tqdm

'''
In this file there is a collection of the networks used to solve the classification problem.
'''


class SingleCNNLayer(nn.Module):
    ''' Single CNN layer. 2 linear layers
    '''
    
    num_my_conv_layers=1
    num_linear_layers=2
    
    def __init__(self,params):
        super(SingleCNNLayer,self).__init__()
        
        self.params=params
        
        layers=[]
        for i in range(self.num_my_conv_layers): 
            layers+=MyConv1D(*self.params[i]).layers
        
        layers.append(Flatten())
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers]))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers+1]))
        
        self.sequential=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.sequential(x)
        
        return x

class DoubleCNNLayers(nn.Module):
    ''' 2 CNN layer. 2 linear layers
    '''
    
    num_my_conv_layers=2
    num_linear_layers=2
    
    def __init__(self,params):
        super(DoubleCNNLayers,self).__init__()
        
        self.params=params
        
        layers=[]
        for i in range(self.num_my_conv_layers): 
            layers+=MyConv1D(*self.params[i]).layers
        
        layers.append(Flatten())
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers]))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers+1]))
        
        self.sequential=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.sequential(x)
        
        return x


class ThreeCNNLayers(nn.Module):
    ''' 3 CNN layer. 2 linear layers
    '''
    
    num_my_conv_layers=3
    num_linear_layers=2
    
    def __init__(self,params):
        super(ThreeCNNLayers,self).__init__()
        
        self.params=params
        
        layers=[]
        for i in range(self.num_my_conv_layers): 
            layers+=MyConv1D(*self.params[i]).layers
        
        layers.append(Flatten())
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers]))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers+1]))
        
        self.sequential=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.sequential(x)
        
        return x

class FourCNNLayers(nn.Module):
    ''' 4 CNN layer. 2 linear layers
    '''
    
    num_my_conv_layers=4
    num_linear_layers=2
    
    def __init__(self,params):
        super(FourCNNLayers,self).__init__()
        
        self.params=params
        
        layers=[]
        for i in range(self.num_my_conv_layers): 
            layers+=MyConv1D(*self.params[i]).layers
        
        layers.append(Flatten())
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers]))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers+1]))
        
        self.sequential=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.sequential(x)
        
        return x

class TenCNNLayers(nn.Module):
    ''' 10 CNN layer. 2 linear layers
    '''
    
    num_my_conv_layers=10
    num_linear_layers=2
    
    def __init__(self,params):
        super(TenCNNLayers,self).__init__()
        
        self.params=params
        
        layers=[]
        for i in range(self.num_my_conv_layers): 
            layers+=MyConv1D(*self.params[i]).layers
        
        layers.append(Flatten())
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers]))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers+1]))
        
        self.sequential=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.sequential(x)
        
        return x

class FullConnect(nn.Module):
    num_my_conv_layers=0
    num_linear_layers=4
    
    def __init__(self,params):
        super(FullConnect,self).__init__()
        self.dropouts=[0.5,0.5,0.5,0]
        self.params=params
        layers=[]
        layers.append(Flatten())
        for i in range(self.num_linear_layers):
            layers.append(nn.Linear(*self.params[self.num_my_conv_layers+i]))
            layers.append(nn.BatchNorm1d(self.params[self.num_my_conv_layers+i][1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropouts[i]))
        
        self.sequential=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.sequential(x)
        
        return x
        
class FullConv(nn.Module):
    num_my_conv_layers=4
    num_linear_layers=0
    def __init__(self,params):
        super(FullConv,self).__init__()
        
        self.params=params
        layers=[]
        for i in range(self.num_my_conv_layers):
            layers+=MyConv1D(*self.params[i]).layers
        
        layers.append(Flatten())
        self.sequential=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.sequential(x)
        
        return x
        
class ThreeLayers2D(nn.Module):
    
    num_my_conv_layers=2
    num_linear_layers=3
    
    def __init__(self,params):
        super(ThreeLayers2D,self).__init__()
        
        self.params=params
        
        layers=[]
        layers.append(Tensor4D())
        for i in range(self.num_my_conv_layers): 
            layers+=MyConv2D(*self.params[i]).layers
            
        layers.append(Flatten())
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers]))
        layers.append(nn.BatchNorm1d(self.params[self.num_my_conv_layers][1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers+1]))   
        layers.append(nn.BatchNorm1d(self.params[self.num_my_conv_layers+1][1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0))
        
        layers.append(nn.Linear(*self.params[self.num_my_conv_layers+2]))
        
        self.sequential=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.sequential(x)
        
        return x
