import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import KFold
from result import *

from tqdm import tqdm

class CrossValidation:
    ''' Main class for training a network and for tuning the parameters.

    It can perform the cross validation and/or training the network on the 
    full training dataset.

    #####
    Usage example:
    cv=CrossValidation(k=4,train_dataset=train_dataset,test_dataset=test_dataset,
                   train_target=train_target,test_target=test_target,cuda=True)

    net_type=ThreeCNNLayers
    optimizer_type=optim.Adam
    criterion_type=nn.CrossEntropyLoss
    network_params=NetworkParams(conv_filters=[14,28,42],conv_kernels=[3,3,3],
                                 linear_filters=[200,2],
                                 dropout_rate=dropout,batch_norm=True,conv1D=True)
    optimizer_params=OptimizerParams()
    train_params=TrainParams(max_epoch=300,mini_batch_size=79)


    params=Params(net_type,optimizer_type,criterion_type,network_params=network_params,
                  optimizer_params=optimizer_params,train_params=train_params,cuda=True,plot=False)

    cv(params,repetitions=1,cross_validation=True,repetitions_test=4)
    #####




    '''
    
    def __init__(self,train_dataset,test_dataset,train_target,test_target,k=4,cuda=False):
        self._k=k
        self._kfold=KFold(n_splits=self.k,shuffle=True)
        self._train_dataset=train_dataset
        self._train_target=train_target
        self._test_dataset=test_dataset
        self._test_target=test_target
        self._cuda=cuda
        
    def __call__(self,params,repetitions=5,repetitions_test=4,cross_validation=True):
        ''' Function that executes the cross validation/training

        @args:
        - params: an instance of Params
        - repetitions: number of time to repeat the full cross validation
        - repetitions_test: average the training testing on multiple repetitions. Useful to reduce the noise
        - cross_validation: wheter to perform the cross validation


        '''
        print(params.network)
        self._result=Result(params)
        for i in range(repetitions):
            self.result.start(cross_validation=True)
            if cross_validation:
                self.train_validate_network(params,self.result)
            for j in range(repetitions_test):
                self.train_test_network(params,self.result)
            
            if params.plot:
                self.result.plot_last()
            else:
                self.result.print_performance_last()
        
        
    def train_validate_network(self,params,result):
        for train_indexes,validation_indexes in self._kfold.split(self._train_dataset):
            
            result.new_kfold()

            for param in params.network.parameters():
                param.data.normal_(0, params.train_params.weights_initialization)
            
            train_indexes,validation_indexes=torch.LongTensor(train_indexes),torch.LongTensor(validation_indexes)
            train=self._train_dataset[train_indexes]
            train_target=self._train_target[train_indexes]
            validation=self._train_dataset[validation_indexes]
            validation_target=self._train_target[validation_indexes]

            for epoch in tqdm(range(params.train_params.max_epoch)):

                self.train_epoch(params,train,train_target,result)
                self.validate_epoch(params,validation,validation_target,result)
                
             
    
    def train_test_network(self,params,result):
        result.new_train_test()
        
        for param in params.network.parameters():
            param.data.normal_(0, params.train_params.weights_initialization)
        
        for epoch in tqdm(range(params.train_params.max_epoch)):
            self.train_epoch(params,self._train_dataset,self._train_target,result)
            self.validate_epoch(params,self._test_dataset,self._test_target,result)
                
    def train_epoch(self,params,train_dataset,target,result):
        # Set training True
        params.network.train(True)
        
        # Randomize training dataset
        if params.train_params.randomize_training_dataset:
            random_permutation=torch.randperm(train_dataset.shape[0])
        else:
            random_permutation=torch.arange(train_dataset.shape[0])

        train_dataset_shuffled=train_dataset[random_permutation]
        target_shuffled=target[random_permutation]

        if self.cuda:
            train_dataset_shuffled=train_dataset_shuffled.cuda()
            target_shuffled=target_shuffled.cuda()
        
        
        # Iterate on the dataset
        total_loss=0
        output_target=torch.zeros(target_shuffled.shape[0])
        for b in range(0,train_dataset_shuffled.shape[0],params.train_params.mini_batch_size):

            train_element=train_dataset_shuffled.narrow(0,b,params.train_params.mini_batch_size)
            target_element=target_shuffled.narrow(0,b,params.train_params.mini_batch_size)

            params.optimizer.zero_grad()

            out=params.network(train_element)
            output_target[b:b+params.train_params.mini_batch_size]=(out[:,1]>out[:,0]).data

            loss=params.criterion(out,target_element)
            loss.backward()
            params.optimizer.step()
            total_loss+=loss.data[0]
        
        error_train=np.sum(list(output_target.cpu().long()!=target_shuffled.cpu().long().data))/target_shuffled.shape[0]    
        
        result.error_train_epoch(error_train,total_loss)
    
    
    def validate_epoch(self,params,validation_dataset,target,result):
        params.network.train(False)

        if self.cuda:
            validation_dataset=validation_dataset.cuda()
            target=target.cuda()

        out=params.network.forward(validation_dataset)
        output_validation=(out[:,1]>out[:,0]).data.long()
        error_validation=np.sum(list(output_validation.cpu().long()!=target.cpu().long().data))/target.shape[0]
        #  print(error_validation)
        
        result.error_validation_epoch(error_validation)
        
    @property
    def cuda(self):
        return self._cuda

    @property
    def k(self):
        return self._k
    
    @property
    def result(self):
        return self._result
        

