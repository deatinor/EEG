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

class Experiment:
    def __init__(self):
        self._errors_train=[]
        self._loss=[]
        self._errors_validation=[]
    
    def error_train_epoch(self,error_train,loss):
        self._errors_train.append(error_train)
        self._loss.append(loss)
        
    def error_validation_epoch(self,error_validation):
        self._errors_validation.append(error_validation)

class ExperimentCrossValidation:
    def __init__(self):
        self._kfolds=[]
        self._train_test=[]
        self._last_experiment=None
        self._performance_train=0
        self._performance_validation=0
        self._performance_test=0

    def new_kfold(self):
        self._kfolds.append(Experiment())
        self._last_experiment=self._kfolds[-1]

    def new_train_test(self):
        self._train_test.append(Experiment())
        self._last_experiment=self._train_test[-1]

    def error_train_epoch(self,error_train,loss):
        self._last_experiment.error_train_epoch(error_train,loss)

    def error_validation_epoch(self,error_validation):
        self._last_experiment.error_validation_epoch(error_validation)

    def compute_errors(self):
        global_loss=[]
        errors_train=[]
        errors_validation=[]
        errors_test=[]
        for experiment in self._kfolds:
            global_loss.append(experiment._loss)
            errors_train.append(experiment._errors_train)
            errors_validation.append(experiment._errors_validation)
        for experiment in self._train_test:
            if not len(self._kfolds):
                errors_train.append(experiment._errors_train)
            errors_test.append(experiment._errors_validation)

        errors_train=np.mean(np.array(errors_train),0)
        if len(self._kfolds):
            global_loss=np.mean(np.array(global_loss),0)
            errors_validation=np.mean(np.array(errors_validation),0)
        if len(self._train_test):
            errors_test=np.mean(np.array(errors_test),0)

        return global_loss,errors_train,errors_validation,errors_test

    def compute_performance(self):
        global_loss,errors_train,errors_validation,errors_test=self.compute_errors()

        error_test_size=errors_train.shape[0]
        performance_index=int(0.75*error_test_size)
        self._performance_train=np.mean(errors_train[performance_index:])
        print('Performance train:',self.performance_train)
        if len(self._kfolds):
            self._performance_validation=np.mean(errors_validation[performance_index:])
            print('Performance validation:',self.performance_validation)
        if len(self._train_test):
            self._performance_test=np.mean(errors_test[performance_index:])
            print('Performance test:',self.performance_test)


    def plot(self):
        global_loss,errors_train,errors_validation,errors_test=self.compute_errors()


        plt.plot(errors_train,label='Error training')
        if len(self._kfolds):
            plt.plot(errors_validation,label='Error validation')
        if len(self._train_test):
            plt.plot(errors_test,label='Error test')
        plt.title('Errors train/validation')
        plt.legend()
        plt.show()

        self.compute_performance()

        #  error_test_size=errors_test.shape[0]
        #  performance_index=int(0.75*error_test_size)
        #  print('Performance train:',np.mean(errors_train[performance_index:]))
        #  print('Performance validation:',np.mean(errors_validation[performance_index:]))
        #  print('Performance test:',np.mean(errors_test[performance_index:]))

    @property
    def performance_train(self):
        return self._performance_train

    @property
    def performance_validation(self):
                return self._performance_validation

    @property
    def performance_test(self):
        return self._performance_test

class Result:
    def __init__(self,params):
        self._experiments=[]

    def start(self,cross_validation=True):
        if cross_validation:
            self._experiments.append(ExperimentCrossValidation())
        else:
            self._experiments.append(Experiment())

    def new_kfold(self):
        self._experiments[-1].new_kfold()

    def new_train_test(self):
        self._experiments[-1].new_train_test()

    def error_train_epoch(self,error_train,loss):
        self._experiments[-1].error_train_epoch(error_train,loss)

    def error_validation_epoch(self,error_validation):
        self._experiments[-1].error_validation_epoch(error_validation)

    def plot(self):
        for experiment in self._experiments:
            experiment.plot()

    def plot_last(self):
        self._experiments[-1].plot()
        
    def compute_errors(self):
        for experiment in self._experiments:
            experiment.compute_errors()

    @property
    def params(self):
        return self._params



class CrossValidation:
    
    def __init__(self,train_dataset,test_dataset,train_target,test_target,k=4,cuda=False):
        self._k=k
        self._kfold=KFold(n_splits=self.k,shuffle=True)
        self._train_dataset=train_dataset
        self._train_target=train_target
        self._test_dataset=test_dataset
        self._test_target=test_target
        self._cuda=cuda
        
    def __call__(self,params,repetitions=5,repetitions_test=4,cross_validation=True):
        self._result=Result(params)
        for i in range(repetitions):
            self.result.start(cross_validation=True)
            print('Repetition',i)
            if cross_validation:
                self.train_validate_network(params,self.result)
            for j in range(repetitions_test):
                self.train_test_network(params,self.result)
            
            if params.plot:
                self.result.plot_last()
        
        
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
        


class Train:
    ''' Class to train a network
    
    Usage example:
    net_type=ThreeLayers
    optimizer_type=optim.Adam
    criterion_type=nn.CrossEntropyLoss
    network_params=NetworkParams(linear_filters=[200,2])
    optimizer_params=OptimizerParams()
    train_params=TrainParams(max_epoch=400)


    params=Params(net_type,optimizer_type,criterion_type,network_params=network_params,
                  optimizer_params=optimizer_params,train_params=train_params,)

    train=Train()
    train(params)
    
    '''
    
    def __init__(self,train_dataset,test_dataset,train_target,test_target):
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.train_target=train_target
        self.test_target=test_target

    
    def __call__(self,params,repetitions=5):
        epochs=[]
        errors_max=[]
        errors_mean=[]
        for i in range(repetitions):
            for param in params.network.parameters():
                param.data.normal_(0, params.train_params.weights_initialization)
            print('Repetition',i)
            epoch,error_max,error_mean=self.train_test_network(params)
            epochs.append(epoch)
            errors_max.append(error_max)
            errors_mean.append(error_mean)
        
        return epochs,errors_max,errors_mean
    
    
    def train_test_network(self,params):
    
        errors_train=[]
        errors_test=[]
        
        
        for epoch in tqdm(range(params.train_params.max_epoch)):
            
            total_loss,output_train,error_train=self.train_epoch(params)
            output_test,error_test=self.test_epoch(params)
        
        
            errors_train.append(error_train)
            errors_test.append(error_test)
            
            if epoch%10==0 and params.verbose:
                print('Epoch:',epoch,'Loss:',total_loss,'Correct:',str(error_train*100)[:5]+"%",
                     'Correct test:',str(error_test*100)[:5]+"%")
                
        if params.plot:
            
            print('Performance:',np.mean(errors_test[300:]))
            plt.plot(list(range(epoch+1)),errors_train,label='Errors train')
            plt.plot(list(range(epoch+1)) ,errors_test,label='Errors test')
            
            plt.legend()
            plt.show()
                 
        return epoch,np.max(errors_test),np.mean(errors_test[300:])
    
    
    
    def train_epoch(self,params):
        
        # Set training True
        params.network.train(True)
        
        # Randomize training dataset
        if params.train_params.randomize_training_dataset:
            random_permutation=torch.randperm(self.train_dataset.shape[0])
        else:
            random_permutation=torch.arange(self.train_dataset.shape[0])

        train_dataset_shuffled=self.train_dataset[random_permutation]
        target_shuffled=self.train_target[random_permutation]
        
        
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
            
        error_train=np.sum(list(output_target.long()==target_shuffled.data))/self.train_target.shape[0]    
        
        return total_loss,output_target,error_train
    
    def test_epoch(self,params):
        params.network.train(False)
        output_test=torch.zeros(self.test_target.shape[0])
        out=params.network.forward(self.test_dataset)
        output_test=(out[:,1]>out[:,0]).data.long()
        error_test=np.sum(list(output_test==self.test_target.data))/self.test_target.shape[0]
        
        return output_test,error_test




