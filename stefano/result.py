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

''' Classes to keep track of the results during training.
'''

class Experiment:
    ''' Class that keeps track of all the errors for each epoch
    of a training/testing run
    '''
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
    ''' Class that keep track of all the results of a cross validation 
    experiment. These include the results of the k folds and the result
    of the successive training and testing with the whole training/testing
    dataset

    Each training/testing is stored in an Experiment
    '''

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


        plt.figure(figsize=(10,8))
        plt.plot(errors_train,label='Error training')
        if len(self._kfolds):
            plt.plot(errors_validation,label='Error validation')
        if len(self._train_test):
            plt.plot(errors_test,label='Error test')
        plt.title('Errors train/validation')
        plt.legend()
        plt.grid()
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

    def print_performance_last(self):
        self._experiments[-1].compute_performance()
        
    def compute_errors(self):
        for experiment in self._experiments:
            experiment.compute_errors()

    @property
    def params(self):
        return self._params


