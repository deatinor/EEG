import numpy as np

class LayerParams:
    ''' Class that defines the parameter for a generic nn.Module
        
        The parameters are stored as a list
        
        Params:
            params (list): a list of the parameters of a nn.Module
    '''
    
    def __init__(self,*args):
        self._params=[*args]
        
    def __str__(self):
        return str(self._params)
    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self,key):
        return self._params[key]
    
    @property
    def params(self):
        return self._params

class LayersParams:
    ''' Class that stores a list of LayerParams
    
        Params:
            layers_params (list): a list of LayerParams
    '''
    
    def __init__(self,*args):
        self._layers_params=[*args]
        
    def __getitem__(self,key):
        return self._layers_params[key]
    
    @property
    def layers_params(self):
        return self._layers_params

    def __str__(self):
        return_str='Network:\n\n'
        for i in self._layers_params:
            return_str+=str(i)
            return_str+='\n'
        return return_str

    def __repr__(self):
        return str(self)

class TrainParams:
    ''' Parameters used to train the network
    
        Methods:
            params (list): it returns the list of parameters
    
        Params:
            mini_batch_size (int)
            max_epoch (int)
            weights_initialization (float)
            randomize_training_dataset (Bool): each batch is randomly selected from the training dataset
    '''
    
    def __init__(self,mini_batch_size=79,
                 max_epoch=1000,
                 weights_initialization=0.02,
                 randomize_training_dataset=True):
        self._mini_batch_size=mini_batch_size
        self._max_epoch=max_epoch
        self._weights_initialization=weights_initialization
        self._randomize_training_dataset=randomize_training_dataset
        
    def params(self):
        return [self._mini_batch_size,self._max_epoch,self._max_epoch,self._randomize_training_dataset]
    
    @property
    def mini_batch_size(self):
        return self._mini_batch_size
    @property
    def max_epoch(self):
        return self._max_epoch
    @property
    def weights_initialization(self):
        return self._weights_initialization
    @property
    def randomize_training_dataset(self):
        return self._randomize_training_dataset

class NetworkParams:
    ''' Parameters used to create the network
    
    Methods:
            params (list): it returns the list of parameters
    
    Params:
        conv_filters (Bool/int/list)
        conv_kernels (Bool/int/list)
        dropout_rate (Bool/int/list)
        batch_norm (Bool/int/list)
        linear_filters (Bool/int/list)
        conv1D (Bool)
    
    '''
    def __init__(self,conv_filters=False,conv_kernels=3,stride=1,dilation=1,dropout_rate=0.8,batch_norm=True,\
                 linear_filters=False,conv1D=True):
        self._conv_filters=conv_filters
        self._conv_kernels=conv_kernels
        self._stride=stride
        self._dilation=dilation
        self._dropout_rate=dropout_rate
        self._batch_norm=batch_norm
        self._linear_filters=linear_filters
        self._conv1D=conv1D
        
    def params(self):
        return [self._conv_filters,self._conv_kernels,self._stride,self._dilation,\
                self._dropout_rate,self._batch_norm,self._linear_filters,self._conv1D]


class OptimizerParams:
    ''' Parameters used to create the optimizer

    Methods:
            params (list): it returns the list of parameters

    Params:
        learning_rate (float)
        weight_decay (float)

    '''
    def __init__(self,learning_rate=0.001,
                 weight_decay=0):
        self._learning_rate=learning_rate
        self._weight_decay=weight_decay

    def params(self):
        return {'lr':self._learning_rate,'weight_decay':self._weight_decay}

    @property
    def learning_rate(self):
        return self._learning_rate
    @property
    def weight_decay(self):
        return self._weight_decay

class Params:
    ''' Class that defines important parameters used while training/testing a network
        
    Params:
        network_type (nn.Module)
        optimizer_type (torch.optim.Optimizer)
        criterion_type
        newtork_params (NetworkParams)
        optimizer_params (OptimizerParams)
        train_params (TrainParams)
        plot (Bool)
        verbose (Bool)
    
    '''
    
    
    def __init__(self,network_type,
                 optimizer_type,
                 criterion_type,
                 network_params=NetworkParams(),
                 optimizer_params=OptimizerParams(),
                 train_params=TrainParams(),
                 cuda=False,
                 plot=True,
                 verbose=False):
        
        # Set up network
        self._network_type=network_type
        self._network_params=network_params
        if self.network_params._conv1D:
            self._input_shape=[28,50]
        else:
            self._input_shape=[1,28,50]
        self._layers_params=self.set_up_network_params(*self.network_params.params())        
        
        self._network=network_type(self._layers_params)
        if cuda:
            self._network.cuda()

        # Set up optimizer
        self._optimizer_params=optimizer_params
        self._optimizer=optimizer_type(self.network.parameters(), **self.optimizer_params.params())

        # Set up criterion
        self._criterion=criterion_type()
        
        # Set up training parameters
        self._train_params=train_params
        
        # Set up visualization parameters
        self._plot=plot
        self._verbose=verbose
        
    def set_up_network_params(self,conv_filters,conv_kernels,stride,dilation,dropout_rate,batch_norm,linear_filters,conv1D):
        ''' It creates a network of type self._network_type with the given parameters.
        
        Example 1:
        
        network_type=ThreeLayers
        conv_filter=False
        conv_kernels=3
        dropout_rate=[0.2,0.3,0.5]
        batch_norm=True
        linear_filters=[200,2]
        
        # It creates the following network:
        layers=[]
        layers+=MyConv1D(28,28,3,0.2,True).layers
        layers+=MyConv1D(28,28,3,0.3,True).layers
        layers+=MyConv1D(28,28,3,0.5,True).layers
        
        layers.append(Flatten())
        layers.append(nn.Linear(28*44,200))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(200,2))
        
        
        Example 2:
        
        network_type=ThreeLayers
        conv_filter=[54,12,6]
        conv_kernels=3
        dropout_rate=0.4
        batch_norm=False
        linear_filters=[45,2]
        
        # It creates the following network:
        layers=[]
        layers+=MyConv1D(28,54,3,0.4,False).layers
        layers+=MyConv1D(54,12,3,0.4,False).layers
        layers+=MyConv1D(12,6,3,0.4,False).layers
        
        layers.append(Flatten())
        layers.append(nn.Linear(6*44,45))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(45,2))
        
        
        '''
        
        # We automatically create LayersParams based on the input given
        self._num_my_conv_layers=self._network_type.num_my_conv_layers
        self._num_linear_layers=self._network_type.num_linear_layers
        
        # Set up conv_filters
        self._conv_filters=[self._input_shape[0]]+\
                    self.add_params_sequence(self._num_my_conv_layers,conv_filters,self._input_shape[0])
        
        # Set up conv_kernel
        self._conv_kernels=self.add_params_sequence(self._num_my_conv_layers,conv_kernels,3)
        if not conv1D:
            try:
                len(self._conv_kernels[0])
            except:
                self._conv_kernels=[(x,x) for x in self._conv_kernels]

        # Set up stride
        self._stride=self.add_params_sequence(self._num_my_conv_layers,stride,1)
        
        #Set up dilation
        self._dilation=self.add_params_sequence(self._num_my_conv_layers,dilation,1)

        # Set up dropout
        self._dropout_rate=self.add_params_sequence(self._num_my_conv_layers,dropout_rate,0)
        
        # Set up batch norm
        self._batch_norm=self.add_params_sequence(self._num_my_conv_layers,batch_norm,False)
        
        
        # Set up linar_layers
        if conv1D:
            dim1=self._input_shape[1]
            for conv,stride,dilation in zip(self._conv_kernels,self._stride,self._dilation):
                dim1=(dim1-dilation*(conv-1))/stride
            #print(dim1)
            if self._num_my_conv_layers==0:
                self._linear_layer_start_filters=int(np.prod(self._input_shape))
            else:
                self._linear_layer_start_filters=int(self._conv_filters[-1]*dim1)
        else:
            # Dimension 0
            dim0=self._input_shape[1]
            dim1=self._input_shape[2]
            for conv,stride,dilation in zip(self._conv_kernels,self._stride,self._dilation):
                dim0=(dim0-dilation*(conv[0]-1))/stride
                dim1=(dim1-dilation*(conv[1]-1))/stride
            if self._num_my_conv_layers==0:
                self._linear_layer_start_filters=int(np.prod(self._input_shape))
            else:
                self._linear_layer_start_filters=int(self._conv_filters[-1]*dim0*dim1)

        self._linear_filters=[self._linear_layer_start_filters]+\
                        self.add_params_sequence(self._num_linear_layers,linear_filters,False)
        
        self._layer_params_list=[]
        for i in range(self._num_my_conv_layers):
            self._layer_params_list.append(LayerParams(*self._conv_filters[i:i+2],self._conv_kernels[i],\
                    self._stride[i],self._dilation[i],self._dropout_rate[i],self._batch_norm[i]))
        for i in range(self._num_linear_layers):
            self._layer_params_list.append(LayerParams(*self._linear_filters[i:i+2]))
        
        _layer_params=LayersParams(*self._layer_params_list)
        return _layer_params
        
    def add_params_sequence(self,num_layers,params,default_value):
        if not params:
            params=[default_value]*num_layers
        try:
            if len(params)!=num_layers:
                params=[default_value]*num_layers
        except:
            params=[params]*num_layers
            
        return params
    
    
    @property
    def network(self):
        return self._network
    @property
    def optimizer(self):
        return self._optimizer
    @property
    def criterion(self):
        return self._criterion
    @property
    def network_params(self):
        return self._network_params
    @property
    def optimizer_params(self):
        return self._optimizer_params
    @property
    def train_params(self):
        return self._train_params
    @property
    def plot(self):
        return self._plot
    @property
    def verbose(self):
        return self._verbose
