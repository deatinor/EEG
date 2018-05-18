import torch.nn as nn

class Flatten(nn.Module):
    ''' Flatten the input vector.

    Used to pass from convolutional to linear layers.
    '''
    def forward(self, input):
        return input.view(input.size(0), -1)

class Tensor4D(nn.Module):
    ''' Creates a 4D tensor from a 3D one.

    Used for convolutional 2D.
    '''
    def forward(self,input):
        return input.view(input.shape[0],1,*input.shape[1:])

class Tensor3D(nn.Module):
    ''' Creates a 3D tensor 

    Used for convolutional 1D.
    '''
    def forward(self,input):
        return input.view(input.shape[0],input.shape[1],-1)

class MyConv1D:
    ''' Conv1d layer with ReLU and optional batch_norm and dropout

    @params
    - input_channels, output_channels. Input and output channels of the convolutional layer.
    - kernel. Kernel size
    - stride.
    - dilation.
    - dropout_rate. 0 if no dropout
    - batch_norm. True or False
    '''
    def __init__(self,input_channels,output_channels,kernel,stride,dilation,dropout_rate=0.8,batch_norm=True):
        self.conv=nn.Conv1d(input_channels,output_channels,kernel,stride=stride,dilation=dilation)
        self.batch_norm=nn.BatchNorm1d(output_channels)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout_rate)
        if batch_norm:
            self._layers=[self.conv,self.batch_norm,self.relu,self.dropout]
        else:
            self._layers=[self.conv,self.relu,self.dropout]
    
    @property
    def layers(self):
        return self._layers

class MyConv2D:
    ''' Conv2d layer with ReLU and optional batch_norm and dropout

    @params
    - input_channels, output_channels. Input and output channels of the convolutional layer.
    - kernel. Kernel size
    - stride.
    - dilation.
    - dropout_rate. 0 if no dropout
    - batch_norm. True or False
    '''
    def __init__(self,input_channels,output_channels,kernel,stride,dilation,dropout_rate=0.8,batch_norm=True):
        self.conv=nn.Conv2d(input_channels,output_channels,kernel,stride=stride,dilation=dilation)
        self.batch_norm=nn.BatchNorm2d(output_channels)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout_rate)
        if batch_norm:
            self._layers=[self.conv,self.batch_norm,self.relu,self.dropout]
        else:
            self._layers=[self.conv,self.relu,self.dropout]
    
    @property
    def layers(self):
        return self._layers
