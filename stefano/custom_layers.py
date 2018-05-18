import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Tensor4D(nn.Module):
    def forward(self,input):
        return input.view(input.shape[0],1,*input.shape[1:])

class Tensor3D(nn.Module):
    def forward(self,input):
        return input.view(input.shape[0],input.shape[1],-1)

class MyConv1D:
    def __init__(self,input_channels,output_channels,kernel,stride,dropout_rate=0.8,batch_norm=True):
        self.conv=nn.Conv1d(input_channels,output_channels,kernel,stride=stride)
        self.batch_norm=nn.BatchNorm1d(output_channels)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout_rate)
        if batch_norm:
            self._layers=[self.conv,self.relu,self.batch_norm,self.dropout]
        else:
            self._layers=[self.conv,self.relu,self.dropout]
    
    @property
    def layers(self):
        return self._layers

class MyConv2D:
    def __init__(self,input_channels,output_channels,kernel,dropout_rate=0.8,batch_norm=True):
        self.conv=nn.Conv2d(input_channels,output_channels,kernel)
        self.batch_norm=nn.BatchNorm2d(output_channels)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout_rate)
        if batch_norm:
            self._layers=[self.conv,self.relu,self.batch_norm,self.dropout]
        else:
            self._layers=[self.conv,self.relu,self.dropout]
    
    @property
    def layers(self):
        return self._layers
