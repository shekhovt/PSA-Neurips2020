import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

class ScaleBias(nn.Module):
    """
    applies per-channel scale and bias y(n,c,w,h) = x(n,c,w,h) * w(c) + b(c)
    """
    
    def __init__(self, input_channels, **kwargs):
        nn.Module.__init__(self)
        self.bias = Parameter(Tensor(input_channels))
        self.weight = Parameter(Tensor(input_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        #for renormalization, not random
        self.bias.data.fill_(0.0)
        self.weight.data.fill_(1.0)
    
    def align_shape(self, x: Tensor):
        # align dimensions to input for broadcasting
        if x.dim() == 4:
            shape = [1, -1, 1, 1]
        elif x.dim() == 2:
            shape = [1, -1]
        else:
            raise ValueError("don\'t know how to treat such input dimension")
        w = self.weight.view(shape)
        b = self.bias.view(shape)
        return w, b
    
    def forward(self, x, **kwargs):
        w, b = self.align_shape(x)
        return x * w + b # internally works by broadcasting w and b to the size of x

    def __repr__(self):
        # fore pretty pringting
        tmpstr = 'bias: {:.2g}-{:.2g} scale: {:.2g}-{:.2g}'.format(self.bias.min().item(), self.bias.max().item(), self.weight.min().item(),
                                                                   self.weight.max().item())
        return tmpstr
