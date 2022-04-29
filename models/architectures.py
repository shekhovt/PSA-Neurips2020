import context
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from models.methods import *
import numbers
import copy
import time


class Flatten(nn.Module):
    def forward(self, *args, **kwargs):
        return flatten(*args, **kwargs)


class Preactivation1d(nn.Sequential):  #
    def __init__(self, num_features, dropout=0.0, bn=False, **kwargs):
        ll = []
        if bn:
            ll += [nn.BatchNorm1d(num_features=num_features)]
        if dropout > 0:
            ll += [nn.Dropout(dropout)]
        super().__init__(*ll)


class Preactivation2d(nn.Sequential):  #
    def __init__(self, num_features, dropout=0.0, bn=False, **kwargs):
        ll = []
        if bn:
            ll += [nn.BatchNorm2d(num_features=num_features)]
        if dropout > 0:
            ll += [nn.Dropout2d(dropout)]
        super().__init__(*ll)


class LastLinear(nn.Linear):
    def __init__(self, channels_per_class, groups, K):
        super().__init__(K * channels_per_class * groups, K)
        w = self.weight.data
        del self.weight
        self.bias = None
        self.register_buffer('weight', w)
        #
        w = w.view((K, channels_per_class, groups, K))
        w.fill_(0.0)
        # coeff: for max 95% confidence
        n = channels_per_class * groups
        s = 1 / (2 * n) * (math.log(K - 1) - math.log(1 / 0.95 -1))
        # partition in_channels
        for c in range(K):
            w[c, :, :, c] = s

class NetWithMethod(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.layers = nn.ModuleList()
    
    def forward(self, x, method=None, **kwargs):
        method = method if method is not None else self.method
        z = method.preprocess_input(x, **kwargs)
        for i in range(len(self.layers) - 1):
            start_time = time.time()
            layer = self.layers[i]
            last_binary = (i == len(self.layers) - 2)
            dispatch_f = method.dispatch(layer.__class__)
            if dispatch_f is not None:
                z = dispatch_f(layer, *to_tuple(z), last_binary=last_binary, **kwargs)
            else:
                z = layer.forward(*to_tuple(z))  # just call whatever the layer was doing, without extra args
            layer.time = time.time() - start_time
        z = flatten(*to_tuple(z))
        start_time = time.time()
        layer = self.layers[-1]
        z = method.output(layer, *to_tuple(z), **kwargs)
        layer.time = time.time() - start_time
        return z




def reparam_with_norm(net):
    for (i, l) in enumerate(net.layers):
        if isinstance(l, (nn.Linear, nn.Conv2d)):
            net.layers[i] = LinearWithNorm(l)
    net.method = MethodWithNorm(method=net.method)


class FullyConnected(NetWithMethod):
    
    def __init__(self, method, args):
        super().__init__(method)
        self._make_layers(args)
    
    def _make_layers(self, args):
        input_sizes = [np.prod(args.input_shape)] + [args.hidden_size] * (args.n_layers - 1)
        output_sizes = [args.hidden_size] * args.n_layers
        self.layers += [Flatten()]
        for i in range(len(input_sizes)):
            self.layers += [nn.Linear(input_sizes[i], output_sizes[i])]
        self.layers += [nn.Linear(output_sizes[-1], args.n_classes)]


class ConvSumFC(NetWithMethod):
    def __init__(self, method, args):
        super().__init__(method)
        self.eval()
        # activation = ABlock(activation, **kwargs)
        ll = self.layers
        C = args.hidden_size
        ll += [nn.Conv2d(args.input_shape[0], C, kernel_size=5)]
        ll += [Sum2dSB(C)]
        ll += [nn.Linear(C, 10)]
        
        
class CIFAR_S1(NetWithMethod):
    def __init__(self, method, args):
        super().__init__(method)
        self.eval()
        #activation = ABlock(activation, **kwargs)
        ll = self.layers
        # conv layers
        ksize  = [ 3,  3,  3,   3,   3,   3,   3,   1, 1]
        stride = [ 1,  1,  2,   1,   1,   2,   1,   1, 1]
        odepth = [96, 96, 96, 192, 192, 192, 192, 192, 10]
        # padd   = [ 1,  1,  1,   1,   1,   1,   1,   0, 0]
        idepth = 3  # expect color image
        CC = nn.Conv2d
        for l in range(len(ksize)-1):
            ll += [CC(idepth, odepth[l], kernel_size=ksize[l], stride=stride[l], padding=0)]
            idepth = odepth[l]
        # average pooling 6
        ll += [nn.Linear(idepth * 4, 10)]
        # print(self)


class CIFAR_S1F(NetWithMethod):
    def __init__(self, method, args):
        super().__init__(method)
        self.eval()
        #activation = ABlock(activation, **kwargs)
        ll = self.layers
        # conv layers
        ksize  = [ 3,  3,  3,   3,   3,   3,   3,   1, 1]
        stride = [ 1,  1,  2,   1,   1,   2,   1,   1, 1]
        odepth = [96, 96, 96, 192, 192, 192, 192, 200, 10]
        # padd   = [ 1,  1,  1,   1,   1,   1,   1,   0, 0]
        idepth = 3  # expect color image
        CC = nn.Conv2d
        for l in range(len(ksize)-1):
            ll += [CC(idepth, odepth[l], kernel_size=ksize[l], stride=stride[l], padding=0)]
            idepth = odepth[l]
        # average pooling 6
        # ll += [nn.Linear(idepth * 4, 10)]
        ll += [LastLinear(channels_per_class=20, groups=4, K=10)]
        # print(self)


"""_______________________________________Not Refactored Below___________________________________________"""


class ConvNet(nn.Module):
    def __init__(self, method, args):
        super().__init__()
        self.method = method
        self._make_layers(args)
    
    def forward(self, x):
        z = self.method.preprocess_input(x)
        for layer in self.conv_layers:
            z = layer(z)
        z = self.method.pooling(z)
        z = self.method.flatten(z)
        for layer in self.fc_layers:
            z = layer(z)
        return self.output_layer(z)
    
    def _get_channels_and_strides(self, arch):
        channels = []
        strides = []
        
        for x in arch.split(','):
            s = x.split('/')
            if len(s) == 1:
                channels.append(int(s[0]))
                strides.append(1)
            else:
                channels.append(int(s[0]))
                strides.append(int(s[1]))
        
        return channels, strides
    
    def _make_layers(self, args):
        out_channels_list, strides = self._get_channels_and_strides(args.arch)
        in_channels_list = [args.input_shape[1]] + out_channels_list[:-1]
        
        self.conv_layers = nn.ModuleList([
            Conv2dWithMethod(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, method=self.method)
            for in_channels, out_channels, stride in
            zip(in_channels_list, out_channels_list, strides)
        ])
        
        w = args.input_shape[2]
        for s in strides:
            w //= s
        
        input_sizes = [out_channels_list[-1]] + [args.hidden_size] * (args.n_layers - 1)
        output_sizes = [args.hidden_size] * args.n_layers
        
        self.fc_layers = nn.ModuleList([
            LinearWithMethod(in_features, out_features, method=self.method)
            for in_features, out_features in
            zip(input_sizes, output_sizes)
        ])
        
        self.output_layer = OutputLinearWithMethod(output_sizes[-1], args.n_classes, method=self.method)


class ConvFC(nn.Module):
    def __init__(self, method, args):
        self.method = method
        super().__init__()
        self.conv = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.output_layer = None
    
    def forward(self, x, method=None, **kwargs):
        method = method if method is not None else self.method
        z = method.preprocess_input(x)
        for (i, layer) in enumerate(self.conv):
            if i == 0:
                z = layer(*to_tuple(z), method=method, first_layer=True, **kwargs)
            else:
                z = layer(*to_tuple(z), method=method, **kwargs)
        z = method.flatten(*to_tuple(z))
        for (i, layer) in enumerate(self.fc):
            last_binary = (i == len(self.fc) - 1)
            z = layer(*to_tuple(z), method=method, last_binary=last_binary, **kwargs)
        return self.output_layer(*to_tuple(z), method=method, **kwargs)


class LeNet5(ConvFC):
    def __init__(self, method, args):
        super().__init__(method, args)
        self.conv = nn.ModuleList([
            Conv2dWithMethod(1, 6, kernel_size=5, padding=2, stride=2, method=method, bn=args.add_batch_norm),
            Conv2dWithMethod(6, 16, kernel_size=5, padding=0, stride=2, method=method, bn=args.add_batch_norm),
        ])
        
        self.fc = nn.ModuleList([
            LinearWithMethod(400, 120, method=method, bn=args.add_batch_norm),
            LinearWithMethod(120, 84, method=method, bn=args.add_batch_norm)
        ])
        
        self.output_layer = OutputLinearWithMethod(84, 10, method=method)


class LeNet5_torch(ConvFC):
    def __init__(self, method, args):
        super().__init__(method, args)
        self.conv += [Conv2dWithMethod(1, 32, kernel_size=3, padding=0, stride=1, method=method, bn=args.add_batch_norm)]
        self.conv += [Conv2dWithMethod(32, 64, kernel_size=3, padding=0, stride=2, method=method, bn=args.add_batch_norm, dropout=0.0)]  # 0.25
        self.fc += [LinearWithMethod(9216, 128, method=method, bn=args.add_batch_norm, dropout=0.0)]  # 0.5
        self.output_layer = OutputLinearWithMethod(128, 10, method=method)


class VGGLike(ConvFC):
    def __init__(self, method, args):
        super().__init__()
        self.method = method
        self.conv = nn.ModuleList([
            Conv2dWithMethod(3, 32 * args.vgg_multiplier, kernel_size=3, padding=1, method=method, bn=args.add_batch_norm),
            Conv2dWithMethod(32 * args.vgg_multiplier, 32 * args.vgg_multiplier, kernel_size=3, padding=1, stride=2, method=method, bn=args.add_batch_norm),
            Conv2dWithMethod(32 * args.vgg_multiplier, 64 * args.vgg_multiplier, kernel_size=3, padding=1, method=method, bn=args.add_batch_norm),
            Conv2dWithMethod(64 * args.vgg_multiplier, 64 * args.vgg_multiplier, kernel_size=3, padding=1, stride=2, method=method, bn=args.add_batch_norm),
            Conv2dWithMethod(64 * args.vgg_multiplier, 128 * args.vgg_multiplier, kernel_size=3, padding=1, method=method, bn=args.add_batch_norm),
            Conv2dWithMethod(128 * args.vgg_multiplier, 128 * args.vgg_multiplier, kernel_size=3, padding=1, stride=2, method=method, bn=args.add_batch_norm)
        ])
        self.fc = nn.ModuleList([
            LinearWithMethod(16 * 128 * args.vgg_multiplier, 256, method=method),
        ])
        self.output_layer = OutputLinearWithMethod(256, 10, method=method)


class ResidualLinearWithMethod(nn.Linear):
    def __init__(self, in_features, out_features, method=None):
        super().__init__()
        if in_features != out_features:
            self.proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.proj = None
        self.method = method
    
    def forward(self, x, *args, **kwargs):
        kwargs.update(training=self.training)
        z = self.method.linear(self.weight, self.bias, x, *args, **kwargs)
        if self.proj is None:
            add = x
        else:
            add = self.proj(x)
        return self.model.residual_update(z, add)


class FCResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._make_layers(args)
    
    def forward(self, x):
        z = self.method.preprocess_input(x)
        z = self.method.flatten(x)
        for layer in self.layers:
            z = layer(z)
        return self.output_layer(z)
    
    def _make_layers(self, args):
        input_sizes = [np.prod(args.input_shape)] + [args.hidden_size] * (args.n_layers - 1)
        output_sizes = [args.hidden_size] * args.n_layers
        
        self.layers = nn.ModuleList([
            ResidualLinearWithMethod(in_features, out_features, method=self.method)
            for in_features, out_features in
            zip(input_sizes, output_sizes)
        ])
        
        self.output_layer = OutputLinearWithMethod(output_sizes[-1], args.n_classes, method=self.method)


class GumbelConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        in_channels, out_channels = args
        use_bn = kwargs.pop('bn', False)
        
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
    
    def forward(self, x):
        z = self.conv(x)
        z = self.bn(z)
        z = GumbelMethod._apply_gumbel(z, training=self.training)
        return z


class DetConvVGGLike(nn.Module):
    def __init__(self, method, args):
        super().__init__()
        self.method = method
        
        if args.add_batch_norm:
            batch_norm = nn.BatchNorm2d
        else:
            batch_norm = nn.Identity
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            batch_norm(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            batch_norm(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            batch_norm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            batch_norm(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            batch_norm(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        )
        self.fc = nn.ModuleList([
            LinearWithMethod(16 * 128, 256, method=method),
        ])
        self.output_layer = OutputLinearWithMethod(256, 10, method=method)
    
    def forward(self, x):
        z = self.method.preprocess_input(self.conv(x))
        z = self.method.flatten(z)
        for layer in self.fc:
            z = layer(z)
        return self.output_layer(z)


class GumbelConvVGGLike(nn.Module):
    def __init__(self, method, args):
        super().__init__()
        self.method = method
        self.conv = nn.Sequential(
            GumbelConv2d(3, 32, kernel_size=3, padding=1, bn=args.add_batch_norm),
            GumbelConv2d(32, 32, kernel_size=3, padding=1, stride=2, bn=args.add_batch_norm),
            GumbelConv2d(32, 64, kernel_size=3, padding=1, bn=args.add_batch_norm),
            GumbelConv2d(64, 64, kernel_size=3, padding=1, stride=2, bn=args.add_batch_norm),
            GumbelConv2d(64, 128, kernel_size=3, padding=1, bn=args.add_batch_norm),
            GumbelConv2d(128, 128, kernel_size=3, padding=1, stride=2, bn=args.add_batch_norm)
        )
        self.fc = nn.ModuleList([
            LinearWithMethod(16 * 128, 256, method=method),
        ])
        self.output_layer = OutputLinearWithMethod(256, 10, method=method)
    
    def forward(self, x):
        z = self.method.preprocess_input(self.conv(x))
        z = self.method.flatten(z)
        for layer in self.fc:
            z = layer(z)
        return self.output_layer(z)


def model_class(model_name: str):
    model_class = {
        'fc': FullyConnected,
        'conv': ConvNet,
        'lenet5': LeNet5,
        'lenet5T': LeNet5_torch,
        'vgglike': VGGLike,
        'dcvgglike': DetConvVGGLike,
        'gcvgglike': GumbelConvVGGLike,
        'fc_resnet': FCResNet,
        'ConvSumFC': ConvSumFC
    }
    if model_name in model_class:
        return model_class[model_name]
    else:
        return globals()[model_name]
