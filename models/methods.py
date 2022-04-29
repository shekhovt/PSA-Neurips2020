import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .losses import *

import models.sah_functions as SAH
from .utils import *

from torch.nn import Parameter
from .random_variable import RandomVar
import copy


class Sum2d(nn.Module):
    def forward(self, x):
        return x.sum(dim=(2, 3))


class Sum2dSB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.A = Sum2d()
        self.weight = Parameter(torch.ones(in_channels))
        self.bias = Parameter(torch.zeros(in_channels))
    
    def forward(self, x):
        s = self.weight.view([1, -1])
        b = self.bias.view([1, -1])
        y = self.A(x) * s + b
        return y


class ScaleBias(nn.Module):
    """
    applies per-channel scale and bias y(n,c,w,h) = x(n,c,w,h) * w(c) + b(c)
    """
    
    def __init__(self, input_channels, **kwargs):
        super().__init__()
        self.bias = Parameter(Tensor(input_channels))
        self.weight = Parameter(Tensor(input_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # this is ment to be for renormalization, not random
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
        return x * w + b
    
    def __repr__(self):
        tmpstr = 'bias: {:.2g}-{:.2g} scale: {:.2g}-{:.2g}'.format(self.bias.min().item(), self.bias.max().item(), self.weight.min().item(),
                                                                   self.weight.max().item())
        return tmpstr


def same_layer_new_weight(layer, weight, bias=None):
    """
    Make a copy of the layer performing the same operation but with new weight and bias
    """
    l = copy.copy(layer)
    l.__dict__['_parameters'] = dict()  # do not shallow copy parameter or buffer dicts
    l.__dict__['_buffers'] = dict()
    if isinstance(weight, Parameter):
        l.weight = weight
    else:
        l.register_buffer('weight', weight)
    if bias is not None:
        if isinstance(bias, Parameter):
            l.bias = bias
        else:
            l.register_buffer('bias', bias)
    else:
        l.register_parameter('bias', None)
    # # assign in the weight foinf around nn.Module's set_attribute mechanics and force there any tensors
    # l.__dict__['_parameters'] = dict()  # do not shallow copy thin
    # l_params = l.__dict__['_parameters']
    # l_params['weight'] = weight
    # if bias is None:
    #     l.bias = None
    # else:
    #     l_params['bias'] = bias
    return l


def align_to_weight(s, weight):
    # always the first dimension in weight is the output dim
    return s.view([-1] + [1] * (weight.dim() - 1))


class LinearWithNorm(nn.Module):
    def __init__(self, linear_layer):
        super().__init__()
        self.layer = linear_layer
        self.initialized = False
        self.sb = ScaleBias(input_channels=linear_layer.bias.size(0))
        self.sb.to(linear_layer.weight.device)
        self.stat = RandomVar()
    
    def forward_norm(self, x: RandomVar):
        """ Reparametriation: W'X + b' =  (WX - mu(WX))/sigma(WX) * s + b
            (WX-mu(WX))/sigma(WX) has statistics (0,1)
            The output has statistics (b, s^2)
            The reparametrized layer has efficient weight: W/sigma * s
            and efficient bias: b - mu/sigma * s
            
        """
        """Compute normalization by propagating channel statistics"""
        # compute and save layer statistics for forward
        w = self.layer.weight
        w2 = self.layer.weight ** 2
        m = x.mean
        v = x.var
        if isinstance(self.layer, nn.Conv2d):
            w = w.sum(dim=(2, 3), keepdim=True)
            w2 = w2.sum(dim=(2, 3), keepdim=True)
            m = m.view([1, -1, 1, 1])
            v = v.view([1, -1, 1, 1])
            self.stat.mean = F.conv2d(m, weight=w, bias=None, stride=1, padding=0).view([1, -1]) # since layer bias cancels. we do not add it here
            self.stat.var = F.conv2d(v, weight=w2, bias=None, stride=1, padding=0).view([1, -1])
        else:
            # handle the flattening here, because we need to know the real size
            S = self.layer.weight.size(1) // m.size(1)
            # intriduce a spatial demension, repat over it and flatten
            m = m.unsqueeze(dim=2).repeat([1, 1, S]).view([1, -1])
            v = v.unsqueeze(dim=2).repeat([1, 1, S]).view([1, -1])
            self.stat.mean = same_layer_new_weight(self.layer, w).forward(m)
            self.stat.var = same_layer_new_weight(self.layer, w2).forward(v)
        #
        self.stat.var += 1e-10
        if not self.initialized:
            self.sb.weight.data = self.stat.std.detach().view([-1])
            self.sb.bias.data = (self.layer.bias - self.stat.mean).detach().view([-1])
            self.initialized = True
        # output statistics: (b, s^2)
        y = RandomVar(self.sb.bias.view(self.stat.mean.size()), (self.sb.weight ** 2).view(self.stat.var.size()))
        #
        self.stat.mean = self.stat.mean[0]  # remove batch dimension
        self.stat.var = self.stat.var[0]
        # send normalized statistics further through
        return y
    
    @property
    def weight(self):
        """
        Normalized weight
        """
        return self.layer.weight * align_to_weight(self.sb.weight / self.stat.std, self.layer.weight)

    @property
    def bias(self):
        # ignoring the layer bias parameters, they cancel in normalization
        return self.sb.bias - self.stat.mean / self.stat.std * self.sb.weight

    def normed_layer(self):
        return same_layer_new_weight(self.layer, self.weight, self.bias)


def _sample_logistic(shape):
    U = torch.rand(shape)
    U = U.add(1e-8).log() - (1 - U).add(1e-8).log()  # this seems to run in CPU
    return U

def _gumbel_sigmoid(logits, T):
    g = _sample_logistic(logits.shape).to(logits)
    return torch.sigmoid(logits.add(g).div(T))

def log1p_exp(x: Tensor) -> Tensor:
    """
    compute log(1+exp(a)) = log(exp(0)+exp(a))
    numerically stabilize so that exp never overflows
    """
    m = torch.clamp(x, min=0).detach()  # max(x,1)
    return m + torch.log(torch.exp(x - m) + torch.exp(-m))


def logit(x: Tensor) -> Tensor:
    return torch.log(x) - torch.log(1 - x)


def flatten(*args, start_dim=1, **kwargs) -> [Tensor, tuple]:
    # return a tensor or a tuple
    z = tuple(x.flatten(start_dim=start_dim, **kwargs) if x is not None else x for x in args)
    return untuple(z)


def makeSquare(obj):
    return lambda a: obj(a) ** 2


class Method:
    def __init__(self, args=None):
        pass

    def preprocess_input(self, x, **kwargs):
        return x
    
    def pooling(self, x, *args, **kwargs):
        return F.avg_pool2d(x, kernel_size=x.shape[2])
    
    def residual_update(self, x, add, *args, **kwargs):
        return x + add
    
    def linear(self, layer, x, *args, **kwargs):
        raise NotImplementedError()
    
    def dispatch(self, layer_cls, **kwargs):
        """
        This maps layer class to one of the member functions of the method
        Since even standard method needs to insert activation after Linear and Conv2d, this is what we commonly dispatch to a custom impl
        """
        try:
            return {nn.Linear: self.linear,
                    nn.Conv2d: self.linear,
                      Sum2dSB: self.linear,
                    }[layer_cls]
        except KeyError:
            # return layer.__class__.forward
            return None
    
    def output(self, layer, x, *args, objs=obj_log_softmax, **kwargs):
        """
        :param objs: one or a list of objs to compute on activations, this is important for expectation methods
        :return: evaluated each obj (one or a tuple)
        """
        a = layer.forward(x)
        res = tuple(obj(a) for obj in to_tuple(objs))
        return untuple(res)


class StandardMethod(Method):
    def __init__(self, args):
        self.activation = dict(sigmoid=torch.sigmoid, tanh=torch.tanh, relu=F.relu)[args.activation]
    
    def linear(self, layer, x, **kwargs):
        return self.activation(layer.forward(x))
    

class InitMethod(StandardMethod):
    
    def normalize(self, weight, bias, a):
        dims = (0, *range(2, a.dim()))  # statistics dimensions
        m = a.mean(dim=dims, keepdim=False)
        s = a.std(dim=dims, keepdim=False)
        # apply
        bias.data = (bias.data - m) / s
        weight.data /= s.view([-1] + [1] * (weight.dim() - 1))
    
    def linear(self, layer, x, **kwargs):
        a = layer.forward(x)
        self.normalize(layer.weight, layer.bias, a)
        a = layer.forward(x)
        return self.activation(a)


class SampleMethod(Method):
    """ This is meant as a test time method only, no extra computation added, just sample forward
    """

    def preprocess_input(self, x, **kwargs):
        self.entropy = 0  # entroypy
        self.bin_units = 0  # number of binary unitns
        return x

    def linear(self, A, x, score=None, entropy=None, last_binary=False, **kwargs):
        # compute pre-activations
        a = A.forward(x)
        # sample output state according to the conditional probability
        check_real(a)
        p0 = a.sigmoid().detach()
        # compute entropy
        batch = p0.size(0)
        H = F.binary_cross_entropy(p0.view([batch, -1]), p0.view([batch, -1]), reduction='none').sum(dim=-1) / math.log(2)
        self.entropy += H
        self.bin_units += p0[0].numel()
        #
        #out = p0.bernoulli() * 2 - 1  # -1, 1
        out = sign_bernoulli(p0)
        if not last_binary:
            return out
        else:
            # probability of the state sampled:
            p_out = (p0 * out - (out - 1) / 2).detach()
            return out, p_out

    def _output_objs(self, A, x, p_x, objs, **kwargs):
        assert (isinstance(A, nn.Linear))
        batch_sz, C_in = x.shape
        a0 = A.forward(x)
        # a_bij = a0_bj - 2*w_ij * x_i
        a = a0[:, None, :] + torch.einsum("ji, bi -> bij", A.weight, -2 * x)  # [B C_in C_out]
        p_flip = (1 - p_x) / C_in  # flip probability [B C_in]
        p_base = (1 - p_flip.sum(dim=1, keepdim=True))
        EE = ()
        for obj in to_tuple(objs):
            f_base = obj(a0)
            f = obj(a)  # [B C_in C_out] -> [B C_in K]
            # simplae part of the output
            E = f_base * p_base
            # heavy part of the output
            E += torch.einsum("bik, bi -> bk", f, p_flip)
            EE = EE + (E,)
        #
        return untuple(EE)
    
    def output(self, A, x, p_x, objs=obj_log_softmax, compute_variances=False, **kwargs):
        """
         objs - a single or a tuple of objectives
         if compute_variances=True, along with stochastic estimate of E[obj], also Var[obj] of the inner samples is returned per objective as tuples (E, V)
        """
        EE = self._output_objs(A, x, p_x, objs=objs)
        if not compute_variances:
            return untuple(EE)
        else:
            objs2 = tuple(makeSquare(obj) for obj in to_tuple(objs))  # Estimate obj**2
            EE2 = self._output_objs(A, x, p_x, objs=objs2)
            T = tuple((E, (M2 - E ** 2).clamp(min=0)) for (E, M2) in zip(EE, EE2))  # clamp is for num accuracy
            # for (E,V) in T:
            #     check_var(V)
            return untuple(T)


class Renormalize(SampleMethod):
    
    def normalize(self, weight, bias, a):
        dims = (0, *range(2, a.dim()))  # statistics dimensions
        m = a.mean(dim=dims, keepdim=False)
        s = a.std(dim=dims, keepdim=False)
        # apply
        # bias.data = (bias.data - m) / s
        # clip weight scales and biases
        sc = s.clamp(min=0.5, max=20)  # from the unit std
        s = s / sc
        m = m / s
        mc = m.clamp(min=-5, max=5)  # from the unit std
        m = m - mc  # if m is large this is non-zero, and will de projected
        #
        bias.data = bias.data / s - m
        weight.data /= s.view([-1] + [1] * (weight.dim() - 1))
    
    def linear(self, layer, x, **kwargs):
        a = layer.forward(x)
        self.normalize(layer.weight, layer.bias, a)
        return super().linear(layer, x, **kwargs)


class ScoreMethod(Method):
    
    def _cond_logp(self, y, a):
        """ compute log conditional probability of output given the output activations a = W x + b
        y: [B C] in {-1,1}
        a: [B C]
        """
        # probability of units in state +1:  p = P(a - Z >= 0) = F(a); lop p = log F(a) = - log (1+exp(-x))
        # probability of units in state -1:  p = P(a - Z < 0) = F(-a); lop p = log F(-a) = - log (1+exp(x))
        logp = -log1p_exp(-a * y).view([a.size(0), -1]).sum(dim=1, keepdim=True)
        return logp
    
    def linear(self, A, x, score=None, **kwargs):
        # compute pre-activations
        a = A(x)
        # sample output state according to the conditional probability
        check_real(a)
        #out = a.sigmoid().bernoulli().detach() * 2 - 1  # -1, 1
        out = sign_bernoulli(a.sigmoid())
        # compute log probability of the samples state -- score
        logp = self._cond_logp(out, a)
        # evaluate for subsequent layers
        score = logp if score is None else score + logp
        return out, score
    
    def output(self, A, x, score, **kwargs):
        EE = super().output(A, x, **kwargs)
        E1 = tuple(E + E.detach() * (score - score.detach()) for E in to_tuple(EE))
        return untuple(E1)
    

class SAHMethod(Method):
    
    def linear(self, A: nn.Linear, x, q=None, last_binary=False, **kwargs):
        return SAH.linear_binary_inner_SAH(x, q, A.weight, A.bias, last_binary=last_binary)

    # def linear_01(self, A, scale, bias, x, q=None, **kwargs):
    #     return SAH.linear_01_binary_SAH(x, q, A, scale, bias, **kwargs)

    def Sum2dSB(self, L, x, q=None, **kwargs):
        # return self.linear_01(layer.A, layer.scale, layer.bias, x, **kwargs)
        s = L.weight.view([1, -1])
        b = L.bias.view([1, -1])
        return SAH.linear_01_binary_SAH(x, q, L.A, s, b, **kwargs)

    def real_to_binary_conv2d(self, A: nn.Conv2d, x, last_binary=False, **kwargs):
        x_out, q_out = SAH.conv_binary_first(x, A.weight, A.bias, padding=A.padding, stride=A.stride, last_binary=last_binary)
        return x_out, q_out

    def conv2d(self, A, x, q=None, last_binary=False, **kwargs):
        if q is None:
            return self.real_to_binary_conv2d(A, x, last_binary, **kwargs)
        else:
            return SAH.conv_binary_inner_SAH(x, q, A.weight, A.bias, padding=A.padding, stride=A.stride, last_binary=last_binary)

    def output(self, A, x, q, p_x, objs=obj_log_softmax, **kwargs):
        assert (isinstance(A, nn.Linear))
        return SAH.binary_out_SAH_vec(x, q, p_x, A.weight, A.bias, objs=objs)

    def dispatch(self, layer_cls, **kwargs):
        """
        This is the only method so far that needs to distinguish Linear and Conv2d -> specialized dispatch
        """
        try:
            return {nn.Linear: self.linear,
                    nn.Conv2d: self.conv2d,
                      Sum2dSB: self.Sum2dSB,
                    }[layer_cls]
        except KeyError:
            return super().dispatch(layer_cls, **kwargs)


class GumbelMethod(Method):
    def __init__(self, args):
        self.T = args.init_temp
    
    def bernoulli(self, logits):
        # sample standard logistic noise for the output units
        U = torch.empty_like(logits).uniform_()
        Z = logit(U)
        # apply soft threshold on noisy activations
        out = torch.tanh((logits - Z) / self.T)
        return out
    
    def linear(self, A, x, *args, **kwargs):
        return self.bernoulli(A(x))


V_S = (math.pi ** 2) / 3  # variance of standard logistic distribution


def Phi_approx(x):
    """
    Approximate narmal_cdf with logistic
    """
    return torch.sigmoid(x * math.sqrt(V_S))  # by matching variance

def sign_bernoulli_AP2(x: RandomVar):
    # approx moments of hard threshold
    # add Logistic var
    s = torch.sqrt(x.var + V_S)
    a = x.mean / s
    y = RandomVar()
    y.mean = Phi_approx(a)
    # variance approx
    y.var = y.mean * (1 - y.mean)
    # rescale to {+1, -1}: y = 2*y -1
    y.mean = y.mean * 2 - 1
    y.var = y.var * 4
    return y

class AP2Method(Method):
    def __init__(self, args):
        super().__init__()
        self.sample_preactivations = False
        
    
    def preprocess_input(self, x, **kwargs):
        return RandomVar(mean=x, var=torch.zeros_like(x))

    def linear(self, layer, x, **kwargs):
        # CLT pre-activations
        y = RandomVar()
        y.mean = layer.forward(x.mean)
        w = layer.weight
        y.var = same_layer_new_weight(layer, weight=w ** 2).forward(x.var)
        #
        # sample from it or propagate the variance
        if self.sample_preactivations:
            a = RandomVar(y.sample())
        else:
            a = y
        return sign_bernoulli_AP2(a)

    def output(self, A, x: RandomVar, objs=obj_log_softmax, **kwargs):
        # apply last linear layer
        y = self.linear(A, x, **kwargs)
        # sample pre-activations
        a = y.sample()
        # compute objective for the sample
        res = tuple(obj(a) for obj in to_tuple(objs))
        return untuple(res)


class LocalReparamMethod(AP2Method):
    def __init__(self, args):
        super().__init__(args)
        self.sample_preactivations = True


class STMethod(Method):
    def linear(self, A, x, **kwargs):
        # compute pre-activations
        a = A.forward(x)
        # sample output state according to the conditional probability
        check_real(a)
        p0 = a.sigmoid()
        out = sign_bernoulli(p0)
        # take value of the sample and gradient of tanh = 2*sigmoid - 1
        out = out.detach() + 2 * (p0 - p0.detach())
        return out


class HardSTMethod(Method):
    def linear(self, A, x, **kwargs):
        # compute pre-activations
        a = A.forward(x)
        # sample output state according to the conditional probability
        check_real(a)
        p0 = a.sigmoid()
        out = sign_bernoulli(p0)
        # clipping identity generator
        a_clip = F.hardtanh(a)
        # take value of the sample and gradient of sigmoid
        out = out.detach() + (a_clip - a_clip.detach())
        return out


class MethodWithNorm(Method):
    def __init__(self, method=None):
        super().__init__()
        self.method = method

    def preprocess_input(self, x, **kwargs):
        sz = [1, x.size(1)]  # channels only
        dev = x.device
        x_norm = RandomVar(torch.zeros(sz, device=dev), var=torch.ones(sz, device=dev))
        x = self.method.preprocess_input(x, **kwargs)
        return (x_norm, *to_tuple(x))

    def LinearWithNorm(self, norm_layer, x_norm: RandomVar(), x, *args, **kwargs):
        if x_norm.size(1) != x.size(1):
            dev = x.device
            x_norm = RandomVar(mean=torch.zeros([1, x.size(1)], device=dev), var=torch.ones([1, x.size(1)], device=dev))
        y_norm = norm_layer.forward_norm(x_norm)
        y_norm = sign_bernoulli_AP2(y_norm)
        # apply normalization, implicit in layer
        layer = norm_layer.normed_layer()
        dispatch_f = self.method.dispatch(layer.__class__)
        y = dispatch_f(layer, x, *args, **kwargs)
        return (y_norm, *to_tuple(y))

    def output(self, norm_layer, x_norm: RandomVar, x, *args, objs=obj_log_softmax, **kwargs):
        norm_layer.forward_norm(x_norm)
        layer = norm_layer.normed_layer()
        assert (isinstance(layer, nn.Linear))
        y = self.method.output(layer, x, *args, objs=objs, **kwargs)
        return y

    def dispatch(self, layer_cls, **kwargs):
        try:
            return {LinearWithNorm: self.LinearWithNorm,
                    }[layer_cls]
        except KeyError:
            return super().dispatch(layer_cls, **kwargs)


class BatchBarrier(STMethod):
    def preprocess_input(self, x, **kwargs):
        self.barrier = 0
        return x
    
    def a_barrier(self, a):
        # mean of activations across data and spatial locations
        dims = (0, *range(2, a.dim()))  # statistics dimensions
        m = a.mean(dim=dims, keepdim=False)
        # penalty on m
        barrier = m.abs().clamp(min=10.0) - 10.0  # penalize m outside of interval [-10, 10]
        self.barrier += barrier.mean()  # channels
    
    def linear(self, layer, x, **kwargs):
        a = layer.forward(x)
        self.a_barrier(a)
        return super().linear(layer, x, **kwargs)
    
    def output(self, layer, x, *args, **kwargs):
        self.linear(layer, x, **kwargs)
        return self.barrier


"""_______________________________________Not Refactored Below___________________________________________"""

class SAHBWMethod(SAHMethod):
    def __init__(self, args):
        self.bw_test = args.bw_test

    def preprocess_input(self, x):
        q = torch.zeros_like(x).to(x)
        return x, q

    def linear(self, weight, bias, x, *args, **kwargs):
        x, q = x[:2]
        if not kwargs.get('training', False) and self.bw_test:
            out = F.linear(x, (weight > 0).float() * 2 - 1, bias=None)
            return out, torch.zeros_like(out), torch.zeros_like(out)

        p0 = weight.sigmoid()
        weight = p0.clamp(1e-6, 1-1e-6).bernoulli().detach() * 2 - 1
        qw = -weight * p0
        qw = qw - qw.detach()
        q = q + qw.sum(dim=0)[None]

        return SAH.linear_binary_inner_SAH(x, q, weight, bias=None)

    def real_to_binary_conv2d(self, weight, bias, x, *args, **kwargs):
        padding = kwargs.get('padding', 0)
        stride = kwargs.get('stride', 1)

        p0 = weight.sigmoid()
        weight = p0.clamp(1e-6, 1-1e-6).bernoulli().detach() * 2 - 1
        qw = -weight * p0
        qw = qw - qw.detach()

        #x_out, q_out = SAH.conv_binary_first(x[0], weight, bias, padding=padding)
        x, q = x
        q = q + F.conv2d(torch.ones_like(x)[:1], qw, padding=padding).sum(dim=1, keepdims=True)
        x_out, q_out = SAH.conv_binary_inner_SAH(x, q, weight, bias, padding=padding)
        x_out = x_out[:, :, ::stride[0], ::stride[1]].contiguous() # self.stride = (stride_h, stride_w)
        q_out = q_out[:, :, ::stride[0], ::stride[1]].contiguous()
        return x_out, q_out

    def conv2d(self, weight, bias, x, *args, **kwargs):
        x, q = x[:2]
        padding = kwargs.get('padding', 0)
        stride = kwargs.get('stride', 1)

        p0 = weight.sigmoid()
        weight = p0.clamp(1e-6, 1-1e-6).bernoulli().detach() * 2 - 1
        qw = -weight * p0
        qw = qw - qw.detach()
        q = q + F.conv2d(torch.ones_like(x)[:1], qw, padding=padding).sum(dim=1, keepdims=True)
        bias = None

        x_out, q_out = SAH.conv_binary_inner_SAH(x, q, weight, bias, padding=padding)
        x_out = x_out[:, :, ::stride[0], ::stride[1]].contiguous() # self.stride = (stride_h, stride_w)
        q_out = q_out[:, :, ::stride[0], ::stride[1]].contiguous()
        return x_out, q_out
       



class CLTMethod(Method):
    def __init__(self, args):
        self.sample = args.clt_sample
        self.binarize = args.clt_binarize
        self.T = args.init_temp

    def preprocess_input(self, x):
        return x

    def linear(self, weight, bias, x, *args, **kwargs):
        mean = F.linear(x, weight, bias=bias)
        if self.sample:
            std = F.linear(x.pow(2), weight.pow(2)).add_(1e-6).sqrt()
        else:
            D = np.prod(mean.shape[1:])
            std = (D - F.linear(x.pow(2), weight.pow(2)).add_(1e-6)).sqrt()

        return self._activation(mean, std, **kwargs)

    def real_to_binary_conv2d(self, weight, bias, x, *args, **kwargs):
        return self.conv2d(weight, bias, x, *args, **kwargs)

    def conv2d(self, weight, bias, x, *args, **kwargs):
        training = kwargs.pop('training', False)
        mean = F.conv2d(x, weight, bias=bias, **kwargs)
        if self.sample:
            std = F.conv2d(x.pow(2), weight.pow(2), **kwargs).add_(1e-6).sqrt()
        else:
            D = np.prod(mean.shape[1:])
            std = (D - F.conv2d(x.pow(2), weight.pow(2), **kwargs).add_(1e-6)).sqrt()

        return self._activation(mean, std, **kwargs)
       
    def output(self, weight, bias, x, *args, **kwargs):
        return F.linear(x, weight, bias=bias)

    def pooling(self, x, *args, **kwargs):
        return F.avg_pool2d(x, kernel_size=x.shape[2])

    def residual_update(self, x, add, *args, **kwargs):
        return x + add

    def _activation(self, mean, std, training=False):
        d = dist.Normal(mean, std)
        p = 1 - d.cdf(0) # p = P(sign(h) == 1)

        if self.sample:
            p = _gumbel_sigmoid(p.add(1e-6).log(), self.T)
            
        if self.binarize or (self.sample and not training):
            p = (p > 0.5).type(torch.float32)

        return 2 * p - 1

class GumbelBWMethod(GumbelMethod):
    def linear(self, weight, bias, x, *args, **kwargs):
        training = kwargs.get('training', False)
        weight = self._apply_gumbel(weight, self.T, training)
        logits = F.linear(x, weight, bias=None)
        return self._apply_gumbel(logits, self.T, training)

    def real_to_binary_conv2d(self, weight, bias, x, *args, **kwargs):
        training = kwargs.get('training', False)
        weight = self._apply_gumbel(weight, self.T, training)
        return self.conv2d(weight, bias, x, *args, **kwargs)

    def conv2d(self, weight, bias, x, *args, **kwargs):
        training = kwargs.pop('training', False)
        weight = self._apply_gumbel(weight, self.T, training)
        logits = F.conv2d(x, weight, bias=None, **kwargs)
        return self._apply_gumbel(logits, self.T, training)


def method_class(method_name: str):
    method_class = {
        'standard': StandardMethod,
        'sample': SampleMethod,
        'score': ScoreMethod,
        'sah': SAHMethod,
        'gumbel': GumbelMethod,
        'clt': CLTMethod,
        'sahbw': SAHBWMethod,
        'gumbelbw': GumbelBWMethod,
        'AP2': AP2Method,
        'LocalReparam': LocalReparamMethod,
        'ST': STMethod,
        'HardST': HardSTMethod,
    }[method_name]
    return method_class

# def get_method(default_method, method = None, args=dict()):
#     if method is None:
#         return default_method
#     else:
#         if isinstance(method, str):
#             return method_class(method)

