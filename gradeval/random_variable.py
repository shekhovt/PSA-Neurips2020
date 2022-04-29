import torch
import numbers
from typing import Union, Dict, Callable

from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod, abstractproperty
from utils import *
import traceback, sys, code

import threading
from threading import current_thread

threadLocal = threading.local()
threadLocal.AttributeError = None

all_checks = False
# all_checks = True

from torch.distributions.gamma import _standard_gamma
# from torch.distributions.utils import _finfo


# some workarounds for Tensor and Tensor conversions
def to_tensor(x: Union[Tensor, numbers.Number]) -> [Tensor, None]:
    if x is None or torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


def to_variable(x: Union[Tensor, Parameter, numbers.Number]) -> Tensor:
    return to_tensor(x)
    
    # if x is None:
    #     return x
    # elif torch.is_tensor(x):
    #     return Tensor(x, requires_grad=False)
    # elif isinstance(x, Tensor) or isinstance(x, Parameter):
    #     return x
    # elif isinstance(x, numbers.Number):
    #     return Tensor(Tensor([x]), requires_grad=False)
    # raise ValueError('do not know %s' % type(x))


class RandomVar:
    """
    Holds a pair of mean and variance, which may be Tensor / Tensor / Parameter
    """
    
    # @property
    # @abstractmethod
    # def mean(self) -> Tensor:
    #     pass
    # 
    # @property
    # @abstractmethod
    # def var(self) -> Tensor:
    #     pass
    # 
    # @mean.setter
    # @abstractmethod
    # def mean(self, value):
    #     pass
    # 
    # @var.setter
    # @abstractmethod
    # def var(self, value):
    #     pass
    
    def __init__(self, mean=None, var=None):
        self.mean = to_variable(mean)
        self.var = to_variable(var)
        # self.mean = mean
        # self.var = var
    
    # shape, concatentation, slicing, resizing
    def size(self):
        return self.mean.size()
    
    def stride(self):
        return self.mean.stride()
    
    def dim(self) -> int:
        if hasattr(self.mean, 'dim'):
            return self.mean.dim()
        else:
            return 0
    
    def cat(self, other, dim) -> 'RandomVar':
        return self.__class__(torch.cat(self.mean, other.mean, dim), torch.cat(self.var, other.var, dim))
    
    def __getitem__(self, key):
        return self.__class__(self.mean.__getitem__(key), self.var.__getitem__(key))
    
    def index_select(self, dim, index, out: 'RandomVar' = None) -> 'RandomVar':
        if out is None:
            return self.__class__(self.mean.index_select(dim, index), self.var.index_select(dim, index))
        else:
            return self.__class__(self.mean.index_select(dim, index, out.mean), self.var.index_select(dim, index, out.var))
    
    def narrow(self, dim: int, start: int, length: int) -> 'RandomVar':
        return self.__class__(self.mean.narrow(dim, start, length), self.var.narrow(dim, start, length))
    
    def split(self, split_size: int, dim: int) -> ['RandomVar']:
        mslice = torch.split(self.mean, split_size, dim)
        vslice = torch.split(self.var, split_size, dim)
        r = []
        for i in range(len(mslice)):
            r = r + [self.__class__(mslice[i], vslice[i])]
        return r
    
    def expand(self, sz) -> 'RandomVar':
        return self.__class__(self.mean.expand(sz), self.var.expand(sz))
    
    def clone(self) -> 'RandomVar':
        return self.__class__(self.mean.clone(), self.var.clone())
    
    def view(self, sz) -> 'RandomVar':
        if list(sz) == list(self.size()):
            return self
        return self.__class__(self.mean.view(sz), self.var.view(sz))
    
    def transpose(self, dim1, dim2) -> 'RandomVar':
        return self.__class__(self.mean.transpose(dim1, dim2), self.var.transpose(dim1, dim2))
    
    def dims_as(self, x: Tensor) -> 'RandomVar':
        sz1 = list(self.mean.size())
        sz1 = sz1 + [1] * (x.dim() - len(sz1))
        sz2 = list(self.var.size())
        sz2 = sz2 + [1] * (x.dim() - len(sz2))
        return self.__class__(self.mean.view(sz1), self.var.view(sz2))
    
    def flatten(self) -> 'RandomVar':
        n = self.mean.size()[0]
        return self.__class__(self.mean.view(n, -1), self.var.view(n, -1))
    
    def contiguous(self) -> 'RandomVar':
        return self.__class__(self.mean.contiguous(), self.var.contiguous())
    
    def detach(self) -> 'RandomVar':
        return self.__class__(self.mean.detach(), self.var.detach())
    
    def type_as(self, x: ['RandomVar', Tensor, Tensor]) -> 'RandomVar':
        if isinstance(x, RandomVar):
            return self.__class__(self.mean.type_as(x.mean), self.var.type_as(x.var))
        else:
            return self.__class__(self.mean.type_as(x), self.var.type_as(x))
    
    def type(self):
        return self.mean.type()
    
    @property
    def is_sparse(self):
        assert self.mean.data.is_sparse == self.var.data.is_sparse
        return self.mean.data.is_sparse

    def cuda(self, device=None) -> 'RandomVar':
        """Returns a copy of this object in CUDA memory"""
        if self.mean is not None:
            return self.__class__(self.mean.cuda(device=device), self.var.cuda(device=device))
        else:
            return self
    
    def cpu(self) -> 'RandomVar':
        """Returns a copy of this object in CPU memory"""
        if self.mean is not None:
            return self.__class__(self.mean.cpu(), self.var.cpu())
        else:
            return self
    
    @property
    def data(self):
        return self.__class__(self.mean.data, self.var.data)
    
    # scalar RV
    def is_scalar(self):
        return self.dim() == 1 and self.size()[0] == 1
    
    # arithmetics
    def __add__(self, b) -> 'RandomVar':
        if isinstance(b, RandomVar):
            return self.__class__(self.mean + b.mean, self.var + b.var)
        else:  # Tensor, constant, etc
            return self.__class__(self.mean + b, self.var)
    
    def __sub__(self, b) -> 'RandomVar':
        if isinstance(b, RandomVar):
            return self.__class__(self.mean - b.mean, self.var + b.var)
        else:  # Tensor, constant, etc
            return self.__class__(self.mean - b, self.var)
    
    def __mul__(self, b) -> 'RandomVar':
        if isinstance(b, RandomVar):
            return self.__class__(self.mean * b.mean, self.var * b.var + self.var * square(b.mean) + square(self.mean) * b.var)
        else:  # Tensor, constant, etc
            return self.__class__(self.mean * b, self.var * b * b)
    
    def __truediv__(self, s) -> 'RandomVar':
        if isinstance(s, RandomVar):
            raise ValueError('value error')
        else:  # Tensor, constant, etc
            return self.__class__(self.mean / s, self.var / (s * s))
    
    # sampling
    def sample(self, bounded=False) -> Tensor:
        """
        sample from Normal distribution (self.mean, self.var), differentiable in mean, var
        """
        n = self.mean.data.new(self.size())
        n.normal_()
        if bounded:
            n = torch.clamp(n, min=-3, max=3)
        if all_checks:
            if not (self.var.data >= 0).all():
                print(self.var.data.min())
            assert (self.var.data >= 0).all()
            assert (self.var.data == self.var.data).all()
            assert (self.var.data != float('inf')).all()
        y = self.mean + n * torch.sqrt(self.var + 1e-16)
        # y = self.mean
        return y
    
    def KL(self, Y: 'RandomVar'):
        """ KL divergence between two Gaussians, component-wise """
        assert self.dim() == Y.dim() or (Y.dim() == 0)  # broadcastable
        return 0.5 * (torch.log(Y.var) - torch.log(self.var) + (self.var + square(self.mean - Y.mean)) / Y.var - 1)
    
    def KL_from_delta(self, x):
        """ KL divergence from delta distribution at x to self """
        v0 = 1e-4
        assert self.dim() == x.dim() or (x.dim() == 0)  # broadcastable
        return 0.5 * (-math.log(v0) + torch.log(self.var + v0) + (v0 + square(self.mean - x)) / (self.var + v0) - 1)
    
    def KLV(self, Y: 'RandomVar'):
        """ KL divergence between two Gaussians, component-wise """
        assert self.dim() == Y.dim() or (Y.dim() == 0)  # broadcastable
        assert (self.var > 1e-16).all()
        assert (Y.var > 1e-16).all()
        return 0.5 * (torch.log(Y.var) - torch.log(self.var) + self.var / Y.var - 1)
        # return 0.5 * (torch.log(Y.var) - torch.log(self.var) + self.var / Y.var )
        # return 0.5 * self.var / Y.var


# class RandomVar(RandomVar):
#     def __init__(self, mean=None, var=None):
#         self._mean = None
#         self._var = None
#         RandomVar.__init__(self, mean, var)
#
#     @property
#     def mean(self) -> Tensor:
#         return self._mean
#
#     @property
#     def var(self) -> Tensor:
#         return self._var
#
#     @mean.setter
#     def mean(self, value):
#         self._mean = value
#
#     @var.setter
#     def var(self, value):
#         self._var = value

#
# class RandomParam(RandomVar):
#     def __init__(self, mean=None, log_var=None):
#         # self._mean = None
#         # self._log_var = None
#         # RandomVar.__init__(self, mean, var)
#         #self._mean = to_variable(mean)
#         #self._log_var = to_variable(log_var)
#         self._mean = mean
#         self._log_var = log_var
#
#     @property
#     def mean(self) -> Tensor:
#         return self._mean
#
#     @property
#     def var(self) -> Tensor:
#         return self._log_var.exp()
#
#     @mean.setter
#     def mean(self, value):
#         self._mean = value
#
#     @var.setter
#     @abstractmethod
#     def var(self, value):
#         pass
#         #assert (value > 0).all()
#         #self._log_var = value.log()
#
#     def cuda(self, device=None, async=False) -> 'RandomVar':
#         """Returns a copy of this object in CUDA memory"""
#         if self.mean is not None:
#             return self.__class__(self._mean.cuda(device=device, async=async), self._log_var.cuda(device=device, async=async))
#         else:
#             return self
#


def upgrade_to_RV(x) -> RandomVar:
    if isinstance(x, RandomVar):
        return RandomVar(x.mean, x.var)  # to avoid confusion whether RandomVar is constructed or pointer taken
    elif isinstance(x, Tensor) or isinstance(x, Tensor):
        return RandomVar(mean=to_variable(x), var=None)


def square(x):
    return x * x


def cat(seq, dim=0):
    if isinstance(seq[0], torch.Tensor):
        return torch.cat(seq, dim)
    elif isinstance(seq[0], RandomVar):
        return RandomVar(torch.cat([x.mean for x in seq], dim), torch.cat([x.var for x in seq], dim))
    else:
        raise ValueError


def cat_reduce(seq, dim=0, **kwargs):
    if (kwargs.get('compute_normalization') or kwargs.get('init_normalization')):
        out_sz = list(seq[0].size())
        out_sz[dim] = sum([x.size()[dim] for x in seq])
        seq = [contract_spatial(x) for x in seq]
        x = cat(seq, dim)
        x = x.expand(out_sz)
    else:
        x = cat(seq, dim)
    return x


def shallow_copy(x):
    if isinstance(x, torch.Tensor):
        return x.view(x.size())
        return x.view([-1]).view(x.size())
    elif isinstance(x, RandomVar):
        return RandomVar(shallow_copy(x.mean), shallow_copy(x.var))
    else:
        raise ValueError


def contract(x: [Tensor, RandomVar]):
    # x = shallow_copy(x)
    st = list(x.stride())
    sz = list(x.size())
    y = torch.tensor()
    
    raise NotImplementedError()
    
    # workaround in pytorch 4.0
    # for i in range(len(sz)):
    #     if st[i] == 0:
    #         sz[i] = 1
    # if isinstance(x, Tensor):
    #     x.data.set_(x.data.storage(), storage_offset=0, size=sz, stride=x.stride())
    # elif torch.is_tensor(x):
    #     x.set_(x.storage(), storage_offset=0, size=sz, stride=x.stride())
    # elif isinstance(x, RandomVar):
    #     x.mean.data.set_(x.mean.data.storage(), storage_offset=0, size=sz, stride=x.mean.stride())
    #     x.var.data.set_(x.var.data.storage(), storage_offset=0, size=sz, stride=x.var.stride())
    # else:
    #     raise ValueError()
    # return x


def contract_spatial(x: [Tensor, RandomVar]):
    # x = shallow_copy(x)
    st = list(x.stride())
    sz = list(x.size())
    if x.dim() < 4:
        return x
    for i in (2, 3):
        if st[i] == 0:
            sz[i] = 1
    
    # x = x.clone()
    #
    if st[2] == 0 and st[3] == 0:
        if isinstance(x, Tensor):
            return x[:, :, 0, 0].view(sz)
        elif isinstance(x, RandomVar):
            y = RandomVar()
            y.mean = x.mean[:, :, 0, 0].view(sz)
            y.var = x.var[:, :, 0, 0].view(sz)
            return y
        else:
            raise ValueError()
    
    return x
    
    # workaround in pytorch 4.0
    
    # x = x.clone()
    # if isinstance(x, Tensor):
    #     x.data.set_(x.data.storage(), storage_offset=0, size=sz, stride=x.stride())
    # elif torch.is_tensor(x):
    #     x.set_(x.storage(), storage_offset=0, size=sz, stride=x.stride())
    # elif isinstance(x, RandomVar):
    #     x.mean.data.set_(x.mean.data.storage(), storage_offset=0, size=sz, stride=x.mean.stride())
    #     x.var.data.set_(x.var.data.storage(), storage_offset=0, size=sz, stride=x.var.stride())
    # else:
    #     raise ValueError()
    # return x


class StochasticParameter(ModuleBase):
    def __init__(self, t: Tensor, var_init=10.0 ** 2, var_prior=10.0 ** 2, var_parameter=True, mean_prior=False):
        torch.nn.Module.__init__(self)
        self._mean = Parameter(t)
        sz = list(t.size())
        if t.dim() == 4:
            sz[1] = 1  # input channels hack
            sz[2] = 1
            sz[3] = 1
        # self._std = Parameter(t.new(torch.Size(sz)).fill_( math.sqrt(var_init) ))
        if var_parameter:
            # self._var = Parameter(t.new(torch.Size(sz)).fill_(math.log(var_init))) #
            self._std = Parameter(t.new(torch.Size(sz)).fill_(math.sqrt(var_init)))
        else:
            # self.register_buffer('_std', t.new(torch.Size(sz)).fill_(math.log(var_init)))  #
            self.register_buffer('_std', t.new(torch.Size(sz)).fill_(math.sqrt(var_init)))  #
        self.register_buffer('prior', RandomVar(float(0.0), float(var_prior)))  # bias prior
        # self.prior = RandomVar(0.0, var_prior).type_as(t)
        self._sample = None
        self.reg_loss = 0
        self.sampling = False
        self.mean_prior = mean_prior
        self.eps = 1e-8
    
    @property
    def size(self):
        return list(self._mean.size())
    
    @property
    def var(self):
        return square(self._std) + self.eps
        # return self._var
        # return self._var.exp()
    
    def project(self):
        self._std.data = self._std.data.clamp(min=0)
        # self._var.data = self._var.data.clamp(min=self.eps)
        # self._var.data = self._var.data.clamp(max=20)
    
    @property
    def RV(self):
        return RandomVar(self._mean, self.var.expand(self.size))
    
    def new_sample(self):
        assert self.sampling
        self._sample = self.RV.sample(bounded=True)
        self.compute_KL()
    
    @property
    def sample(self):
        if self._sample is None:
            self.new_sample()
        return self._sample
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def current(self):
        if self.sampling:
            return self._sample
        else:
            return self._mean
    
    @current.setter
    def current(self, value):
        assert not self.sampling
        self._mean = value
    
    def forward(self, *input):
        raise ValueError('should not be calling forward on this')
    
    def compute_KL(self, **kwargs):
        if self.mean_prior:
            self.reg_loss = self.RV.KL(self.prior).sum()
        else:
            self.reg_loss = self.RV.KLV(self.prior).sum()
        pass
    
    def train(self, training=True):
        if training:
            self.sampling = True
        else:
            self.sampling = False
        super().train(training)


class StochasticParam(ModuleBase):
    def __init__(self, t: Tensor, var_sz=None, var_init=10.0 ** 2, mean_prior=100.0, var_prior=100.0 ** 2, scale_reparam=1, positive_mean=True):
        torch.nn.Module.__init__(self)
        self.sampling = False
        self.eps = 1e-8
        self.positive_mean = positive_mean
        self.scale_reparam = float(scale_reparam)
        self.var_prior = var_prior
        if var_sz is None:
            var_sz = list(t.size())
        #
        self._mean = Parameter()
        self._y = Parameter()
        if isinstance(var_init, numbers.Number):
            var_init = t.new(size=var_sz).fill_(var_init)
        self.set_mean_var(t, var_init)
        #
        self.register_buffer('prior_N', RandomVar(float(mean_prior), float(var_prior)))
        #
        self._sample = None
        self.reg_loss = 0
        #
        # debug
        # self.freeze()
        # self._mean.fill_(1.0)
        # self._y.fill_(-10.0)
    
    def is_stochastic(self):
        return True
    
    def freeze_mean(self):
        _mean = self._mean
        del self._mean
        self.register_buffer('_mean', _mean.data)
    
    def freeze_var(self):
        _y = self._y
        del self._y
        self.register_buffer('_y', _y.data)
    
    def freeze(self):
        self.freeze_mean()
        self.freeze_var()
    
    @property
    def size(self):
        return list(self._mean.size())
    
    def set_mean_var(self, m, v):
        self.mean = m
        self.var = v
    
    @property
    def mean(self):
        return self._mean
    
    @mean.setter
    def mean(self, value):
        self._mean.data = value
    
    @property
    def var(self):
        # return 1 / square(self._is * self.scale_reparam)
        # return square(self.std)
        # return torch.min(y.exp(), math.exp(1)/2*(y**2 + 1) )
        # y = self._y * self.scale_reparam
        # return y.exp().min(y.abs() + 1)
        # -- var
        # return self._y * self.scale_reparam
        # -- std
        return square(self.std)
        # -- linear grads
        # mask = (self._y <= 0).detach().float()
        # sq = square(self._y * self.scale_reparam)
        # v = torch.exp(-sq) * mask + (sq + 1) * (1 - mask)
        # return v * self.var_prior
        # --
    
    @var.setter
    def var(self, v):
        # self._is.data =  1 / v.sqrt() / self.scale_reparam
        # self._s.data = v.sqrt() / self.scale_reparam
        # -- var
        # self._y.data = v / self.scale_reparam
        # return None
        # -- std
        self.std = v.sqrt()
        # mask = v > 1
        # y = v.log()
        # t = (2 / math.exp(1) * v - 1).sqrt()
        # y.masked_scatter_(mask, t)
        # --
        # self.std = v.sqrt()
        # -- linear grads
        # v = v / self.var_prior
        # mask = v > 1
        # y = -(-(v.clamp(max=1)).log()).sqrt()
        # y.masked_scatter_(mask, (v - 1).sqrt())
        # self._y.data = y / self.scale_reparam
        # --
    
    @property
    def ivar(self):
        raise NotImplementedError
        # return 1/square(self.std)
    
    @property
    def std(self):
        # return 1 / (self._is * self.scale_reparam)
        # return self._s * self.scale_reparam
        # return self._s * self.scale_reparam
        # return (y/2).exp().min((square(y) + 1).sqrt())
        # -- exp-linear
        y = self._y * self.scale_reparam
        return y.exp().min(y.abs() + 1) * math.sqrt(self.var_prior)
        # -- var, linear grads
        # return self.var.sqrt()
        # -- std
        # return self._y * self.scale_reparam
    
    @std.setter
    def std(self, s):
        # return 1 / (self._is * self.scale_reparam)
        # self._s.data = s / self.scale_reparam
        #
        # self.var = s ** 2
        # -- exp-linear
        s = s / math.sqrt(self.var_prior)
        mask = s > 1
        y = s.log()
        y.masked_scatter_(mask, s - 1)
        self._y.data = y / self.scale_reparam
        # -- var, linear grads
        # self.var = s ** 2
        # -- std
        # self._y.data =  s / self.scale_reparam
    
    def project(self):
        if (self.positive_mean):
            self._mean.data.clamp_(min=self.eps)
        # -- var
        # self._y.data.clamp_(min=1e-6 / self.scale_reparam, max=1e3 / self.scale_reparam)
        # -- std
        # self._y.data.clamp_(min=1e-3 / self.scale_reparam, max=1e3 / self.scale_reparam)
        # --
        # self._s.data.clamp_(min=1e-3 / self.scale_reparam, max=1e3 / self.scale_reparam)
        #
        # -- exp-linear std
        self._y.data.clamp_(min=-10 / self.scale_reparam, max=1e3 / self.scale_reparam)
        
        # -- linear grads
        # self._y.data.clamp_(min=-3 / self.scale_reparam, max=2 / self.scale_reparam)
        # --
    
    @abstractmethod
    def new_sample(self, sz):
        pass
    
    @abstractmethod
    def compute_KL(self, **kwargs):
        pass
    
    @property
    def current(self):
        if self.sampling:
            return self._sample
        else:
            return self.mean
    
    @current.setter
    def current(self, value):
        assert not self.sampling
        self.mean = value
    
    def forward(self, *input):
        raise ValueError('should not be calling forward on the Stochastic Parameter')
    
    def train(self, training=True):
        if training:
            self.sampling = True
        else:
            self.sampling = False
        super().train(training)
    
    def __repr__(self):
        tmpstr = '({}) [ '.format(self.__class__.__name__)
        nsr = self.std / self.mean
        tmpstr += 'mean: [{:.2g}, {:.2g}] std: [{:.2g}, {:.2g}] NSR: [{:.2g}, {:.2g} ,{:.2g}]'.format(self.mean.min().item(), self.mean.max().item(),
                                                                                                      self.std.min().item(), self.std.max().item(),
                                                                                                      nsr.min().item(), nsr.mean().item(), nsr.max().item())
        tmpstr += ']'
        return tmpstr


class StochasticParamGamma(StochasticParam):
    def __init__(self, t: Tensor, var_sz=None, var_init=10.0 ** 2, mean_prior=100, var_prior=100.0 ** 2, **kwargs):
        super().__init__(t, var_sz=var_sz, var_init=var_init, mean_prior=mean_prior, var_prior=var_prior, **kwargs)
        #
        self.prior = torch.distributions.gamma.Gamma(square(self.prior_N.mean) / self.prior_N.var, self.prior_N.mean / self.prior_N.var)
    
    def new_sample(self, sz):
        if len(sz) == 4:
            sz[2] = 1
            sz[3] = 1
        assert self.sampling
        alpha = square(self.mean) * self.ivar
        beta = self.mean * self.ivar
        self._sample = _standard_gamma(alpha.cpu().expand(sz)).cuda() / beta
        self._sample.data.clamp_(min=1e-10)
        self.compute_KL()
    
    def compute_KL(self, **kwargs):
        # sum over channels and batch dimension
        alpha = square(self.mean) * self.ivar
        beta = self.mean * self.ivar
        d = torch.distributions.gamma.Gamma(alpha, beta)
        self.reg_loss = (d.log_prob(self._sample).sum() - self.prior.log_prob(self._sample).sum()) / self._sample.size()[0]


class StochasticParamNormal(StochasticParam):
    def __init__(self, t: Tensor, var_sz=None, var_init=10.0 ** 2, mean_prior=100, var_prior=100.0 ** 2, **kwargs):
        super().__init__(t, var_sz=var_sz, var_init=var_init, mean_prior=mean_prior, var_prior=var_prior, **kwargs)
    
    def new_sample(self, sz):
        if len(sz) == 4:
            sz[2] = 1
            sz[3] = 1
        assert (self.sampling)
        n = self.mean.new(size=sz).normal_()
        n.data.clamp_(min=-3, max=3)
        self._sample = self.mean.expand(sz) + n * self.std.expand(sz)
        self.compute_KL()
    
    # @property
    # def var(self):
    #     return super().var * square(self.mean)
    #
    # @var.setter
    # def var(self, value):
    #     if to_tensor(value).size() != self._mean.size():
    #         v = value / square(self.mean.mean())
    #     else:
    #         v = value / square(self.mean)
    #     StochasticParam.var.fset(self, v)
    
    def compute_KL(self, **kwargs):
        # sum over channels
        # if self._y.requires_grad or self._mean.requires_grad:
        self.reg_loss = RandomVar(self.mean, self.var).KL(self.prior_N).sum()
        # g = torch.autograd.grad(self.reg_loss, self._is, retain_graph=True)[0][0, 0].item()
        # return g
        #
        # d = torch.distributions.normal.Normal(self.mean, self.std)
        # self.reg_loss = (d.log_prob(self._sample).sum() - self.prior.log_prob(self._sample).sum()) / self._sample.size()[0]


class StochasticParam_LogUniformPrior(StochasticParam):
    def compute_KL(self, **kwargs):
        # log uniform prior, approximation due to Molchanov et al. 2017
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        alpha = self.var.expand(self._sample.size())
        log_alpha = torch.log(alpha)
        NKL = k1 * F.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log(1 + 1 / alpha)
        # sum over channels
        self.reg_loss = -NKL.sum()
    
    def project(self):
        # -- exp-linear std
        self._y.data.clamp_(min=-10 / self.scale_reparam, max=0)  # constraints the variance in the range [exp(-10), 1)]


class StochasticParamNormal_LogUniform(StochasticParam_LogUniformPrior):
    def __init__(self, t: Tensor, var_sz=None, var_init=10.0 ** 2, **kwargs):
        super().__init__(t, var_sz=var_sz, var_init=var_init, **kwargs)
        self.freeze_mean()
    
    def new_sample(self, sz):
        # to make it equivalent to Gaussina Dropout: sample over spatial dimensions and channels, compute KL proportionally
        assert self.sampling
        n = self.mean.new(size=sz).normal_()
        n.data.clamp_(min=-3, max=3)
        self._sample = self.mean.expand(sz) + n * self.std.expand(sz)
        self.compute_KL()


class StochasticParamNormal_VariationalDropout(StochasticParam_LogUniformPrior):
    def __init__(self, t: Tensor, var_sz=None, var_init=10.0 ** 2, **kwargs):
        super().__init__(t, var_sz=var_sz, var_init=var_init, **kwargs)
    
    @property
    def std(self):
        s_ = StochasticParam_LogUniformPrior.std.fget(self)
        return self.mean.abs() * s_
    
    @std.setter
    def std(self, s):
        s_ = s / self.mean.abs()
        StochasticParam_LogUniformPrior.std.fset(self, s_)
    
    def new_sample(self):
        assert self.sampling
        n = self.mean.new(size=self.mean.size()).normal_()
        n.data.clamp_(min=-3, max=3)
        self._sample = self.mean + n * self.std
        self.compute_KL()


class StochasticParamNormal_VariationalDropoutRT(StochasticParam_LogUniformPrior):
    def __init__(self, t: Tensor, var_sz=None, var_init=10.0 ** 2, **kwargs):
        super().__init__(t, var_sz=var_sz, var_init=var_init, **kwargs)
    
    def compute_KL(self, **kwargs):
        # log uniform prior, approximation due to Molchanov et al. 2017
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        alpha = self.var.expand(self.mean.size())
        log_alpha = torch.log(alpha)
        NKL = k1 * F.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log(1 + 1 / alpha)
        # sum over channels
        self.reg_loss = -NKL.sum()
    
    def project(self):
        # -- exp-linear std
        self._y.data.clamp_(min=-10 / self.scale_reparam, max=0)  # constraints the variance in the range [exp(-10), 1)]
    
    @property
    def var(self):
        alpha = (2 * self._y).exp()
        return square(self.mean) * alpha
    
    @var.setter
    def var(self, v):
        self.std = v.sqrt()
    
    @property
    def std(self):
        # s_ = StochasticParam_LogUniformPrior.std.fget(self)
        s_ = self._y.exp()
        return self.mean.abs() * s_
    
    @std.setter
    def std(self, s):
        # s_ = s / self.mean.abs() # problem is that mean is of larger size
        self._y.data = s.log()
        # StochasticParam_LogUniformPrior.std.fset(self, s_)


class StochasticParamLogNormal(StochasticParamNormal):
    def __init__(self, t: Tensor, var_sz=None, var_init=10.0 ** 2, mean_prior=100, var_prior=100.0 ** 2, **kwargs):
        if var_sz is None or list(var_sz) != list(t.size()):
            lv = math.log(1.0 + var_init / t.mean().item() ** 2)
        else:
            lv = (1.0 + var_init / t ** 2).log()
        lm = t.log() - lv / 2
        
        pv = math.log(1.0 + var_prior / mean_prior ** 2)
        pm = math.log(mean_prior) - pv / 2
        super().__init__(lm, var_sz=var_sz, var_init=lv, mean_prior=pm, var_prior=pv, positive_mean=False, scale_reparam=10, **kwargs)
    
    @property
    def current(self):
        if self.sampling:
            return self._sample.exp()
        else:
            return (self.mean + self.var / 2).exp()
    
    @current.setter
    def current(self, value):
        assert not self.sampling
        self.mean = value.log() - self.var / 2
