import torch
import numbers
from typing import Union, Dict, Callable

TypeError = NotImplemented

from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod, abstractproperty

from gradeval.utils import ModuleBase
import traceback, sys, code

import threading
from threading import current_thread
# from functions import *
# from utils import *

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


class TensorCat:
    """
        Holds a tuple of Tensors of equal sizes
    """

    @property
    def first(self) -> Tensor:
        return self.list[0]

    @first.setter
    def first(self, value):
        self.list[0] = value

    @property
    def second(self) -> Tensor:
        return self.list[1]

    @second.setter
    def second(self, value):
        self.list[1] = value

    def __init__(self, tensors):
        # self.tensor = torch.cat([t.view([1] + list(t.size())) for t in tensors], dim = 0)
        # self.list = [self.tensor.select(dim=0, index=i) for i in range(len(tensors))]
        self.list = list(tensors)

    # shape, concatentation, slicing, resizing
    def size(self, *args):
        return self.list[0].size(*args)

    @property
    def device(self):
        return self.list[0].device

    def stride(self, *args):
        return self.list[0].stride(*args)

    def dim(self) -> int:
        if hasattr(self.list[0], 'dim'):
            return self.list[0].dim()
        else:
            return 0

    def cat(self, other, dim) -> 'TensorCat':
        return self.__class__([torch.cat(self.list[i], self.list[i], dim) for i in range(len(self.list))])
    
    def __getitem__(self, key):
        return self.__class__([self.list[i].__getitem__(key) for i in range(len(self.list))])

    def expand(self, sz) -> 'TensorCat':
        return self.__class__([self.list[i].expand(sz) for i in range(len(self.list))])

    def flatten(self, **kwargs) -> 'TensorCat':
        return self.__class__([x.flatten(**kwargs) for x in self.list])

    def clone(self) -> 'TensorCat':
        return self.__class__([self.list[i].clone() for i in range(len(self.list))])
    
    def new_zeros(self, sz) -> 'TensorCat':
        return self.__class__([self.list[i].new_zeros(sz) for i in range(len(self.list))])

    def new_ones(self, sz) -> 'TensorCat':
        return self.__class__([self.list[i].new_ones(sz) for i in range(len(self.list))])

    def zeros_like(self) -> 'TensorCat':
        return self.new_zeros(self.size())

    def view(self, sz) -> 'TensorCat':
        return self.__class__([self.list[i].view(sz) for i in range(len(self.list))])

    def contiguous(self) -> 'TensorCat':
        return self.__class__([self.list[i].contiguous() for i in range(len(self.list))])

    def detach(self) -> 'TensorCat':
        return self.__class__([self.list[i].detach() for i in range(len(self.list))])

    # statement arithmetics
    def __iadd__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            [self.list[i].__iadd__(b.list[i]) for i in range(len(self.list))]
        else:  # Tensor, constant, etc
            [self.list[i].__iadd__(b) for i in range(len(self.list))]
        return self
            
    def __isub__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            [self.list[i].__isub__(b.list[i]) for i in range(len(self.list))]
        else:  # Tensor, constant, etc
            [self.list[i].__isub__(b) for i in range(len(self.list))]
        return self
            
    def __imul__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            [self.list[i].__imul__(b.list[i]) for i in range(len(self.list))]
        else:  # Tensor, constant, etc
            [self.list[i].__imul__(b) for i in range(len(self.list))]
        return self

    # arithmetics
    def __neg__(self):
        return self.__class__([self.list[i].__neg__() for i in range(len(self.list))])
    
    def __add__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__add__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__add__(b) for i in range(len(self.list))])

    def __sub__(self, b) -> 'TenorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__sub__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__sub__(b) for i in range(len(self.list))])

    def __mul__(self, b) -> 'TenorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__mul__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__mul__(b) for i in range(len(self.list))])
        
    def __rmul__(self, b):
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__rmul__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__rmul__(b) for i in range(len(self.list))])

    def __truediv__(self, b) -> 'TenorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__truediv__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__truediv__(b) for i in range(len(self.list))])
        
    def __rtruediv__(self, b) -> 'TenorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__rtruediv__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__rtruediv__(b) for i in range(len(self.list))])

    def fill_(self, val):
        for t in self.list:
            t.fill_(val)



class RandomVar(TensorCat):
    """
    Holds a pair of mean and variance, which may be Tensor / Tensor / Parameter
    """
    
    # @property
    # @abstractmethod
    @property
    def mean(self) -> Tensor:
        return self.list[0]

    @mean.setter
    def mean(self, value):
        self.list[0] = value

    @property
    def var(self) -> Tensor:
        return self.list[1]

    @var.setter
    def var(self, value):
        self.list[1] = value

    @property
    def std(self) -> Tensor:
        return self.var.sqrt()
        
    def __init__(self, mean=None, var=None):
        if isinstance(mean, (list, tuple)):
            TensorCat.__init__(self, mean)
        else:
            if var is None and mean is not None:
                var = torch.zeros_like(mean)
            TensorCat.__init__(self, [mean, var])
    
    # arithmetics
    def __add__(self, b) -> 'RandomVar':
        if isinstance(b, TensorCat):
            return self.__class__(self.mean + b.mean, self.var + b.var)
        else:  # Tensor, constant, etc
            return self.__class__(self.mean + b, self.var)
    
    def __sub__(self, b) -> 'RandomVar':
        if isinstance(b, TensorCat):
            return self.__class__(self.mean - b.mean, self.var + b.var)
        else:  # Tensor, constant, etc
            return self.__class__(self.mean - b, self.var)
    
    def __mul__(self, b) -> 'RandomVar':
        if isinstance(b, RandomVar):
            return self.__class__(self.mean * b.mean, self.var * b.var + self.var * square(b.mean) + square(self.mean) * b.var)
        elif isinstance(b, TensorCat):
            raise ValueError('Operation not defined')
        else:  # Tensor, constant, etc
            return self.__class__(self.mean * b, self.var * b * b)
    
    def __truediv__(self, s) -> 'RandomVar':
        if isinstance(s, TensorCat):
            raise ValueError('Operation not defined')
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

    # Gaussian #
    def p(self, x):
        return gauss_pdf((x - self.mean) / self.std) / self.std

    def log_p(self, x):
        return -0.5 * ((x - self.mean) ** 2) / self.var - 0.5*self.var.log()

    def F(self, x):
        return Phi_approx((x - self.mean) / self.std)
    
    def true_F(self, x):
        return gauss_cdf((x - self.mean) / self.std)

    def log_deltaF(self, a, b):
        a1 = (a - self.mean) / self.std * math.sqrt(V_S)
        b1 = (b - self.mean) / self.std * math.sqrt(V_S)
        return logistic_log_F_diff(a1, b1)

    def F_grad(self, x0):
        p = self.p(x0)
        gmu = -p
        gv = 0.5 * (self.mean - x0) / self.var * p
        return TensorCat([gmu, gv])
    
    def min_grad(self, x0):
        #computes grad of the expected value of min(X-a, 0)
        return TensorCat([self.F(x0), -0.5 * self.p(x0)])
    
    def p_grad(self, x0):
        u = (x0 - self.mean) / self.var
        p = self.p(x0)
        gmu = p * u
        gv = 0.5 * p * (u ** 2 - 1 / self.var)
        
        return TensorCat([gmu, gv])


class RandomVarWithGrad(TensorCat):
    def __init__(self, X):
        if isinstance(X, (list, tuple)):
            TensorCat.__init__(self, X)
        elif isinstance(X, RandomVar):
            Xgrad = TensorCat([torch.zeros_like(t) for t in [X.mean, X.var]])
            TensorCat.__init__(self, [X, Xgrad])
        else:
            raise ValueError('Operation not defined')
    @property
    def rv(self) -> Tensor:
        return self.list[0]

    @rv.setter
    def rv(self, value):
        self.list[0] = value

    @property
    def mean(self) -> Tensor:
        return self.rv.mean

    @mean.setter
    def mean(self, value):
        self.rv.mean = value

    @property
    def var(self) -> Tensor:
        return self.rv.var

    @var.setter
    def var(self, value):
        self.rv.var = value
        
    @property
    def std(self) -> Tensor:
        return self.rv.mean
        
    @property
    def grad(self):
        return self.list[1]

    @grad.setter
    def grad(self, value):
        self.list[1] = value
        
    def zero_o_grad(self):
        self.grad = self.grad.detach()
        self.grad.fill_(0)
        self.mean.grad = None
        self.var.grad = None

    def MD_step(self, step_size, prec_f = None):
        
        check_real(self.mean)
        check_var(self.var)
        
        precision = 1 / self.var
        precision = precision * (1 - step_size) - 2 * step_size * self.grad.list[1]
        if prec_f is not None: # precision post processing function
            precision = prec_f(precision)
        self.var = 1 / precision
        self.mean = self.mean + step_size * self.var * self.grad.list[0]
        # check_real(self.mean)
        # check_var(self.var)

    # Gaussian #
    def p(self, x):
        return self.rv.p(x)

    def log_p(self, x):
        return self.rv.log_p(x)

    def F(self, x):
        return self.rv.F(x)
    
    def true_F(self, x):
        return self.rv.F(x)

    def log_deltaF(self, a, b):
        return self.rv.log_deltaF(a, b)

    def F_grad(self, x0):
        return self.rv.F_grad(x0)

    def p_grad(self, x0):
        return self.rv.p_grad(x0)

        
    
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
