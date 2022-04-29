import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from experiments.options import odict
from gradeval.utils import *
from gradeval.random_variable import *
import copy
from models.sah_functions import *


V_S = (math.pi ** 2) / 3  # variance of standard logistic distribution


def Phi_approx(x):
    """
    Approximate narmal_cdf with logistic
    """
    return torch.sigmoid(x * math.sqrt(V_S))  # by matching variance


def log1p_exp(x: Tensor) -> Tensor:
    """
    compute log(1+exp(a)) = log(exp(0)+exp(a))
    numerically stabilize so that exp never overflows
    """
    m = torch.clamp(x, min=0).detach()  # max(x,1)
    return m + torch.log(torch.exp(x - m) + torch.exp(-m))


def logit(x: Tensor) -> Tensor:
    return torch.log(x) - torch.log(1 - x)


class LinkedLayer(nn.Module):
    def __init__(self, out_units=1, prev=None, options=odict()):
        nn.Module.__init__(self)
        self.options = options
        self.out_units = out_units
        self.link = odict()
        self.next = None
        if prev is not None:
            self.link.prev = prev
            prev.next = self
            self.in_units = prev.out_units
        else:
            self.link.prev = None
            self.in_units = None
        self.reset_parameters()
    
    def reset_parameters(self):
        pass
    
    def forward(self, *args, method=None, **kwargs):
        if method == 'enumerate':
            return self.fw_enumerate(*args, method=method, **kwargs)
        if method == 'determ':
            return self.fw_determ(*args, method=method, **kwargs)
        if method == 'score':
            return self.fw_score(*args, method=method, **kwargs)
        if method == 'sample':
            return self.fw_sample(*args, method=method, **kwargs)
        if method == 'concrete':
            return self.fw_concrete(*args, method=method, **kwargs)
        if method == 'AP1':
            return self.fw_AP1(*args, method=method, **kwargs)
        if method == 'ST':
            return self.fw_ST(*args, method=method, **kwargs)
        if method == 'SA':
            return self.fw_SA(*args, method=method, **kwargs)
        if method == 'SAH':
            return self.fw_SAH(*args, method=method, **kwargs)
        if method == 'SAH1':
            return self.fw_SAH1(*args, method=method, **kwargs)
        if method == 'AP2-init':
            return self.fw_AP2_init(*args, method=method, **kwargs)
        if method == 'ARM':
            return self.fw_ARM(*args, method=method, **kwargs)
        else:
            raise AttributeError("method % s is not known" % (method))
    
    def grad_vector(self, input):
        """ collect grad of all parameters and of the input (if required) into a single vector"""
        g = []
        if input.requires_grad:
            g.append(input.grad.view([-1]))
        for p in self.parameters():  # recurrently collects parameters
            if p.grad is None:
                g.append(p.view([-1]) * 0)
            else:
                g.append(p.grad.view([-1]))
        g = torch.cat(g)
        return g
    
    def grad_list(self, input=None):
        """ collect grad of all parameters and of the input (if required) into a single vector"""
        g = []
        if input is not None and input.requires_grad:
            g.append(input.grad.view([-1]))
        if not isinstance(self, InputLayer):
            if self.weight.grad is None or self.bias.grad is None:
                g_my = torch.cat([self.weight.view([-1]) * 0, self.bias.view([-1]) * 0])
            else:
                g_my = torch.cat([self.weight.grad.view([-1]), self.bias.grad.view([-1])])
            g.append(g_my)
        if self.next is not None:
            g.extend(self.next.grad_list())
        return g
    
    def zero_grad(self, input=None):
        nn.Module.zero_grad(self)
        if input is not None and input.requires_grad and input.grad is not None:
            input.grad.detach_()
            input.grad.zero_()
    
    def grad_gen(self, input, method=None, n_samples=1, output='list', **kwargs):
        """
        estimate gradient with method using n_samples
        :return: yields the gradient for each sample
        """
        for s in range(n_samples):
            self.zero_grad(input)
            E = self.forward(input, method=method, **kwargs)
            E.backward()
            if output is 'vector':
                yield self.grad_vector(input)
            else:
                yield self.grad_list(input)
    
    def grad(self, input, method=None, n_samples=1, **kwargs):
        """
        estimate gradient with method using n_samples
        :return: the gradient if n_samples=1 or the gradient generator for n_samples>1
        """
        if method == 'enumerate':
            n_samples = 1
        g = self.grad_gen(input, method=method, n_samples=n_samples, **kwargs)
        if n_samples == 1:
            return next(g)
        else:
            return g


# class ScoreFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, function, score):
#         result = function
#         ctx.save_for_backward(function, score)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         foo, score = ctx.saved_tensors
#         return grad_output, foo*score.grad


class InputLayer(LinkedLayer):
    """ Determenistic input layer, can include a linear mapping of the input, but for now just identity"""
    
    def forward(self, x, sample_batch=None, **kwargs):
        # dummy identity mapping for any method
        if 'SA' in kwargs['method']:  # need input correction to {-1,1} assumption
            x = 2 * x - 1  # different state convention, temporary
        if sample_batch is None:
            sample_batch = 1
        if x.dim() == 1:
            x = x.view([1, -1]).expand(sample_batch, -1)
        return self.next.forward(x, **kwargs)


def linear(x, weight, bias=None):
    if x.dim() == 1:
        r = weight.matmul(x)
        if bias is not None:
            r = r + bias
        return r
    else:
        return F.linear(x, weight, bias=bias)


class OutputLayer(LinkedLayer):
    """ Determenistic output layer: a linear combination of inputs and some loss function """
    
    def reset_parameters(self):
        weight = torch.empty(self.out_units, self.in_units).uniform_(-1, 1)
        bias = torch.empty(self.out_units).uniform_(-1, 1)
        # self.register_buffer('weight', weight)
        # self.register_buffer('bias', bias)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
    
    def F(self, a):
        return a ** 2
    
    def generic_F(self, input_state, obj=None, **kwargs):
        a = linear(input_state, self.weight, self.bias)
        if obj is None:
            return a
        else:
            return obj(a)
    
    def generic_FS(self, input_state, obj=None, **kwargs):
        return self.generic_F((input_state + 1) / 2, obj, **kwargs)
    
    def fw_enumerate(self, input_state, **kwargs):
        return self.generic_F(input_state, **kwargs)
    
    def fw_sample(self, input_state, **kwargs):
        return self.generic_F(input_state, **kwargs)
    
    def fw_determ(self, input_state, **kwargs):
        return self.generic_F(input_state, **kwargs)
    
    def fw_concrete(self, input_state, **kwargs):
        return self.generic_F(input_state, **kwargs)
    
    def fw_AP1(self, input_state, **kwargs):
        return self.generic_F(input_state, **kwargs)
    
    def fw_ST(self, input_state, **kwargs):
        return self.generic_F(input_state, **kwargs)

    def fw_score(self, input_state, score=torch.tensor([0.0]), return_logp=False, **kwargs):
        E = self.generic_F(input_state, **kwargs)
        if kwargs.get('multi_sample_ML'):
            E = E.mean().log()
            score = score.sum()
        if return_logp:
            # return E1, score
            LL = torch.logsumexp(E.log() + score, dim=0)
            return LL
        E1 = E + E.detach() * (score - score.detach())
        return E1

    def fw_ARM(self, input_state, grad_obj, **kwargs):
        E = self.generic_F(input_state, **kwargs)
        if kwargs.get('multi_sample_ML'):
            E = E.mean().log()
            grad_obj = grad_obj.sum()
        return E + (grad_obj - grad_obj.detach())

    def fw_SA(self, x, q=None, qmax=None, p_out=None, **kwargs):
        return self.fw_SAH(x, q, qmax, p_out, **kwargs)  # uses somewhat improved estimate of the expected value
        batch_sz = x.size(0)
        # if qmax is not None:
        #     print(qmax.item())
        if q is None:
            q = torch.zeros_like(x)
        F = self.generic_FS(x, **kwargs)
        E = F * (1 - q.sum(dim=1, keepdim=True))
        for i in range(self.in_units):
            x1 = x.clone()
            x1[:, i] = - x[:, i]
            F = self.generic_FS(x1, **kwargs)
            E += F * q[:, i].view([batch_sz, 1])
        # return E.mean()
        return E
    
    def fw_SAH(self, x, q=None, qmax=None, p_out=None, **kwargs):
        # return self.fw_SA(x, q, qmax, **kwargs)
        batch_sz = x.size(0)
        n = self.in_units
        p_out = p_out.view([batch_sz, -1, 1]).detach()
        q = q - q.detach()
        q = q.view([batch_sz, -1, 1])
        p_flip = (1 - p_out) / n + q  # total flip probability
        f_base = self.generic_FS(x, **kwargs)
        E = f_base
        for i in range(n):
            x1 = x.clone()
            x1[:, i] = - x[:, i]
            f_flip = self.generic_FS(x1, **kwargs)
            E = E + (f_flip - f_base) * p_flip[:, i, :]
        return E

    def fw_SAH1(self, x, q=None, qmax=None, p_out=None, obj=None, **kwargs):
        # obj = lambda x : self.generic_FS(x, **kwargs)
        # return binary_out_SAH(x, q, p_out, obj)
        #
        # convert w, b to encoding of x = -1,1
        w = self.weight / 2
        b = self.bias + w.sum(dim=1)
        # if obj is None:
        #     obj = lambda a: a

        def objS(a):
            if a.dim() == 2:
                return obj(a)
            else:
                rr = []
                for s in range(a.size(1)):
                    a1 = a[:, s, :]
                    rr += [obj(a1).unsqueeze(dim=1)]
                return torch.cat(rr, dim=1)
            
        #
        return binary_out_SAH_vec(x, q, p_out, w, b, objS)
    
    def fw_AP2_init(self, X, **kwargs):
        return None

class LogisticBernoulliLayer(LinkedLayer):
    
    def reset_parameters(self):
        # weight = torch.empty(self.out_units, self.in_units).bernoulli_()
        weight = torch.empty(self.out_units, self.in_units).uniform_(-1, 1)
        bias = torch.empty(self.out_units).uniform_(-1, 1)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
    
    def cond_logp(self, y, a):
        """ compute log conditional probability of output given the output activations a = W x + b"""
        # probability of units in state 1:  p(a - Z >= 0) = F(a); lop p = log F(a) = - log (1+exp(-x))
        # probability of units in state 0:  p(a - Z < 0) = 1 - F(a); lop p = log(1 - F(a)) = - log (1+exp(x))
        logp = -log1p_exp(torch.where(y > 0.5, -a, a)).sum(dim=1, keepdim=True)
        return logp
    
    def fw_enumerate(self, input_state, **kwargs):
        batch_sz = input_state.size(0)
        # compute activations
        a = linear(input_state, self.weight, self.bias)
        
        # enumerate all our states
        E = 0
        for i in range(2 ** self.out_units):
            out = input_state.new(batch_sz, self.out_units)
            # decode integer i to a binary vector
            i0 = i
            for k in range(self.out_units):
                out[:, k] = i0 % 2
                i0 = i0 // 2
            # compute probability of this state given the input
            logp = self.cond_logp(out, a)
            # evaluate recurrently for subsequent layers
            E_out = self.next.forward(out, **kwargs)
            E = E + logp.exp() * E_out
        return E
    
    def fw_determ(self, input_state, **kwargs):
        batch_sz = input_state.size(0)
        # compute activations
        a = linear(input_state, self.weight, self.bias)
        # hard threshold activations
        out = (a > 0).type_as(input_state).detach()
        return self.next.forward(out, **kwargs)
    
    def fw_score(self, input_state, score=0, **kwargs):
        # compute activations
        a = linear(input_state, self.weight, self.bias)
        # sample output state according to the conditional probability
        out = a.sigmoid().bernoulli().detach()
        # compute log probability of the samples state -- score
        logp = self.cond_logp(out, a)
        # evaluate for subsequent layers
        return self.next.forward(out, score=score + logp, **kwargs)
    
    def fw_sample(self, input_state, **kwargs):
        # compute activations
        a = linear(input_state, self.weight, self.bias)
        # sample output state according to the conditional probability
        out = a.sigmoid().bernoulli().detach()
        # evaluate for subsequent layers
        return self.next.fw_sample(out, **kwargs)

    def fw_ARM(self, input_state, grad_obj=0, method='ARM', **kwargs):
        # compute activations
        a = linear(input_state, self.weight, self.bias)
        # sample unifor
        U = a.new_empty(a.size()).uniform_()
        # ARM expansion
        b1 = (U > torch.sigmoid(-a)).type_as(a)
        b2 = (U < torch.sigmoid(a)).type_as(a)
        f1 = self.next.forward(b1, **kwargs, method='sample')
        f2 = self.next.forward(b2, **kwargs, method='sample')
        # ARM generator of grad in a, vector of batch size
        ARM = (f1 - f2).detach() * ((a * (U - 0.5)).sum(dim=1, keepdim=True))
        # sample state to go for the next layer
        out = torch.sigmoid(a).bernoulli().detach()
        return self.next.forward(b1, grad_obj + ARM, method=method, **kwargs)
    
    def fw_concrete(self, input_state, tau, **kwargs):
        # compute activations
        a = linear(input_state, self.weight, self.bias)
        # sample standard logistic noise for the output units
        U = torch.empty_like(a).uniform_()
        Z = logit(U)
        # apply soft threshold on noisy activations
        out = torch.sigmoid((a - Z) / tau)
        # evaluate for subsequent layers
        return self.next.forward(out, tau=tau, **kwargs)
    
    def fw_AP1(self, input_state, **kwargs):
        # compute activations
        a = linear(input_state, self.weight, self.bias)
        # expected output
        out = a.sigmoid()
        # evaluate for subsequent layers
        return self.next.forward(out, **kwargs)
    
    def fw_ST(self, input_state, **kwargs):
        # compute activations
        # a = self.weight.matmul(input_state) + self.bias
        a = linear(input_state, self.weight, self.bias)
        # sample output state according to the conditional probability
        out = a.sigmoid().bernoulli().detach()
        # fake gradient
        y = a.sigmoid()
        out = out + (y - y.detach())
        # evaluate for subsequent layers
        return self.next.forward(out, **kwargs)
    
    
    def fw_SA(self, x, q=None, t=100, qmax=None, p_out=None, **kwargs):
        """ input x [batch, channels]
        """
        
        kwargs['t'] = t
        
        def F_Z(a):
            return torch.sigmoid(a)
        
        def Delta(y_col, a0, ai=None):
            if ai is None:
                ai = a0
            a0_bar = a0.detach()
            y_col = y_col.detach()
            Fai = torch.sigmoid(ai)
            if t == 0:
                if ai is a0:
                    return (F_Z(a0_bar) - Fai) * y_col * 1 / 2
                else:
                    return y_col * (F_Z(a0_bar) - Fai) * (((ai - a0_bar) * y_col < 0).type_as(a0))
            else:
                
                b = torch.clamp((ai - a0_bar), min=-t, max=t)
                # # # y_col = -1:
                # P = b / (2 * t) + 1 / 2
                # F1 = 1 / (2 * t) * (log1p_exp(a0_bar + b) - log1p_exp(a0_bar - t))
                # Delta1 = (1 - y_col) / 2 * (P * Fai - F1)  # if y==-1
                # # # y_col = 1:
                # F2 = 1 / (2 * t) * (log1p_exp(a0_bar + t) - log1p_exp(a0_bar + b))
                # Delta2 = (y_col + 1) / 2 * (F2 - (1 - P) * Fai)  # if y==1
                # D = Delta1 + Delta2
                
                pi = b / (2 * t) - y_col / 2
                D = pi * Fai + 1 / (2 * t) * (log1p_exp(a0_bar + t * y_col) - log1p_exp(a0_bar + b))
                return D
        
        batch_sz = x.size(0)
        
        # convert w, b to encoding of x = -1,1
        w = self.weight / 2
        b = self.bias + w.sum(dim=1)
        if q is None:
            q = torch.zeros_like(x)
        if qmax is None:
            qmax = x.new_zeros(1)
        x_row = x.view(batch_sz, 1, self.in_units)
        # w = w.view(1, self.out_units, self.in_units)
        # b = b.view(1, self.out_units, 1)
        
        # compute activations
        # a0_col = w.matmul(x.view([batch_sz, self.in_units, 1])) + b
        a0_col = linear(x, w, b).view([batch_sz, self.out_units, 1])
        # a0_col = a0.view([batch_sz, self.out_units, 1])
        
        ai = a0_col - 2 * w * x_row
        # sample augmented state
        # prior_y_col = F_Z(a0_col.detach())
        if t == 0:
            prior_y_col = F_Z(a0_col.detach())
        else:
            prior_y_col = 1 / (2 * t) * (log1p_exp(a0_col.detach() + t) - log1p_exp(a0_col.detach() - t))
        
        prior_y_col = torch.clamp(prior_y_col, min=0, max=1)
        y_col = prior_y_col.bernoulli().detach() * 2 - 1
        # probability of the sample drawn
        post_y_col = (prior_y_col * y_col - (y_col - 1) / 2)
        
        # Delta0_col = (F_Z(a0_col.detach()) - F_Z(a0_col)) * y_col * 1 / 2
        # Deltai = y_col * (prior_y_col - F_Z(ai)) * ((a0_col.detach() - ai.detach()) * y_col > 0).float()
        Delta0_col = Delta(y_col, a0_col)
        Deltai = Delta(y_col, a0_col.detach(), ai)
        
        # if (Delta0_col < 0).any():
        #     print(Delta0_col.min())
        # if (Deltai < 0).any():
        #     print(Deltai.min())
        
        q_col = q.view([batch_sz, self.in_units, 1])
        Deltaiq_col = Deltai.matmul(q_col)
        p_col = (Delta0_col * (1 - q_col.sum(dim=1, keepdim=True)) + Deltaiq_col) / post_y_col
        # print(post_y_col.min(dim=0)[0])
        
        y = y_col.view([batch_sz, self.out_units])
        p = p_col.view(batch_sz, self.out_units)
        # p = torch.clamp(p, min=0)
        # if (p > 1.0).any():
        #     print(p.max())
        # if (p < 0).any():
        #     print(p.min().item())
        # if (p.sum(dim=1) > 1.0).any():
        #     print(p.sum(dim=1).max().item())

        qmax = torch.max(p.detach().max(), qmax)
        # assume p's are only good for derivatives
        p = p - p.detach()
        # evaluate subsequent layers
        return self.next.forward(y, p, qmax=qmax, p_out=post_y_col, **kwargs)
    
    def fw_SAH(self, x, q=None, qmax=None, p_out=None, **kwargs):
        """ input x [batch, channels]
        """
        
        def F_Z(a):
            return torch.sigmoid(a)
        
        def Delta(y_col, a0, ai=None):
            if ai is None:
                ai = a0
            return -y_col * F_Z(ai)
        
        batch_sz = x.size(0)
        # convert w, b to encoding of x = -1,1
        w = self.weight / 2
        b = self.bias + (self.weight / 2).sum(dim=1)
        if q is None:
            q = torch.zeros_like(x)
        if qmax is None:
            qmax = x.new_zeros(1)
        x_row = x.view(batch_sz, 1, self.in_units)
        
        # compute activations
        a0_col = linear(x, w, b).view([batch_sz, self.out_units, 1])
        
        ai = a0_col - 2 * w * x_row
        # sample augmented state
        prior_y_col = F_Z(a0_col.detach())
        prior_y_col = torch.clamp(prior_y_col, min=0, max=1)
        
        y_col = prior_y_col.bernoulli().detach() * 2 - 1
        post_y_col = (prior_y_col * y_col - (y_col - 1) / 2)
        post_y_col = (F_Z(a0_col) * y_col - (y_col - 1) / 2)
        
        Delta0_col = Delta(y_col, a0_col)
        Deltai = Delta(y_col, a0_col, ai)
        
        q_col = q.view([batch_sz, self.in_units, 1])
        p_col = Delta0_col + (Deltai - Delta0_col).matmul(q_col)
        
        y = y_col.view([batch_sz, self.out_units])
        p = p_col.view(batch_sz, self.out_units)
        # assume p's are only good for derivatives
        p = p - p.detach()
        qmax = torch.max(p.detach().max(), qmax)
        return self.next.forward(y, p, qmax=qmax, p_out=post_y_col, **kwargs)

    def fw_SAH1(self, x, q=None, qmax=None, p_out=None, **kwargs):
        """ input x [batch, channels]
        """
        # convert w, b to encoding of x = -1,1
        w = self.weight / 2
        b = self.bias + (self.weight / 2).sum(dim=1)

        x_out, q_out, p_x_out = linear_binary_inner_SAH(x, q, w, b, last_binary=True)
        return self.next.forward(x_out, q_out, p_out=p_x_out, **kwargs)
    
    def fw_AP2_init(self, X, **kwargs):
        # propagate mean and variance of input, normalize
        with torch.no_grad():
            X = upgrade_to_RV(X)
            Y = RandomVar()
            Y.mean = linear(X.mean, self.weight, self.bias)
            Y.var = linear(X.var, self.weight ** 2)
            # adjust params
            s = torch.sqrt(Y.var + 1e-10)
            self.weight.data = self.weight.data / s.view([-1, 1])
            self.bias.data = (self.bias.data - Y.mean) / s
            # normal statistics
            Y.mean.data.zero_()
            Y.var.data.fill_(1.0)
            # Bernoulli output
            s = torch.sqrt(Y.var + V_S)
            a = Y.mean / s
            Y.mean = Phi_approx(a)
            Y.var = Y.mean * (1 - Y.mean)
        return self.next.forward(Y, **kwargs)


def construct_net(N, out_units):
    l0 = InputLayer(out_units=N[0])
    ll = []
    li = l0
    it = enumerate(N)
    next(it)
    for (i, n) in it:
        li = LogisticBernoulliLayer(out_units=N[i], prev=li)
        ll.append(li)
    OutputLayer(prev=li, out_units=out_units)
    return l0


def train_ML(net, X_all, Y_all, loader, likelihood, NLL, o, call_back=None):
    # create res record
    res = odict()
    res.NLL0 = np.array([])
    res.NLL1 = np.array([])
    res.NLL2 = np.array([])
    res.NLL3 = []
    res.X = X_all
    res.Y = Y_all
    res.kwargs = o
    res.o = o
    res.likelihood = likelihood
    method = o.method
    root_dir = o.root_dir
    exp_name = o.exp_name
    model_name = o.model_name
    lr0 = o.lr0
    #
    t = o.t
    res_file = root_dir + exp_name + '/' + model_name + '.pkl'
    #
    lr = lr0
    # lr_exp_base = 0.97723  # by 0.1 in 100
    lr_exp_base = 1
    # lr_exp_base = math.pow(0.1, 1 / (o.epochs / 2))  # in total epochs decrease by 2 orders
    # lr_exp_base = math.pow(0.1, 1 / (o.epochs))  # in total epochs decrease by 1 orders
    if o.try_load:
        try:
            res = pickle.load(open(res_file, "rb"))
            print('Loaded model')
            net = res.net
            lr = lr0 * 0.1
        except:
            print('Creating new model')
            pass
    #
    if o.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0, momentum=o.SGD_momentum, nesterov=o.SGD_Nesterov)
    elif o.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    elif o.optimizer == 'Rprop':
        optimizer = torch.optim.Rprop(net.parameters(), lr=lr)
    # train
    TE0 = RunningStatAdaptive(0.5, speed=o.batch_size / X_all.size(0))
    TE1 = RunningStatAdaptive(0.5)
    TE2 = RunningStat(0.9)
    P_running = [RunningStatAdaptive(0.2) for i in range(X_all.size(0))]
    # TE1_var = RunningStatAdaptive(0.5)
    
    for epoch in range(o.epochs):
        lr_factor = math.pow(lr_exp_base, epoch)
        # full batch gradient, 100 samples per point
        # get data batch:
        
        # estimate true likelihood: running average of p per training example
        TE3 = RunningStatAvg()
        Emean = 0
        for s in range(o.n_samples_ref):
            obj_sample = lambda eta: likelihood(eta, Y_all)
            EE = net.forward(X_all, **dict(o, method='score'), obj=obj_sample)
            Emean += EE.detach() / o.n_samples_ref
        TE1.update(Emean) # update once in epoch
        TE2.update(Emean)
        TE3.update(Emean)
        
        for batch_ndx, sample in enumerate(loader):
            X = sample[0]  # training batch
            Y = sample[1]
            ii = sample[2]
            if o.method == 'AP1' or (o.method == 'enumerate' and o.ML is False):  # deterministic methods: full likelihood at once
                net.zero_grad()
                obj_batch = lambda eta: NLL(eta, Y)
                EE = net.forward(X, **o, obj=obj_batch)  # computes expected likelihood per training sample
                EEmean = EE.mean()  # average over samples
                EEmean.backward(retain_graph=False)
                TE0.update(-EEmean.detach())
                for p in net.parameters():
                    p.grad = p.grad * lr_factor
            elif o.method == 'enumerate' and o.ML:  # deterministic sum over all configurations
                net.zero_grad()
                obj_batch = lambda eta: likelihood(eta, Y)
                EE = net.forward(X, **o, obj=obj_batch)
                EEmean = -EE.log().mean()  # NLL, average over training samples in batch
                EEmean.backward(retain_graph=False)
                TE0.update(-EEmean.detach())  # record log likelihood
                for p in net.parameters():
                    p.grad = p.grad * lr_factor
            else:  # stochastic methods likelihood
                for p in net.parameters():
                    p.acc_grad = 0
                batch_LL = X.new_zeros(X.size(0))
                for i in range(X.size(0)):
                    net.zero_grad()
                    Xs = X[i].view([1, -1]).expand(o.n_samples, -1)
                    Ys = Y[i].view([1, -1]).expand(o.n_samples, -1)
                    if o.ML is None or o.ML is True:  # stochastic methods: multi-sample likelihood
                        obj_point = lambda eta: likelihood(eta, Ys)
                        if o.method == 'score' and o.ML_grad == 'LB' and False:
                            EE = net.forward(Xs, **o, obj=obj_point, multi_sample_ML=True)
                            LL = EE  # returns n-sample log likelihood
                            batch_LL[i] = LL.detach()
                        if o.ML_grad == 'LB':  # estimate the gradient of the multi-sample lower bound on likelihood
                            EE = net.forward(Xs, **o, obj=obj_point)
                            # we compute EE.mean().log() in a robust way
                            LL = torch.logsumexp(EE.log(), dim=0) - math.log(o.n_samples)  # n_sample log likelihood
                            batch_LL[i] = LL.detach()
                            LL.backward()
                            for p in net.parameters():
                                check_real(p.grad)
                                p.acc_grad += p.grad * lr_factor
                        if o.ML_grad == 'LB':  # estimate the gradient of the multi-sample lower bound on likelihood
                            EE = net.forward(Xs, **o, obj=obj_point) # predictive probability per sample
                            # we compute EE.mean().log() in a robust way
                            LL = torch.logsumexp(EE.log(), dim=0) - math.log(o.n_samples)  # n_sample log likelihood
                            batch_LL[i] = LL.detach()
                            LL.backward()
                            for p in net.parameters():
                                p.acc_grad += p.grad * lr_factor
                        elif o.ML_grad == 'LB_EM':  # estimate the gradient of the multi-sample lower bound on likelihood, optimized weights
                            LL = net.forward(Xs, **o, obj=obj_point, return_logp=True)
                            batch_LL[i] = LL.detach()
                            LL.backward()
                            for p in net.parameters():
                                p.acc_grad += p.grad * lr_factor
                        else: # running average estimate (empirical trial, no LB)
                            # problem : need to remember running gradient per data point
                            batch_LL[i] = LL.detach()
                            EEmean = EE.mean()  # average probability over samples
                            EEmean.backward(retain_graph=False)  # multi-sample gradient of likelihood
                            P_running[ii[i]].update(EEmean.detach())  # running sum of probabilities
                            # P = TE1.mean[ii[i]]  # running average P for this training point
                            P = P_running[ii[i]].mean
                            for p in net.parameters():
                                p.acc_grad = p.acc_grad + p.grad / P * lr_factor
                    
                    else:  # stochastic methods: log likelihood
                        obj_point = lambda eta: -NLL(eta, Ys)
                        EE = net.forward(Xs, **o, obj=obj_point)
                        EEmean = EE.mean()  # average over samples
                        # AE[i] = (EEmean.detach()).exp()
                        batch_LL[i] = EEmean.detach()
                        EEmean.backward(retain_graph=False)
                        for p in net.parameters():
                            p.acc_grad += p.grad * lr_factor
                for p in net.parameters():
                    p.grad = -p.acc_grad / X.size(0)  # maximizing, mean over data points
                
                # TE0.update(AE.log())
                TE0.update(batch_LL.mean()) # update once per batch
            
            # optimize only in parameters of the last layer, make grad in all other layers zero
            # l = net.next
            # while l.next is not None:
            #     l.weight.grad.zero_()
            #     l.bias.grad.zero_()
            #     l = l.next
            
            optimizer.step()
        
        # total objective
        NLL0 = -TE0.mean.item()  # own objective (lower bound)
        NLL1 = -TE1.mean.log().mean().item()  # mean over epochs, NLL, mean over data points
        NLL2 = -TE2.mean.log().mean().item()
        print('epoch:{:3d} NLL0:{:.4f} NLL1:{:.4f}'.format(epoch, NLL0, NLL1))
        res.NLL0 = np.append(res.NLL0, NLL0)
        res.NLL1 = np.append(res.NLL1, NLL1)
        res.NLL2 = np.append(res.NLL2, NLL2)
        res.NLL3.append(copy.deepcopy(TE3))
        
        # if (epoch + 1) % (o.epochs // 2 + 1) == 0:  # save and evaluate model
        # if epoch == o.epochs // 2 - 1 or epoch == o.epochs - 1:
        if epoch + 1 in o.checkpoints or epoch + 1 == o.epochs:
            res.net = net
            res_file1 = root_dir + exp_name + '/' + 'chk-{}'.format(epoch + 1) + '/' + model_name + '.pkl'
            force_path(res_file1)
            save_object(res_file1, res)
            force_path(res_file)
            save_object(res_file, res)
            # plot right away
            if call_back is not None:
                call_back()


def test1():
    # architecture -- units per layer
    # N = [1]
    # N = [1, 1]
    # N = [2, 2]
    # N = [3, 3, 3, 3, 3]
    # N = [1, 3, 1, 3]
    
    N = [2, 3, 2, 5, 4]
    
    # fore reproducibility
    seed = 2
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    l0 = InputLayer(out_units=N[0])
    ll = []
    li = l0
    it = enumerate(N)
    next(it)
    for (i, n) in it:
        li = LogisticBernoulliLayer(out_units=N[i], prev=li)
        ll.append(li)
    lN = OutputLayer(prev=li)
    net = l0
    
    x = torch.empty(N[0]).uniform_()
    x.fill_(2.0)
    
    # x = x.cuda()
    # l0.cuda()
    
    # include gradient in x into evaluation
    # x.requires_grad_(True)
    x.requires_grad_(False)
    
    # test correlated case
    # if True:
    #     for li in ll:
    #         li.bias.data.zero_()
    #         li.weight.data.fill_(0)
    #
    #     x.fill_(1)
    #     ll[1].weight.data.fill_(10)
    #     ll[1].bias.data.fill_(-5)
    
    # test expectation
    E = net.forward(x, method='enumerate')
    print('E={}'.format(E.item()))
    
    # exact gradient
    g_t = net.grad(x, method='enumerate', output='list')
    print('GT gradient as list:')
    print(g_t)
    
    # for i in range(10):
    #     g = net.grad(x, method='SAH', output='list')
    #     print(g)
    
    # exact gradient
    g_t = net.grad(x, method='enumerate', output='vector')
    # print(g_t)
    #
    # sampling based gradients
    n_samples = 100
    
    sample_batch = 1  # compute average gradient for a batch at once, currently implemented only for SA (sample-analytic method)
    
    # for method in ['score', 'concrete', 'SA', 'SAH', 'SAH1']:
    for method in ['SAH', 'SAH1']:
        seed = 2
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        print("_______________________ Method: %s" % method)
        g_s = RunningStatAvg()  # computes running mean and variance
        sim = RunningStatAvg()  # cosine similarity statistics
        MSE = RunningStatAvg()
        for g in net.grad(x, method=method, n_samples=n_samples, sample_batch=sample_batch, tau=0.1, output='vector', t=0):
            # print(g)
            g_s.update(g)
            c = F.cosine_similarity(g, g_t, dim=0)
            sim.update(c)
            MSE.update(((g - g_t) ** 2).sum())
        # print("Abs deviation of the mean: %f" % (g_s.mean - g_t).abs().sum())
        Bias = g_s.mean - g_t
        print("Av Bias: %f  +- %f" % ((Bias).abs().mean(), g_s.std_of_mean.mean()))
        VN = g_s.var.mean().item()
        print("RMSE: %f" % (MSE.mean.sqrt()))
        if method is 'SA':
            V1 = VN * sample_batch
            # MSE = (Bias ** 2 + 1 / n_samples * V1).sum()
            MSE = (Bias ** 2 + V1).sum()
            RMSE = MSE.sqrt()
        else:
            V1 = VN
            RMSE = MSE.mean.sqrt()
        print("RMSE: %f" % (RMSE))
        # print("Variance per parameter: %f" % (g_s.var.mean().item()))
        print("Variance per parameter: %f" % V1)
        print("Cosine similarity: %f +- %f" % (sim.mean, sim.var ** 0.5))
        # print(g_s.mean)


def logistic_likelihood(eta, y):
    return torch.sigmoid(eta * y)


def logistic_NLL(eta, y):
    return log1p_exp(-eta * y)


def likelihood(eta, y):
    return torch.sigmoid(eta * y)


def NLL(eta, y):
    return log1p_exp(-eta * y)


if __name__ == "__main__":
    test1()
