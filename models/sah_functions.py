import numpy as np
import torch
import torch.nn.functional as F
import context
from extensions import ratio_conv2d_backward
from .utils import *


class LinearBinary_SAH(torch.autograd.Function):
    """
    This class implements custom gradient for the "hard" part of the computation -> to be implemented in CUDA
    """
    
    @staticmethod
    def forward(ctx, q, x_in, x_out, a0, w):
        """
        The function computes q_new_linear = sum_i \Delta_{i,j} q_i,
        where Delta_ij = - x_out_j * sigmoid (a_{0,j} - 2*w_{j,i} * x_in_i)
        """
        q_out_linear = torch.zeros_like(x_out, requires_grad=True)  # maybe latter we will want to compute it, but for now only going to compute its gradient
        ctx.save_for_backward(q, x_in, x_out, a0, w)
        return q_out_linear
    
    @staticmethod
    def backward(ctx, g_out):
        """
            Here really compute: g_in_i = sum_j \Delta_{i,j} g_out_j,
            g_out gradient of loss in q_out_linear
        """
        q, x_in, x_out, a0, w = ctx.saved_tensors
        g_in = None
        if ctx.needs_input_grad[0]:  # q needs grad
            # Reference solution: compute full matrix of Delta_ij and multiply
            # a_bij = a0_bj - 2 * w_{ji} * x_in{b,i}
            a = a0[:, None, :] + torch.einsum("ji, bi -> bij", w, -2 * x_in)
            # g_in_bi = sum_j sigmoid(a_{bij}) * (-x_out_bj * g_out_bj)
            g_in = torch.einsum("bij, bj -> bi", torch.sigmoid(a), -x_out * g_out)
            check_real(g_in)
        return g_in, None, None, None, None

# alias for convenient call
linearBinary_SAH = LinearBinary_SAH.apply

def linear_binary_inner_SAH(x, q, weight, bias, last_binary = False):
    """
    An inner stochastic binary layer
    
    :param x: - input state Tensor [B C_in] +/-1 (also good for real input with zero flip probabilities not requiring grad)
    :param q: - input state linearized flip probabilities (assume very small) [B C_in]
    :param weight: - layer weight [C_out C_in]
    :param bias: - layer bias [C_out]
    :return:
    x_out - output binary state [B C_out] +/-1
    q_out - its linearized flip probabilities [B C_out]
    p_x_out - probabilities of having generated x_out [B C_out]
    """
    
    if q is None:
        q = torch.zeros_like(x)
    
    a0 = F.linear(x, weight, bias=bias)
    check_real(a0)
    p0 = a0.sigmoid()
    with torch.no_grad():
        # x_out = p0.bernoulli() * 2 - 1
        x_out = sign_bernoulli(p0)
    Delta0 = -x_out * p0
    # the part that is easy to compute and good for automatic differentiation
    q_out = Delta0 - torch.einsum("bj, bi -> bj", Delta0.detach(), q)
    # the difficult part
    q_out += linearBinary_SAH(q, x, x_out, a0, weight)
    # assume q_out are only good for derivatives
    q_out = q_out - q_out.detach()
    if not last_binary:
        return x_out, q_out
    else:
        # probability of the sampled state for improved last layer estimate
        p_x_out = (p0 * x_out - (x_out - 1) / 2).detach()
        return x_out, q_out, p_x_out

"""__________________depricated, use vec version below________________"""
# def binary_out_SAH(x, q, p_x, obj):
#     """
#     Last layer with generic function of binary variables
#
#     :param x: - input state [B C]
#     :param q: - input state linearized flip probabilities (assume very small) [B C]
#     :param p_x: - probabilities of having generated x [B C]
#     :param obj: - target objective function, callable(Tensor[B C])
#     :return:
#     E - expected objective value, differentiable same shape as what obj returns
#     """
#
#     batch_sz = x.size(0)
#     n = x.size(1)
#     p_out = p_x.view([batch_sz, -1, 1])
#     q = q.view([batch_sz, -1, 1])
#     p_flip = (1 - p_out) / n + q  # total flip probability
#     f_base = obj(x)
#     E = f_base
#     for i in range(n):
#         x1 = x.clone()
#         x1[:, i] = - x[:, i]
#         f_flip = obj(x1)
#         E = E + (f_flip - f_base) * p_flip[:, i, :]
#     return E

def binary_out_SAH_vec(x, q, p_x, weight, bias, objs):
    """
    Last layer with linear and a generic function on top

    :param x: - input state [B C_in]
    :param q: - input state linearized flip probabilities (assume very small) [B C_in]
    :param weight: [C_out C_in]
    :param bias: [C_out]
    :param obj: - target objective function, callable(float Tensor[B S C_out]), S - dimension for gradient test samples
    :return:
    E - expected objective value, differentiable same shape as what obj returns
    """
    batch_sz, C_in = x.shape
    a0 = F.linear(x, weight, bias=bias)  # [B C_out]
    # a_bij = a0_bj - 2*w_ij * x_i
    a = a0[:, None, :] + torch.einsum("ji, bi -> bij", weight, -2 * x) # [B C_in C_out]
    C_out = a0.size(1)
    p_flip = (1 - p_x) / C_in + q  # total flip probability [B C_in]
    # combine dimension i as a batch
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


class ConvBinary_SAH(torch.autograd.Function):
    """
    This class implements custom gradient for the "hard" part of the computation -> to be implemented in CUDA
    """
    
    @staticmethod
    def forward(ctx, q, x_in, x_out, a0, w, stride=1, padding=0):
        """
        :param q:     [B C_in H W]
        :param x_in:  [B C_in H W]
        :param x_out: [B C_out H1 W1]
        :param a0:    [B C_out H1 W1]
        :param w:     [C_out C_in K K]
        The function computes q_new_linear_{c,j} = sum_{r,@i} \Delta_{c,r,i-j} q_{r,i},
        where Delta_{c,r,i-j} = - x_out_{c,j} * sigmoid (a0_{c,j} - 2*w_{c,r,i-j} * x_in_{r,i} )
        """
        q_out_linear = torch.zeros_like(x_out, requires_grad=True)  # maybe latter we will want to compute it, but for now only going to compute its gradient
        if isinstance(stride, tuple):
            stride = stride[0]
        if isinstance(padding, tuple):
            padding = padding[0]
        ctx.save_for_backward(q, x_in, x_out, a0, w, torch.tensor([stride], dtype=int), torch.tensor([padding], dtype=int))
        return q_out_linear
    
    @staticmethod
    def backward(ctx, g_out):
        """
            Here really compute:
            g_in_{r,i} = sum_{c,@j} \Delta_{c,r,i-j} g_out_{c,j}
            g_out gradient of loss in q_out_linear
        """
        q, x_in, x_out, a0, w, stride, padding = ctx.saved_tensors
        stride = stride.item()
        padding = padding.item()
        #padding = ps[0].item()
        #stride = ps[1].item()
        g_in = None
        if ctx.needs_input_grad[0]:  # q needs grad
            # sum_{c,@j}  - x_out_{c,j} * sigmoid (a0_{c,j} - 2*w_{c,r,i-j} * x_in_{r,i}) * g_out_{c,j}
            #
            # first compute sum_{c,@j} b_{c,j} * sigmoid(a0_{c,j} + 2*w_{c,r,i-j}) (for x_ri = -1)
            # reshape it to conv2d format (we are going to do derivative of conv2d, which is like conv2d_transposed):
            W = w.permute([1, 0, 2, 3]).flip(dims=[2, 3])  # "transposed" kernel [C_in C_out, K, K]
            # premultiply
            g = - x_out * g_out  # [B C_out H1 W1]
            # pre-exponentiate:
            a0exp = torch.exp(-a0)  # [B C_out H1 W1]
            Wexp_p = torch.exp(+ 2 * W)  # [C_in C_out, K, K]
            Wexp_m = torch.exp(- 2 * W)  # [C_in C_out, K, K]
            # compute:
            # for x_ri = -1: sum_{c,@j} g_{c,j} * 1/(1 + exp(-a0)_{c,j} * exp(-2*W)_{r, c, j-i} )
            # for x_ri = +1: sum_{c,@j} g_{c,j} * 1/(1 + exp(-a0)_{c,j} * exp(+2*W)_{r, c, j-i} )
            # print("Backward: ", g.shape, a0exp.shape, Wexp_p.shape, x_in.shape)
            g_in = ratio_conv2d_backward(g, a0exp, Wexp_p, Wexp_m, x_in, stride=stride, padding=padding, impl_v=1)[0]
            check_real(g_in)
        return g_in, None, None, None, None, None, None

# alias for convenient call
convBinary_SAH = ConvBinary_SAH.apply


def conv_binary_first(x, weight, bias, padding=0, stride=1, last_binary=False):
    """
    The first layer with real-valued inputs and binary outputs

    :param x: - input state Tensor [B C_in H W], real
    :param weight: - layer weight [C_out C_in K K]
    :param bias: - layer bias [C_out]
    :return:
    x_out - output binary state [B C_out H1 W1] +/-1
    q_out - its linearized flip probabilities [B C_out H1 W1]
    """
    # degub
    #cw = weight.cpu()
    #cb = bias.cpu()
    #cx = x.cpu()
    # degub
    a0 = F.conv2d(x, weight, bias=bias, padding=padding, stride=stride)  # [B C_out H1 W1]
    #ca0 = a0.cpu()
    check_real(a0)
    p0 = a0.sigmoid()
    #cp0 = p0.cpu()
    #x_out = p0.clamp(1e-6, 1-1e-6).bernoulli().detach() * 2 - 1
    x_out = sign_bernoulli(p0)
    Delta0 = -x_out * p0  # [B C_out H1 W1]
    q_out = Delta0
    # assume q_out are only good for derivatives
    q_out = q_out - q_out.detach()

    if not last_binary:
        return x_out, q_out
    else:
        # probability of the sampled state for improved last layer estimate
        p_x_out = (p0 * x_out - (x_out - 1) / 2).detach()
        return x_out, q_out, p_x_out



def conv_binary_inner_SAH(x, q, weight, bias, x_out=None, padding=0, stride=1, last_binary = False):
    """
    An inner stochastic binary layer

    :param x: - input state Tensor [B C_in H W], +/-1
    :param q: - input state linearized flip probabilities (assume very small) [B C_in H W]
    :param weight: - layer weight [C_out C_in K K]
    :param bias: - layer bias [C_out]
    :param x_out: - output state sample (for debugging)
    :return:
    x_out - output binary state [B C_out H1 W1] +/-1
    q_out - its linearized flip probabilities [B C_out H1 W1]
    """

    assert x.size() == q.size()
    assert x.size(1) == weight.size(1)
    if bias is not None:
        assert weight.size(0) == bias.size(0)
    assert weight.size(2) % 2 == 1, "Kernel size must be odd"
    assert weight.size(3) % 2 == 1, "Kernel size must be odd"

    if q is None:
        q = torch.zeros_like(x)

    a0 = F.conv2d(x, weight, bias=bias, padding=padding, stride=stride)  # [B C_out H1 W1]
    check_real(a0)
    p0 = a0.sigmoid()
    if x_out is None:
        #x_out = p0.clamp(1e-6, 1 - 1e-6).bernoulli().detach() * 2 - 1
        x_out = sign_bernoulli(p0)
    Delta0 = -x_out * p0  # [B C_out H1 W1]
    # the part that is easy to compute and good for automatic differentiation
    # q_out_{b,c,j} = Delta0_{b,c,j} * (1 - sum_{r,@i} q_{b,r,i})
    w_one = weight.new_ones([1, 1, 1, 1]).expand([1, *weight.size()[1:]])  # expand all dimension except output
    q_out = Delta0 * (1 - F.conv2d(q, w_one, padding=padding, stride=stride))
    # the difficult part
    # print(q.shape, x.shape, x_out.shape, a0.shape, weight.shape)
    q_out += convBinary_SAH(q, x, x_out, a0, weight, stride, padding)
    # assume q_out are only good for derivatives
    q_out = q_out - q_out.detach()
    #
    if not last_binary:
        return x_out, q_out
    else:
        # probability of the sampled state for improved last layer estimate
        p_x_out = (p0 * x_out - (x_out - 1) / 2).detach()
        return x_out, q_out, p_x_out


def linear_01_binary_SAH(x, q, A, scale, bias, x_out=None, last_binary=False, **kwargs):
    # mask where inputs = 1
    mask_p = (x + 1) / 2
    # mask where inputs = -1
    mask_m = (1 - x) / 2
    #
    a0 = A(x) * scale + bias
    p0 = a0.sigmoid()
    if x_out is None:
        x_out = sign_bernoulli(p0)
    Delta0 = -x_out * p0

    # positive flips down:
    a0_p = a0 - 2 * scale
    # negative flips up:
    a0_m = a0 + 2 * scale
    
    Delta1_p = -x_out * a0_p.sigmoid()
    Delta1_m = -x_out * a0_m.sigmoid()

    q_p = A(q * mask_p)  # a positive input flips down
    q_m = A(q * mask_m)  # a negative input flips up
    #
    #
    q_out = Delta0 + (Delta1_p - Delta0).detach() * q_p + (Delta1_m - Delta0).detach() * q_m
    #
    q_out = q_out - q_out.detach()
    if not last_binary:
        return x_out, q_out
    else:
        # probability of the sampled state for improved last layer estimate
        p_x_out = (p0 * x_out - (x_out - 1) / 2).detach()
        return x_out, q_out, p_x_out
    

""" depricated, see models.methods.py and models.losses.py """
# def binary_out_SAH_vec_mse(x, q, p_x, weight, bias, y):
#     """
#     Last layer with linear and MSE on top
#
#     :param x: - input state [B C_in]
#     :param q: - input state linearized flip probabilities (assume very small) [B C_in]
#     :param weight: [C_out C_in]
#     :param bias: [C_out]
#     :return:
#     E - expected objective value, differentiable same shape as what obj returns
#     """
#     batch_sz, C_in = x.shape
#     wqx = F.linear((1 - q) * x, weight, bias=bias)  # [B C_out]
#     wx = F.linear((1 - 2 * q) * x, weight, bias=bias)
#     qw_norm = (weight * weight).sum(dim=1).mul(1 - (1 - q).pow(2)).sum()
#     return y.pow(2).sum() - y.mul(wx).sum() * 2 + wqx.pow(2).sum() - qw_norm
