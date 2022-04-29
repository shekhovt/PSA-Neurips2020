import numpy as np
import torch
import torch.nn.functional as F

def compute_delta_linear_dumb(x, q, W):
    a0 = F.linear(x, W)
    p0 = torch.sigmoid(a0)
    new_x = 2 * p0.bernoulli() - 1

    delta_0 = -new_x * (p0 - p0.detach())
    new_q = delta_0

    mask = torch.ones(x.shape[1])

    for i in range(x.shape[1]):
        mask[i] = -1
        ai = F.linear(x * mask[None], W)
        pi = torch.sigmoid(ai)
        delta_i = -new_x * (pi - p0.detach())
        new_q = new_q + q[:, i] * (delta_i - delta_0)
        mask[i] = 1

    return new_x, new_q

def compute_delta_linear_vec(x, q, W):
    a0 = F.linear(x, W)
    p0 = torch.sigmoid(a0)
    a = a0[:, :, None] - 2 * x[:, None] * W[None]
    p = torch.sigmoid(a)

    new_x = 2 * p0.bernoulli() - 1

    delta_0 = -new_x * (p0 - p0.detach())
    delta = -new_x[:, :, None] * (p - p0[:, :, None].detach())

    new_q = delta_0.mul(1 - q.sum(dim=1, keepdim=True)) + delta.bmm(q[:, :, None]).squeeze(-1)
    return new_x, new_q

def test_compute_delta_linear(B, I, O, seed=1234):
    x = torch.randn([B, I]).sign()
    q = torch.rand([B, I])
    W = torch.randn([O, I])

    torch.manual_seed(seed)
    new_x_1, new_q_1 = compute_delta_linear_dumb(x, q, W)
    
    torch.manual_seed(seed)
    new_x_2, new_q_2 = compute_delta_linear_vec(x, q, W)

    assert np.all((new_q_1 - new_q_2).abs().numpy() < 1e-6)
    print("linear OK")

def compute_delta_conv_dumb(x, q, W):
    k = W.shape[-1]
    a0 = F.conv2d(x, W, padding=k//2)
    p0 = torch.sigmoid(a0)
    new_x = 2 * p0.bernoulli() - 1

    delta_0 = -new_x * (p0 - p0.detach())
    new_q = delta_0

    mask = torch.ones(x.shape[1:])

    for c in range(x.shape[1]):
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                mask[c, i, j] = -1
                ai = F.conv2d(x * mask[None], W, padding=k//2)
                pi = torch.sigmoid(ai)
                delta_i = -new_x * (pi - p0)
                new_q = new_q + q[:, c, i, j].view(-1, 1, 1, 1) * (delta_i - delta_0)
                mask[c, i, j] = 1

    return new_x, new_q

def _compute_conv_diffs(x, K, a0):
    B, C_, W_, H_ = a0.shape
    _, C, W, H = x.shape
    _, _, k, _ = K.shape

    a = torch.zeros([k, k, B, C_, C, W, H])
    K_flip = torch.flip(K, [2, 3])

    a0_ = torch.zeros([B, C_, W_ + k, H_ + k])
    a0_[:, :, k//2:-k//2, k//2:-k//2] = a0
    
    for l1 in range(k):
        for l2 in range(k):
            val = a0_[:, :, None, l1:l1+W, l2:l2+W] - 2 * x[:, None] * K_flip[None, :, :, l1, l2, None, None]
            a[l1, l2] = val
    
    return a
    

def _compute_conv_sum(new_x, p, p0, q):
    k, _, B, C_, C, W, H = p.shape
    _, _, W_, H_ = new_x.shape

    pad = k//2
    delta_q = torch.zeros([B, C_, W_ + 2 * pad, H_ + 2 * pad])

    p0_ = torch.zeros([B, C_, W_ + 2 * pad, H_ + 2 * pad])
    p0_[:, :, pad:-pad, pad:-pad] = p0

    y_ = torch.zeros([B, C_, W_ + 2 * pad, H_ + 2 * pad])
    y_[:, :, pad:-pad, pad:-pad] = new_x

    for l1 in range(k):
        for l2 in range(k):
            add = p[l1, l2].sub(p0_[:, :, None, l1:l1+W, l2:l2+H]).mul(q[:, None]).sum(dim=2).mul(-y_[:, :, l1:l1+W, l2:l2+H])
            delta_q[:, :, l1:l1+W, l2:l2+H] += add

    return delta_q[:, :, pad:-pad, pad:-pad]


def compute_delta_conv_vec(x, q, W):
    k = W.shape[-1]
    a0 = F.conv2d(x, W, padding=k//2)
    p0 = torch.sigmoid(a0)
    a = _compute_conv_diffs(x, W, a0)
    p = torch.sigmoid(a)

    new_x = 2 * p0.bernoulli() - 1

    delta_0 = -new_x * (p0 - p0.detach())
    delta_q = _compute_conv_sum(new_x, p, p0, q)

    qs = q.sum(dim=3, keepdim=True)
    qs = qs.sum(dim=2, keepdim=True)
    qs = qs.sum(dim=1, keepdim=True)

    new_q = delta_0.mul(1 - qs) + delta_q
    return new_x, new_q

def test_compute_delta_conv(B, C, W, H, k, C_, seed=1234):
    torch.manual_seed(seed)
    x = torch.randn([B, C, W, H]).sign()
    q = torch.rand([B, C, W, H])
    W = torch.randn([C_, C, k, k])

    torch.manual_seed(seed)
    new_x_1, new_q_1 = compute_delta_conv_dumb(x, q, W)
    
    torch.manual_seed(seed)
    new_x_2, new_q_2 = compute_delta_conv_vec(x, q, W)

    assert np.all((new_q_1 - new_q_2).abs().numpy() < 1e-5)
    print("conv OK")

#test_compute_delta_linear(B=1, I=2, O=3)

#test_compute_delta_conv(B=1, C=1, W=5, H=5, k=3, C_=1)
#test_compute_delta_conv(B=1, C=1, W=5, H=5, k=3, C_=1)
test_compute_delta_conv(B=10, C=4, W=5, H=5, k=5, C_=3)
