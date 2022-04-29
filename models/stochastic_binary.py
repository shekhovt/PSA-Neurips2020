import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticBinaryLinear(nn.Linear):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lin = nn.Linear(input_size, output_size)

    
    def forward(self, x, q):
        a_0 = self.lin(x)
        a = a_0[:, :, None] - 2 * self.lin.weight[None] * x[:, None]

        probs = torch.sigmoid(a_0)
        with torch.no_grad():
            b = torch.bernoulli(probs)
            new_x = 2 * b - 1 

        delta_0 = -new_x * probs
        delta = -new_x[:, :, None] * torch.sigmoid(a)

        new_q = delta_0 + q[:, None].mul(delta - delta_0[:, :, None]).sum(dim=-1)
        return new_x, new_q

class StochasticBinaryLinear_2(nn.Module):
    def __init__(self, input_size, output_size, fast=False, t=0.):
        super().__init__()
        self.lin = nn.Linear(input_size, output_size)
        self.fast = fast
        self.t = t
        self._t = torch.tensor([t])
        self._reset_parameters()

    
    def _reset_parameters(self):
        gain = torch.nn.init.calculate_gain('sigmoid')
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=gain)

    
    def forward(self, x, q):
        if self.fast:
            return self._fast_forward(x, q)
        else:
            return self._norm_forward(x, q)


    def _fast_forward(self, x, q):
        a_0 = self.lin(x) 
        a = a_0.unsqueeze(-1) - 2 * torch.einsum('ji,bi->bji', (self.lin.weight, x))
        #probs = torch.sigmoid(a_0) 

        if self.t == 0.:
            probs = torch.sigmoid(a_0)
        else:
            probs = (F.softplus(a_0 + self.t) - F.softplus(a_0 - self.t)) / (2 * self.t)
            probs = probs.clamp_(0., 1.)
        with torch.no_grad():
            b = torch.bernoulli(probs)
            new_x = 2 * b - 1

        probs_ = probs.detach()

        ###########################################################################################
        #delta_0 = -new_x * probs # delta_0.shape = B x O
        #delta = torch.einsum('bj,bji->bji', (-new_x, torch.sigmoid(a)))

        p0 = torch.sigmoid(a_0)
        p = torch.sigmoid(a)
        if self._t.device != a.device:
            self._t = self._t.to(a.device)

        if self.t == 0.:
            delta_0 = -new_x * p0 
            delta = torch.einsum('bj,bji->bji', (-new_x, p))
        else:
            delta_0 = -0.5 * new_x * p0 + F.softplus(a_0 + self.t * new_x) / (2 * self.t)
            b = torch.min(torch.max(a - a_0.unsqueeze(-1), -self._t), self._t)
            pi = b / (2 * self.t) - new_x.unsqueeze(-1) / 2
            delta = pi * p + (F.softplus(a_0 + self.t * new_x).unsqueeze(-1) - F.softplus(a_0.unsqueeze(-1) + b)) / (2 * self.t)

        ###########################################################################################
        #new_q = delta_0.mul(1 - q.sum(dim=1, keepdim=True)) + torch.einsum('bi,bji->bj', (q, delta))
        new_q = b + delta_0.mul(1 - q.sum(dim=1, keepdim=True)) + torch.einsum('bi,bji->bj', (q, delta))
        return new_x, new_q


    def _norm_forward(self, x, q):
        a_0 = self.lin(x)
        a_0_ = a_0.detach()
        a = a_0.unsqueeze(-1) - 2 * torch.einsum('ji,bi->bji', (self.lin.weight, x))
        
        if self.t == 0.:
            probs = torch.sigmoid(a_0)
        else:
            probs = (F.softplus(a_0 + self.t) - F.softplus(a_0 - self.t)) / (2 * self.t)
            probs = probs.clamp_(0., 1.)
        probs_ = probs.detach()

        with torch.no_grad():
            b = torch.bernoulli(probs)
            new_x = 2 * b - 1

        pi = probs_ * b + (1 - probs_) * (1 - b)

        #delta_0 = -new_x * (probs - probs_) # delta_0.shape = B x O
        #delta = torch.einsum('bj,bji->bji', (-new_x, (torch.sigmoid(a) - probs_.unsqueeze(2))))
        delta_0, delta = self._compute_delta(new_x, a_0, a, t=self.t)
        new_q = delta_0.mul(1 - q.sum(dim=1, keepdim=True)) + torch.einsum('bi,bji->bj', (q, delta))
        new_q = new_q / pi
        return new_x, new_q


    def _compute_delta(self, new_x, a0, a, t=0.):
        p0 = torch.sigmoid(a0)
        p = torch.sigmoid(a)
        if self._t.device != a.device:
            self._t = self._t.to(a.device)

        _t = self._t
        a0_, p0_ = a0.detach(), p0.detach()

        if t == 0.:
            delta_0 = -new_x * (p0 - p0_) 
            delta = torch.einsum('bj,bji->bji', (-new_x, (p - p0_.unsqueeze(2))))
        else:
            delta_0 = -0.5 * new_x * p0 + (F.softplus(a0_ + t * new_x) - F.softplus(a0_)) / (2 * t)
            b = torch.min(torch.max(a - a0_.unsqueeze(-1), -_t), _t)
            pi = b / (2 * t) - new_x.unsqueeze(-1) / 2
            delta = pi * p + (F.softplus(a0_ + t * new_x).unsqueeze(-1) - F.softplus(a0_.unsqueeze(-1) + b)) / (2 * t)

        return delta_0, delta


#class StochasticBinaryFullyConnected(nn.Module):
#    def __init__(self, args):
#        super().__init__()
#        self.fc_layers, self.output_layer = self._make_layers(args)
#        self.input_size = args.input_size
#
#
#    def _make_layers(self, args):
#        layers = []
#        input_size = args.input_size
#
#        for _ in range(args.n_layers):
#            layers.append(StochasticBinaryLinear(input_size, args.hidden_size))
#            input_size = args.hidden_size
#
#        mask = [torch.ones(input_size)]
#        for i in range(input_size):
#            m = torch.ones(input_size)
#            m[i] = -1
#            mask.append(m)
#        self.mask = torch.stack(mask, dim=0)
#
#        output_layer = nn.Linear(input_size, args.n_classes)
#        return nn.ModuleList(layers), output_layer
#
#    
#    def forward(self, x):
#        if x.device != self.mask.device:
#            self.mask = self.mask.to(x.device)
#
#        z = x.view(-1, self.input_size)
#        q = torch.zeros([x.shape[0], self.input_size], device=z.device)
#        for layer in self.fc_layers:
#            z, q = layer(z, q)
#
#        batch_size, size = z.shape
#        inps = torch.stack([z for _ in range(size + 1)], dim=0) # inps.shape = O + 1 x B x O
#        inps = inps.mul(self.mask[:, None, :]).view(-1, size)
#        f = self.output_layer(inps).view(size + 1, batch_size, -1) # O x B x D
#        out = f[0] + (f[1:] - f[:1]).mul(q.t()[:, :, None]).sum(dim=0)
#        return out
#
#class StochasticBinaryFullyConnected_2(nn.Module):
#    def __init__(self, args):
#        super().__init__()
#        self.fc_layers, self.output_layer = self._make_layers(args)
#        self.input_size = args.input_size
#        self.n_classes = args.n_classes
#        self.expect_pred = args.expect_pred
#        self.expect_probs = args.expect_probs
#
#
#    def _make_layers(self, args):
#        layers = []
#        input_size = args.input_size
#
#        for _ in range(args.n_layers):
#            layers.append(StochasticBinaryLinear_2(input_size, args.hidden_size, fast=args.fast, t=args.t))
#            input_size = args.hidden_size
#
#        mask = [torch.ones(input_size)]
#        for i in range(input_size):
#            m = torch.ones(input_size)
#            m[i] = -1
#            mask.append(m)
#        self.mask = torch.stack(mask, dim=0)
#
#        output_layer = nn.Linear(input_size, args.n_classes)
#        return nn.ModuleList(layers), output_layer
#
#
#    def forward(self, x):
#        if x.device != self.mask.device:
#            self.mask = self.mask.to(x.device)
#
#        z = x.view(-1, self.input_size)
#        q = torch.zeros_like(z).to(z)
#
#        for i, layer in enumerate(self.fc_layers):
#            z, q = layer(z, q)
#
#        out = z, q
#        return out
#
#
#    def expectation(self, z, q, y):
#        batch_size, size = z.shape
#        inps = torch.stack([z for _ in range(size + 1)], dim=0) # inps.shape = O + 1 x B x O
#        inps = torch.einsum('abo,ao->abo', (inps, self.mask)).view(-1, size)
#        pred = self.output_layer(inps)
#
#        if self.expect_pred:
#            pred_ = pred.view(size + 1, batch_size, -1) # O x B x D
#            pred_ = pred_[0] + (pred_[1:] - pred_[:1]).mul(q.t()[:, :, None]).sum(dim=0)
#            y = y.view(size + 1, -1)[0]
#            return F.cross_entropy(pred_, y), pred_
#        elif self.expect_probs:
#            pred_ = pred.view(size + 1, batch_size, -1) # O x B x D
#            probs = F.softmax(pred_, dim=2)
#            probs = probs[0] + (probs[1:] - probs[:1]).mul(q.t()[:, :, None]).sum(dim=0)
#            y = y.view(size + 1, -1)[0]
#            return F.nll_loss(probs.add(1e-6).log(), y), probs
#        else:
#            pred_ = pred.view(size + 1, batch_size, -1)[0]
#            f = F.cross_entropy(pred, y, reduction='none').view(size + 1, batch_size)
#            out = f[0].mul(1 - q.sum(dim=1, keepdim=True)) + torch.einsum('ob,bo->b', (f[1:], q))
#            #print(F.cross_entropy(pred_, y.view(size + 1, -1)[0]).item(), f[0].mean().item(), out.mean().item())
#            return out.mean(), pred_


class StochasticBinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, t=0.):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1)
        self.t = t
        self._t = torch.tensor([t])
        self._reset_parameters()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._init = False

    
    def _reset_parameters(self):
        gain = torch.nn.init.calculate_gain('sigmoid')
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=gain)


    def _compute_conv_diffs(self, x, a0):
        k = self.kernel_size
        pad = 2 * (k // 2) - self.padding

        batch_size, out_channels, out_h, out_w = a0.shape
        _, in_channels, in_h, in_w = x.shape

        K_flip = torch.flip(self.conv.weight, [2, 3])

        a = torch.zeros([k, k, batch_size, out_channels, in_channels, in_h, in_w]).to(a0)
        a0_ = torch.zeros([batch_size, out_channels, in_h + k - 1, in_w + k - 1]).to(a0)
        a0_[:, :, pad:-pad, pad:-pad] = a0
        
        for l1 in range(k):
            for l2 in range(k):
                val = a0_[:, :, None, l1:l1 + in_h, l2:l2 + in_w] - 2 * x[:, None] * K_flip[None, :, :, l1, l2, None, None]
                a[l1, l2] = val
        
        return a


    def _compute_delta_sum(self, new_x, p, p0, q):
        k = self.kernel_size
        pad = 2 * (k // 2) - self.padding

        batch_size, out_channels, out_h, out_w = p0.shape
        _, in_channels, in_h, in_w = q.shape

        delta_q = torch.zeros([batch_size, out_channels, in_h + k - 1, in_w + k - 1]).to(p0)

        p0_ = torch.zeros([batch_size, out_channels, in_h + k - 1, in_w + k - 1]).to(p0)
        p0_[:, :, pad:-pad, pad:-pad] = p0
        #self.p0_[:, :, pad:-pad, pad:-pad] = p0

        y_ = torch.zeros([batch_size, out_channels, in_h + k - 1, in_w + k - 1]).to(p0)
        y_[:, :, pad:-pad, pad:-pad] = new_x
        #self.y_[:, :, pad:-pad, pad:-pad] = new_x

        for l1 in range(k):
            for l2 in range(k):
                add = p[l1, l2].sub(p0_[:, :, None, l1:l1 + in_h, l2:l2 + in_w]).mul(q[:, None]).sum(dim=2).mul(-y_[:, :, l1:l1 + in_h, l2:l2 + in_w])
                delta_q[:, :, l1:l1 + in_h, l2:l2 + in_w] += add

        return delta_q[:, :, pad:-pad, pad:-pad]


    def _norm_forward(self, x, q):
        a0 = self.conv(x)
        p0 = torch.sigmoid(a0)
        a = self._compute_conv_diffs(x, a0)
        p = torch.sigmoid(a)

        new_x = 2 * p0.bernoulli() - 1

        delta_0 = -new_x * (p0 - p0.detach())
        delta_q = self._compute_delta_sum(new_x, p, p0, q)

        qs = q.sum(dim=3, keepdim=True)
        qs = qs.sum(dim=2, keepdim=True)
        qs = qs.sum(dim=1, keepdim=True)

        new_q = delta_0.mul(1 - qs) + delta_q
        return new_x, new_q

    
    def forward(self, x, q):
        new_x, new_q = self._norm_forward(x, q)
        new_x = new_x[:, :, ::self.stride, ::self.stride].contiguous()
        new_q = new_q[:, :, ::self.stride, ::self.stride].contiguous()
        return new_x, new_q


class StochasticBinaryLinearSoftmax(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        mask_size = in_features
        mask = [torch.ones(mask_size)]
        for i in range(mask_size):
            m = torch.ones(mask_size)
            m[i] = -1
            mask.append(m)
        self.mask = torch.stack(mask, dim=0)

    def forward(self, x, q):
        if x.device != self.mask.device:
            self.mask = self.mask.to(x.device)

        batch_size, size = x.shape
        inps = torch.stack([x for _ in range(size + 1)], dim=0) # inps.shape = O + 1 x B x O
        inps = torch.einsum('abo,ao->abo', (inps, self.mask)).view(-1, size)
        pred = super().forward(inps)

        pred_ = pred.view(size + 1, batch_size, -1) # O x B x D
        probs = F.softmax(pred_, dim=2)
        probs = probs[0] + (probs[1:] - probs[:1]).mul(q.t()[:, :, None]).sum(dim=0)
        return probs.add(1e-6).log()

class StochasticBinaryArchMixin:
    def _get_linear(self, *args):
        return StochasticBinaryLinear_2(*args)

    def _get_conv(self, *args, **kwargs):
        return StochasticBinaryConv2d(*args, **kwargs)

    def _get_first_conv(self, *args, **kwargs):
        return StochasticBinaryConv2d(*args, **kwargs)

    def _get_output_layer(self, *args, **kwargs):
        return StochasticBinaryLinearSoftmax(*args, **kwargs)

    def _get_input(self, x):
        q = torch.zeros_like(x).to(x)
        return x, q

    def _apply_layer(self, layer, *args):
        z, q = args[:2]
        return layer(z, q)
    

#class StochasticBinaryLeNet5(nn.Module):
#    def __init__(self, args):
#        super().__init__()
#        self.conv_layers = nn.ModuleList([
#            StochasticBinaryConv2d(1, 6, kernel_size=5, padding=2, stride=2),
#            StochasticBinaryConv2d(6, 16, kernel_size=5, padding=0, stride=2)
#        ])
#
#        self.fc_layers = nn.ModuleList([
#            StochasticBinaryLinear_2(400, 120),
#            StochasticBinaryLinear_2(120, 84),
#        ])
#        
#        self.output_layer = nn.Linear(84, 10)
#
#
#    def forward(self, x):
#        z = x
#        q = torch.zeros_like(z).to(z)
#
#        for layer in self.conv_layers:
#            z, q = layer(z, q)
#
#        z = z.view(z.shape[0], -1)
#        q = q.view(q.shape[0], -1)
#        
#        for layer in self.fc_layers:
#            z, q = layer(z, q)
#
#        out = z, q
#        return out
        
