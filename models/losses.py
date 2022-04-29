import torch
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod, abstractproperty
from experiments.options import odict

"""________________________________________________________________________________________________________
Loss estimators available with methods computing / approximating gradient of an expectation
"""


def obj_preactivation(a):
    return a


def obj_accuracy(a, target):
    pred = a.argmax(dim=-1, keepdim=True)
    acc = (pred == target.view([-1] + [1] * (pred.dim() - 1)))
    return acc.to(dtype=a.dtype)


def obj_nll(a, target_idx):
    """
    a activations [N C]  or [N S C]
    target_idx [N]
    output: [N 1] resp. [N S 1]
    """
    pred_logp = F.log_softmax(a, dim=-1)
    sz = list(pred_logp.size())
    C = sz[-1]  # classes
    osz = sz[:-1]
    target_idx = target_idx.view([-1] + [1] * (len(osz) - 1)).expand(osz).contiguous().view([-1])
    pred_logp = pred_logp.view([-1, C])
    L = F.nll_loss(pred_logp, target_idx, reduction='none').view([*osz, 1])
    return L


def obj_hinge(a, target_idx):
    """
    a activations [N C]  or [N S C]
    target_idx [N]
    output: [N 1] resp. [N S 1]
    """
    K = a.size(-1)
    t = F.one_hot(target_idx, num_classes=K).to(dtype=a.dtype) * 2 - 1
    if a.dim() == 3:
        t = t.unsqueeze(dim=1)  # per sample
    return (1 - t * a).clamp(min=0)


def obj_log_softmax(a):
    return F.log_softmax(a, dim=-1)


def obj_softmax(a):
    return F.softmax(a, dim=-1)


def obj_argmax(a):
    K = a.size(-1)
    pred = a.argmax(dim=-1, keepdim=False)
    amax = F.one_hot(pred, num_classes=K).to(dtype=a.dtype)
    return amax


def samples_to_batch(data, n_samples):
    # copy on dimension 0
    rep = [n_samples] + [1] * (data.dim() - 1)
    return data.repeat(rep)


def batch_to_samples(data, n_samples):
    # reshape on dimension 0
    sz = [n_samples, data.size(0) // n_samples] + list(data.size()[1:])
    return data.view(sz)


class ExpectedLoss:
    def __init__(self):
        self.loss_obj = obj_log_softmax
    
    @abstractmethod
    def linear_loss(self, f, v_f, data, target_idx, *targets):
        """
            f [S B K]
            v_f [S B K]
            data [B ...]
            target_idx [B]
            *targets [B ...] extra data from the loader
            return: sloss, sv_loss [S1 B]
        """
        pass

    def make_loss_obj(self, target_idx, *targets):
        """
        usual losses with be linear in the targets and self.obj
        """
        return self.loss_obj
    
    def train_loss(self, net, method, data, target_idx, *targets, n_samples=1):
        sdata = samples_to_batch(data, n_samples)
        f = net.forward(sdata, method=method, objs=self.make_loss_obj(target_idx, *targets))
        f = batch_to_samples(f, n_samples)
        (sloss, sv_loss) = self.linear_loss(f, None, data, target_idx, *targets)
        # mean loss
        loss = sloss.mean(dim=0, keepdim=False)
        return loss
    
    def eval_metrics(self, net, method, data, target_idx, *targets, n_samples=10):
        assert (n_samples >= 1)
        net.train(False)  # test mode on
        sdata = samples_to_batch(data, n_samples)
        starget_idx = samples_to_batch(target_idx, n_samples)
        #
        obj_acc = lambda a: obj_accuracy(a, starget_idx)
        #
        res = net.forward(sdata, method=method, objs=(self.make_loss_obj(starget_idx, *targets), obj_softmax, obj_acc), compute_variances=True)
        res = list([E, V] for (E, V) in res)
        #
        for r in res:
            r[0] = batch_to_samples(r[0], n_samples)
            r[1] = batch_to_samples(r[1], n_samples)
        
        (f, v_f), (p, v_p), (acc1, v_acc1) = res
        # loss
        (sloss, sv_loss) = self.linear_loss(f, v_f, data, target_idx, *targets)
        # mean loss
        loss = sloss.mean(dim=0, keepdim=False)
        v_loss = (sloss.detach().var(dim=0, keepdim=False) + sv_loss.mean(dim=0)) / n_samples
        #
        acc1 = acc1.squeeze(dim=2)
        v_acc1 = v_acc1.squeeze(dim=2)
        v_acc1 = (acc1.var(dim=0) + v_acc1.mean(dim=0)) / n_samples
        acc1 = acc1.mean(dim=0)
        #
        pm = p.mean(dim=0, keepdim=False)
        pred = pm.argmax(dim=1)
        e_acc = (pred == target_idx).to(dtype=pm.dtype)
        #
        #
        return odict(loss=loss, v_loss=v_loss, p=p, v_p=v_p, acc1=acc1, v_acc1=v_acc1, e_acc=e_acc)


class ExpectedLoss_NLL(ExpectedLoss):
    #
    def linear_loss(self, a, v_a, data, target_idx, *targets):
        """
            a [S B K]
            v_a [S B K] / None
            data [B ...]
            target_idx [B]
            *target_idx [B ...]
            return: sloss, sv_loss [S B] / None
        """
        n_samples = a.size(0)
        a = a.flatten(start_dim=0, end_dim=1)
        starget_idx = samples_to_batch(target_idx, n_samples)
        sloss = F.nll_loss(a, starget_idx, reduction='none')  # select a[sample, batch, starget_idx[batch]]
        sloss = batch_to_samples(sloss, n_samples)
        if v_a is not None:
            v_a = v_a.flatten(start_dim=0, end_dim=1)
            sv_loss = F.nll_loss(v_a, starget_idx, reduction='none')
            sv_loss = batch_to_samples(sv_loss, n_samples)
        else:
            sv_loss = None
        #
        return (sloss, sv_loss)


class ExpectedLoss_CE(ExpectedLoss):
    def linear_loss(self, a, v_a, data, target_idx, target_logp, *targets):
        """
        Compute: CE = - sum_i target_p_i * pred_log_p_i
        For exact gt labels (when target_p is binary), equals NLL
            a [S B K]
            v_a [S B K]
            data [B ...]
            target_idx [B]
            target_logp [B]
            return: sloss, sv_loss [S B]
        """
        target_p = target_logp.exp()
        sloss = -(a * target_p.unsqueeze(dim=0)).sum(dim=-1, keepdim=False)
        if v_a is not None:
            sv_loss = (v_a * (target_p ** 2).unsqueeze(dim=0)).sum(dim=-1, keepdim=False)
        else:
            sv_loss = None
            #
        return (sloss, sv_loss)


def loss_mse(a, v_a, target):
    """
        a [S B K]
        v_a [S B K]
        data [B ...]
        target_idx [B]
        target [B]
        return: sloss, sv_loss [1 B]
    """
    # target probabilities
    assert (a.dim() == 3 and a.size(0) >= 2)
    n_samples = a.size(0)
    target = target.unsqueeze(dim=0)
    nn = (n_samples * (n_samples - 1))
    # average over all independent pairs of samples: (1/nn sum_{i!=j} a_i * a_j = 1/nn ((sum_{i} a_i)**2 - sum_i a_i**2))
    E2 = (a.sum(dim=0, keepdim=True) ** 2 - (a ** 2).sum(dim=0, keepdim=True)) / nn  # unbiased estimate of (E[f])**2
    Eg = a.mean(dim=0, keepdim=True) * target
    sloss = E2 - 2 * Eg + target ** 2  # This is loss estimate per class
    sloss = sloss.mean(dim=-1, keepdim=False)  # mean over all classes
    #
    # variance of the (E[f])**2 estimate
    if v_a is not None:
        sv_loss = (v_a.sum(dim=0, keepdim=True) * (a ** 2).sum(dim=0, keepdim=True) - (v_a * a ** 2).sum(dim=0, keepdim=True)) / nn  # variance per class
        sv_loss = sv_loss.sum(dim=-1, keepdim=False)  # this is for the total
        # Eg estimate is dependent, do not include its variance, should be dominated by above
    else:
        sv_loss = None
        #
    return (sloss, sv_loss)


class ExpectedLoss_MSE_logp(ExpectedLoss):
    def __init__(self):
        super().__init__()
        self.loss_obj = obj_log_softmax
    
    def linear_loss(self, a, v_a, data, target_idx, target_logp):
        return loss_mse(a, v_a, target_logp)


class ExpectedLoss_MSE_p(ExpectedLoss):
    def __init__(self):
        super().__init__()
        self.loss_obj = obj_softmax
    
    def linear_loss(self, a, v_a, data, target_idx, target_logp):
        return loss_mse(a, v_a, target_logp.exp())


class ExpectedLoss_Acc(ExpectedLoss_NLL):
    """ We estimate argmax indicator as predictive probability, then CE with this probability is the -Accuracy"""
    
    def __init__(self):
        super().__init__()
        self.loss_obj = obj_argmax

    def linear_loss(self, a, v_a, data, target_idx, target_logp):
        # actually return the error rate, use suepr to compute -a[target_idx] (-1 for correct pred and 0 for incorrect), invert it return the error rate
        (sloss, sv_loss) = super().linear_loss(a, v_a, data, target_idx, target_logp)
        sloss = 1 + sloss
        return (sloss, sv_loss)


class ExpectedLoss_Hinge(ExpectedLoss):
    """ We estimate argmax indicator as predictive probability, then CE with this probability is the -Accuracy"""
    
    def __init__(self):
        super().__init__()
        self.loss_obj = obj_argmax
    
    def make_loss_obj(self, target_idx, *targets):
        return lambda a: obj_hinge(a, target_idx)

    def linear_loss(self, f, v_f, data, target_idx, *targets):
        f = f.mean(dim=-1)
        if v_f is not None:
            v_f = v_f.mean(dim=-1)
        return f, v_f


class ExpectedLoss_HingeSQ(ExpectedLoss_Hinge):
    """ We estimate argmax indicator as predictive probability, then CE with this probability is the -Accuracy"""
    
    def __init__(self):
        super().__init__()
        self.loss_obj = obj_argmax
    
    def make_loss_obj(self, target_idx, *targets):
        return lambda a: obj_hinge(a, target_idx) ** 2
