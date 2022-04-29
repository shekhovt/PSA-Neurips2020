from models.utils import *
import numpy as np
import torch.nn.functional as F
import copy
from gradeval.utils import *
from .options import odict

class ItTracker(odict):
    def __init__(self, max_idx=None):
        super().__init__()
        if max_idx is not None:
            # track the value from the last iteration
            self.max_idx = max_idx
            self.last = np.zeros(self.max_idx)
            self.last_v = np.zeros(self.max_idx)
            self.n = np.zeros(self.max_idx)
            self.sum = 0
            self.sum_v = 0
    
    def update(self, data, idx):
        old_d = self.last[idx]
        self.sum -= np.sum(old_d)
        self.last[idx] = data
        self.sum += np.sum(data)
        #
        old_v = self.last_v[idx]
        self.sum_v -= np.sum(old_v)
        mask = self.n[idx] > 0
        new_v = (data - old_d) ** 2 / 2  # cheap two point variance estimate
        new_v[~mask] = 0
        self.last_v[idx] = new_v
        self.sum_v += new_v.sum()
        #
        self.n[idx] += 1
    
    def mean(self):
        n_valid = np.sum(self.n > 0)
        return self.sum / n_valid
    
    def data_var_of_mean(self):
        n_valid = np.sum(self.n > 1)
        if n_valid == 0:
            return 0
        x = self.last[self.n > 0]
        return x.var() / n_valid
    
    def data_var(self):
        n_valid = np.sum(self.n > 1)
        if n_valid == 0:
            return 0
        x = self.last[self.n > 0]
        return x.var()
    
    def var_of_mean(self):
        n_valid = np.sum(self.n > 1)
        if n_valid == 0:
            return 0
        return self.sum_v / (n_valid ** 2)


class SmoothItTracker(ItTracker):
    def __init__(self, max_idx):
        super().__init__(max_idx)
        self.stat = [RunningStat(pow(0.1, 1 / 10)) for i in range(max_idx)]
    
    def update(self, data, idx):
        super().update(data, idx)
        for d, i in zip(data.tolist(), idx):
            self.stat[i].update(d)
    
    def mean_smooth(self):
        return np.array([self.stat[i].mean for i in range(self.max_idx) if self.stat[i].n > 0]).mean()
    
    def var_smooth(self):
        v = np.array([self.stat[i].var for i in range(self.max_idx) if self.stat[i].n > 1])
        return v.mean() / len(v)


class TrainHist(odict):
    def __init__(self):
        super().__init__()
        pass
    
    def record(self, epoch, b_loss, stats, n_data, batch_size):
        # losses:
        record = odict()
        #
        record.epoch = epoch
        record.b_loss = b_loss
        record.r_loss = stats.r_loss.mean()  # running loss optimized
        record.u_loss = stats.u_loss.mean()  # unbiased loss target
        #
        # variances related to losses:
        record.loss_var_params_samples = stats.r_loss.var_of_mean()  # variance due to parameters change and samples
        record.loss_var_batches = stats.r_loss.data_var() / batch_size  # variance due to random batches
        record.loss_var_samples = stats.u_loss_v.mean() / n_data  # variance due to sample
        record.loss_var_params = stats.u_loss.var_of_mean()  # variance due to parameter change (and residual over samples)
        #
        # accuracies
        record.acc1 = stats.u_acc.mean()  # 1-sample accuracy
        record.acc_e = stats.u_acc_e.mean()  # ensemble accuracy
        # accuracy variances
        record.acc1_var_samples = nan_to_zero(stats.u_acc_v.mean() / n_data)  # var due to samples
        record.acc1_var_params = stats.u_acc.var_of_mean()  # var due to params
        
        # entropy
        record.entropy = stats.entropy.mean()  # var due to params
        #
        record.lock()
        # append the record to arrays
        self.unlock()
        for (k, v) in record.items():
            if k not in self:
                self[k] = np.array([v])
            self[k] = np.append(self[k], v)
        self.lock()
        return record
    
    
    @staticmethod
    def str_record(rec):
        ci_r_loss = rec.loss_var_params_samples ** 0.5
        ci_u_loss = (rec.loss_var_samples + rec.loss_var_params) ** 0.5
        s = 'Losses: '
        s += 'stochastic: {:.4f}'.format(rec.b_loss)
        s = s + ' running: ' + format_std(rec.r_loss, ci_r_loss)
        s = s + ' unbiased: ' + format_std(rec.u_loss, ci_u_loss)
        s = s + ' batch_std: ' + format_std(rec.loss_var_batches ** 0.5)
        #
        acc1_ci = (rec.acc1_var_samples + rec.acc1_var_params) ** 0.5
        s = s + ' Acc: ' + format_std(rec.acc1 * 100, acc1_ci * 100, units='%')
        s = s + ' Ensemble acc: ' + format_std(rec.acc_e * 100, units='%')
        s = s + ' Entropy: ' + format_std(rec.entropy)
        return s


class TrainStats(object):
    def __init__(self, max_idx):
        super().__init__()
        self.max_idx = max_idx
        self.r_loss = ItTracker(self.max_idx)
        self.r_loss = ItTracker(self.max_idx)
        self.u_loss = ItTracker(self.max_idx)
        self.u_loss_v = ItTracker(self.max_idx)
        self.u_acc = ItTracker(self.max_idx)
        self.u_acc_v = ItTracker(self.max_idx)
        self.u_acc_e = ItTracker(self.max_idx)
        self.entropy = ItTracker(self.max_idx)
        self.hist = TrainHist()
        # self.lock()
    
    def hist_record(self, epoch, b_loss, n_data, batch_size):
        return self.hist.record(epoch, b_loss, self, n_data, batch_size)
    
    @staticmethod
    def str_record(rec):
        return TrainHist.str_record(rec)


class DictHist(odict):
    def __init__(self):
        super().__init__()
        self.legend = odict()
    
    def record(self, epoch, rec: dict):
        rec['epoch'] = epoch
        self.unlock()
        for (k, v) in rec.items():
            if k not in self:
                self[k] = np.array([v])
            self[k] = np.append(self[k], v)
        self.lock()

    def set_legend(self, key, val):
        self.legend[key] = val


class Stats:
    def __init__(self, o, train_loader, val_loader):
        super().__init__()
        max_idx = len(train_loader.sampler) + len(val_loader.sampler)
        self.train = TrainStats(max_idx)
        self.val = TrainStats(max_idx)
