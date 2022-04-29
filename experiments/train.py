import context
import models
from experiments.data_loader import *
from experiments.construct import *
import numpy as np
import scipy.optimize
import math
import torch.nn.functional as F
import copy
# from models.architectures import *
# from models.methods import *
# from models.losses import *
from gradeval.utils import *
from models.utils import *
from experiments.save_load import *
from experiments.plot_training import draw_all
from experiments.config import save_config, uncommon_important
from experiments.train_hist import *

import time

np.seterr(divide='ignore', invalid='ignore')


class NoneException(BaseException):
    pass


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Train:
    def __init__(self, o):
        self.o = o.clone()
        self.data = new_dataset(self.o)
        self.o.create.input_shape = self.data.input_shape
        self.o.create.input_size = self.data.input_size
        self.o.create.n_classes = self.data.n_classes
        self.expected_loss = new_loss(self.o)
        
        self.preproc = new_preproc(self.o)
    
    def upgrade(self):  # backward compatibility
        if 'eval_method' not in self.__dict__:
            self.eval_method = models.method_class("sample")()
    
    def log(self, *args):
        o = self.o.file
        force_path(o.log_dir)
        if o.verbose:
            print(*args)
            with open(o.log_dir + 'output.txt', 'a') as f:
                print(*args, file=f)
    
    def new_log(self):
        o = self.o.file
        force_path(o.log_dir)
        with open(o.log_dir + 'output.txt', 'w') as f:
            print('Starting anew', file=f)
    
    def get_state(self):
        s = odict()
        s.stats = self.stats
        s.lock()
        return s
    
    def new_net(self):
        torch.manual_seed(self.o.train.init_seed)
        self.net = new_model(self.o)
        self.net_init()
        self.eval_method = models.method_class("sample")()
        if self.o.train.norm == 'AP2':
            reparam_with_norm(self.net)
            self.eval_method = MethodWithNorm(method=self.eval_method)
    
    def new_loaders(self):
        self.train_loader, self.val_loader = get_train_valid_loader(self.data.train_dataset, train_batch_size=self.o.train.batch_size,
                                                                    val_batch_size=self.o.test.batch_size,
                                                                    random_seed=self.o.train.data_seed,
                                                                    valid_size=self.o.train.val_size, shuffle=True, num_workers=0,
                                                                    pin_memory=False, train_transform=self.data.train_transform,
                                                                    val_transform=self.data.test_transform)
        self.test_loader = torch.utils.data.DataLoader(self.data.test_dataset, batch_size=self.o.test.batch_size, shuffle=True, num_workers=0,
                                                       pin_memory=False)
        self.n_train_data = len(self.train_loader.sampler)
        # self.val_len = len(self.train_loader)
    
    def new_state(self):
        self.stats = Stats(self.o, self.train_loader, self.val_loader)
        phist = DictHist()
        self.hist = odict(train=self.stats.train.hist, val=self.stats.val.hist, precond=phist)
        self.epoch = 0

    def new_optimizer(self, lr):
        sgdargs = dict(lr=lr, weight_decay=0, momentum=self.o.train.momentum, nesterov=self.o.train.Nesterov)
        if self.o.train.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), **sgdargs)
        elif self.o.train.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), amsgrad=False)
        elif self.o.train.optimizer == 'Amsgrad':
            self.optimizer = torch.optim.Adam(self.net.parameters(), amsgrad=True)
    
    def trial(self, lr):
        self.log('Evaluating lr: {}'.format(lr))
        self.new_net()
        self.new_state()
        self.new_optimizer(lr)
        self.set_lr(lr)
        self.o.file.verbose = False
        try:
            for epoch in range(self.o.train.lr_estimate_epochs):
                self.train_epoch(epoch)
                #
                l = self.stats.train.r_loss.mean()
                if l > 1e10:
                    return float('inf')
        except NumericalProblem as e:
            return float('inf')
        self.o.file.verbose = True
        self.log('Loss estimate: {}'.format(l))
        return l
    
    def choose_lr(self):
        self.collect_metrics = False
        f = lambda x: self.trial(lr=pow(10, x))
        llr = scipy.optimize.minimize_scalar(f, method='bounded', bounds=(-6, 0), options=dict(maxiter=self.o.train.lr_estimate_trials, disp=True)).x
        best_lr = pow(10.0, llr)
        self.log('Selected lr: {}'.format(best_lr))
        return best_lr
    
    def trial_saved(self, lr):
        self.load_state(self.o.file.log_dir + 'training.pkl')
        # self.new_optimizer(lr) # discard momentum, etc. But it is post-multiplied with lr, should be fine
        self.set_lr(lr)
        self.log(f' Evaluating lr:{lr}')
        self.o.file.verbose = False
        self.collect_metrics = False
        try:
            for epoch in range(self.o.train.lr_estimate_epochs):
                self.train_epoch(epoch)
                l = self.stats.train.r_loss.mean()
                if l > 1e10:
                    return float('inf')
        except NumericalProblem as e:
            return float('inf')
        self.o.file.verbose = True
        self.log('Loss estimate: {}'.format(l))
        return l

    def choose_lr_saved(self):
        self.log(f'Reoptimizing lr step:')
        self.log(f' Current lr:{self.lr}')
        lr0 = self.lr
        f = lambda x: self.trial_saved(lr=pow(10, x))
        llr = scipy.optimize.minimize_scalar(f, method='bounded', bounds=(math.log10(lr0) - 1, math.log10(lr0) + 1),
                                             options=dict(maxiter=self.o.train.lr_estimate_trials, disp=True)).x
        self.load_state(self.o.file.log_dir + 'training.pkl')
        self.hist.precond.record(self.epoch, dict(step_lr=llr))
        # self.hist.precond.set_legend('step_lr', 'step_lr')
        self.set_lr(pow(10.0, llr))
        self.log('Selected lr: {}'.format(self.lr))

        
    def train_from_scratch(self):
        #
        self.new_log()
        save_config(self.o, 'o')
        #
        self.new_loaders()
        self.new_net()
        self.benchmark_net()
        if self.o.train.lr is None:
            lr = self.choose_lr()
            self.o.train.lr = lr # initial lr
        # save the initial learning rate, need for plotting
        save_config(self.o, 'o1')
        #
        self.new_net()
        self.new_state()
        self.new_optimizer(self.o.train.lr)
        self.set_lr(self.o.train.lr)
        #
        if self.o.train.backtrack:
            self.save_epoch()
        #
        self.continue_training()
    
    def continue_training(self):
        while self.epoch <= self.o.train.epochs:
            e_old = self.epoch
            if self.o.train.backtrack:
                self.train_backtrack(self.o.train.backtrack_epochs)
            else:
                if self.o.train.step_lr and self.epoch > 0 and self.epoch % self.o.train.step_lr == 0:  # time to reoptimize lr
                    self.choose_lr_saved()
                # train as normal
                self.train_plain()
                self.epoch += 1
            self.save_epoch()
            interval = min(50, max((self.o.train.epochs // 10), 1))
            if self.epoch // interval != e_old // interval or self.epoch == self.o.train.epochs:
                try:
                    self.drawings()
                except BaseException as e:
                    self.log('drawing failed', e)
            # self.scheduler.step()
    
    def set_lr(self, lr):
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = lr
        self.lr = lr
    
    def train_plain(self):
        self.collect_metrics = True
        self.log(self.o.file.log_dir)
        self.train_epoch(self.epoch)
        self.val_epoch(self.epoch)
    
    def train_backtrack(self, backtrac_epochs):
        # self.save_state(self.o.file.log_dir + 'backtrack.pkl')
        self.log(f' Current lr:{self.lr}')
        vl = []
        lr_factors = [1.0, 0.9, 1.1]
        for (t, lr_factor) in enumerate(lr_factors):
            self.load_state(self.o.file.log_dir + 'training.pkl')
            self.set_lr(self.lr * lr_factor)
            self.log(f' Trial lr factor:{lr_factor}')
            #
            for e in range(backtrac_epochs):
                torch.manual_seed(self.o.train.init_seed * 12345 + self.epoch)  # reproducible epoch
                self.collect_metrics = True
                self.log(self.o.file.log_dir)
                self.train_epoch(self.epoch)
                self.val_epoch(self.epoch)
                self.epoch += 1
            self.save_state(self.o.file.log_dir + f'backtrack-{t}.pkl')
            vl += [self.hist.val.r_loss[-1]]
        best_t = np.argmin(np.array(vl))
        self.load_state(self.o.file.log_dir + f'backtrack-{best_t}.pkl')
        self.log(f' Best lr factor:{lr_factors[best_t]}')
    
    def train_epoch(self, epoch):
        torch.manual_seed(self.o.train.init_seed * 12345 + epoch)  # reproducible epoch
        self.run_epoch(epoch, self.train_loader, self.stats.train, self.o.train, training=True)
    
    def val_epoch(self, epoch):
        self.log('____________________________________________________________________________________')
        self.run_epoch(epoch, self.val_loader, self.stats.val, self.o.test, training=False)
        if self.o.train.precond:
            (n, v) = self.optimizer.report()
            self.hist.precond.record(epoch, dict(lr=v))
            self.hist.precond.set_legend('lr', n)
        self.log('____________________________________________________________________________________')
    
    # def train_state_dict(self):
    #     d = odict()
    #     d.optimizer = self.optimizer.state_dict()
    #     d.o = self.o
    #     d.epoch = self.epoch
    #     d.
    
    def del_not_state(self):
        self.train_loader = None
        self.val_loader = None
        self.data = None
    
    def restore_state(self):
        if self.data is None:
            self.data = new_dataset(self.o)
        self.new_loaders()
    
    def save_state(self, file_name):
        cpy = copy.copy(self)  # shallow copy
        cpy.del_not_state()
        save_object(file_name, cpy)
    
    def load_state(self, file_name):
        cpy = pickle.load(open(file_name, "rb"))
        d = self.__dict__
        self.__dict__ = cpy.__dict__
        self.data = d['data']
        self.restore_state()
    
    def save_epoch(self):
        self.o.train.epoch = self.epoch
        save_config(self.o, 'o1')
        save_object(self.o.file.log_dir + 'hist.pkl', self.hist)
        save_object(self.o.file.log_dir + 'model_state.pkl', self.net.state_dict())
        self.net.zero_grad()
        self.net.eval()
        self.save_state(self.o.file.log_dir + 'training.pkl')
    
    def run_epoch(self, epoch, loader, stats, o, training):
        if self.o.train.stochastic:
            self.run_epoch_stochastic(epoch, loader, stats, o, training)
        else:
            self.run_epoch_determenistic(epoch, loader, stats, o, training)
    
    def new_iter(self, loader):
        device = next(self.net.parameters()).device
        if loader == self.train_loader or not isinstance(self.preproc, PreprocTransformData):
            iter = PreprocToDevice(device)(self.preproc(loader))
        else:
            iter = PreprocToDevice(device)(loader)
        return iter
    
    def net_init(self):
        iter = self.new_iter(self.train_loader)
        (data_idx, data, *targets) = next(iter)
        method = InitMethod(odict(activation='tanh'))
        with torch.no_grad():
            self.net.forward(data, method=method)
    
    def benchmark_net(self):
        # benchmark the net
        iter = self.new_iter(self.train_loader)
        (data_idx, data, *targets) = next(iter)
        # with torch.no_grad():
        #     self.net.forward(data)
        # print('Nograd mode:')
        # for (i, l) in enumerate(self.net.layers):
        #     n = l.__class__.__name__
        #     t = l.time if hasattr(l, 'time') else 0
        #     print(f'({i})  {n} fw time: ', format_digits(t, 2))

        print('Time benchmakr:')
        self.net.train()
        T = 0
        logp = self.net.forward(data, sync=True)
        for (i, l) in enumerate(self.net.layers):
            n = l.__class__.__name__
            t = l.time if hasattr(l, 'time') else 0
            print(f'({i})  {n} fw time: ', format_digits(t, 2), 'shape:', l.out_shape)
            T += t
        print(f'Total forward: ', format_digits(T, 2))
        # BW
        torch.cuda.synchronize()
        start_time = time.time()
        logp.mean().backward()
        torch.cuda.synchronize()
        t = time.time() - start_time
        #
        print(f'Total backward: ', format_digits(t, 2))
    
    def clip_gradients(self):
        for (n, p) in self.net.named_parameters():
            g = p.grad
            if g is None:
                pass
                # print("{} : Grad is None (Its's Ok if the parameter is unused.)".format(n))
            else:  # clipping
                if torch.isnan(g.data).any().item():
                    print("{} : Grad has NaN".format(n))
                    self.check_param(n, p)
                g.data = g.data.clamp(-1e3, 1e3)
    
    def check_param(self, n, p):
        if torch.isnan(p.data).any().item():
            raise NumericalProblem("{} : Parameter has NaN".format(n))
        if torch.isinf(p.data).any():
            raise NumericalProblem('+-Inf encountered in {}'.format(n))
    
    def check_params(self):
        for (n, p) in self.net.named_parameters():
            self.check_param(n, p)
    
    def run_epoch_stochastic(self, epoch, loader, stats, o, training):
        #
        # self.log('Learning rate: {}'.format(lr_factor * self.args.lr))
        # #  set optimizer lr value
        # for group in self.optimizer.param_groups:
        #     group['lr'] = lr_factor * self.args.lr
        # #
        n_data = len(loader.sampler)
        batch_size = o.batch_size
        n_batches = divup(n_data, batch_size)
        if training:
            log_interval = max(n_batches // 10, 1)
        else:
            log_interval = n_batches
        #
        iter = enumerate(self.new_iter(loader))
        while True:
            try:
                batch_idx, (data_idx, data, *targets) = next(iter)
            except StopIteration:
                # if StopIteration is raised, break from loop
                break
            if self.collect_metrics:
                with torch.no_grad():
                    self.net.train(False)  # test mode on
                    # take an unbiased value estimator
                    
                    m = self.expected_loss.eval_metrics(self.net, self.eval_method, data, *targets, n_samples=self.o.train.stat_samples)
                    """odict(loss=loss, v_loss=v_loss, p=p, v_p=v_p, acc1=acc1, v_acc1=v_acc1, e_acc=e_acc) """
                    for k in m.keys():
                        if m[k].abs().max().item() > 1e10:
                            raise NumericalProblem()
                        m[k] = m[k].cpu().numpy()
                    #
                    stats.u_loss.update(m.loss, data_idx)
                    stats.u_loss_v.update(m.v_loss, data_idx)
                    stats.u_acc.update(m.acc1, data_idx)
                    stats.u_acc_v.update(m.v_acc1, data_idx)
                    stats.u_acc_e.update(m.e_acc, data_idx)
                    
                    if isinstance(self.eval_method, MethodWithNorm):
                        m = self.eval_method.method
                    else:
                        m = self.eval_method
                    H = batch_to_samples(m.entropy, self.o.train.stat_samples).mean(dim=0) / m.bin_units
                    stats.entropy.update(H.cpu().numpy(), data_idx)
            #
            self.net.train()  # training mode on
            if training:
                # renormalize, (stochastically), but no derivatives
                # with torch.no_grad():
                #     self.net.forward(data, method=Renormalize())
                
                def try_forward():
                    self.optimizer.zero_grad()
                    t_loss = self.expected_loss.train_loss(self.net, self.net.method, data, *targets, n_samples=self.o.train.train_samples)
                    # barrier = self.net.forward(data, method=BatchBarrier())
                    # grad_obj = t_loss.mean() + barrier
                    grad_obj = t_loss.mean()
                    grad_obj.backward(retain_graph=False)  # mean over batch
                    t_loss = t_loss.detach()
                    return t_loss
                
                try:
                    t_loss = try_forward()
                    self.clip_gradients()
                except RuntimeError as e:
                    self.log(e)
                    self.log('Retrying, maybe with a different sample will work better...')
                    t_loss = try_forward()
                    self.clip_gradients()
                    #
                    #
                self.optimizer.step()
                #
                self.check_params()
            else:
                with torch.no_grad():
                    t_loss = self.expected_loss.train_loss(self.net, self.net.method, data, *targets, n_samples=self.o.train.train_samples)
            
            # statistics of loss estimates
            l = t_loss.cpu().numpy()
            stats.r_loss.update(l, data_idx)
            
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) * batch_size >= n_data:
                f_epoch = epoch + min((batch_idx + 1) * batch_size / n_data, 1.0)
                rec = stats.hist_record(f_epoch, l.mean(), n_data, batch_size)
                if training:
                    s = 'Train Epoch: {:4d} ({:3.0f}%) '.format(epoch, 100. * (batch_idx + 1) * batch_size / n_data)
                else:
                    s = 'Val Epoch:   {:4d}        '.format(epoch)
                s = s + stats.str_record(rec)
                self.log(s)
    
    def run_epoch_determenistic(self, epoch, loader, stats, o, training):
        n_data = len(loader.sampler)
        batch_size = o.batch_size
        n_batches = divup(n_data, batch_size)
        if training:
            log_interval = max(n_batches // 10, 1)
        else:
            log_interval = n_batches
        #
        device = next(self.net.parameters()).device
        #
        iter = enumerate(self.new_iter(loader))
        while True:
            try:
                batch_idx, (data_idx, data, *targets) = next(iter)
            except StopIteration:
                # if StopIteration is raised, break from loop
                break
            data = data.to(device=device)
            target = ctarget.to(device=device)
            #
            self.net.train()  # training mode on
            if training:
                self.optimizer.zero_grad()
                a = self.net.forward(data)  # log-softmax output for classification
                # t_loss = self.loss(a, target).view([-1])
                t_loss = self.expected_loss.train_loss(self.net, self.net.method, data, target, 1)
                t_loss.mean().backward(retain_graph=False)
                self.optimizer.step()
                t_loss = t_loss.detach()
            else:
                with torch.no_grad():
                    a = self.net.forward(data)
                    t_loss = self.expected_loss.train_loss(self.net, self.net.method, data, target, 1)
                    # t_loss = self.loss(a, target).view([-1])
            
            # statistics of loss estimates
            l = t_loss.cpu().numpy()
            stats.r_loss.update(l, data_idx)
            acc = obj_accuracy(a, target).view([-1]).cpu().numpy()
            stats.u_acc.update(acc, data_idx)
            
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) * batch_size >= n_data:
                f_epoch = epoch + min((batch_idx + 1) * batch_size / n_data, 1.0)
                rec = stats.hist_record(f_epoch, l.mean(), n_data, batch_size)
                if training:
                    s = 'Train Epoch: {:4d} ({:3.0f}%) '.format(epoch, 100. * (batch_idx + 1) * batch_size / n_data)
                else:
                    s = 'Val Epoch:   {:4d}        '.format(epoch)
                s = s + stats.str_record(rec)
                self.log(s)
    
    def drawings(self):
        draw_all(self.o.file.base_dir)


class TrainControl:
    def __init__(self, o):
        self.o = o
        self.err_name = o.file.log_dir + 'error'
        self.lock_name = o.file.log_dir + 'training_running'
        self.train_name = o.file.log_dir + 'training.pkl'
        self.o1_name = o.file.log_dir + 'o1.pkl'
    
    def train_protected(self):
        # check whether we have an up-to-date result or a failed run
        state = None
        o = self.o
        if os.path.exists(self.train_name):
            try:  # try load current state
                # check it is not a ready state
                state = pickle.load(open(self.train_name, "rb"))
            except BaseException as e:
                print('Load failed: ' + str(e), self.train_name)
                state = None
        
        if state is not None:
            print('Load state at epoch: {} '.format(state.epoch), o.file.log_dir)
            # state.__class__ = Train
            state.upgrade()
            # compare to construct options
            o1 = pickle.load(open(o.file.log_dir + 'o.pkl', "rb"))
            # o1 = state.o
            common = odict()
            io1 = uncommon_important(o1, common)
            io = uncommon_important(o, common)
            if io != io1:
                print('Old values:', (io1 - io))
                print('New values:', (io - io1))
                c = query_yes_no('Continue the training using updated parameters (y) or start anew (n) and overwrite?', default="yes")
                if c:
                    print('Answer Yes')
                else:
                    print('Answer No')
                    state = None  # can't use it
        
        if state is not None:
            print('Continuing training', o.file.log_dir)
            # continue, update options
            # state.o = self.o
            state.o.train.epochs = self.o.train.epochs
            state.restore_state()
            state.epoch += 1
            state.continue_training()
        else:
            print('Starting anew ', o.file.log_dir)
            state = Train(self.o)
            state.train_from_scratch()
    
    def training_completed(self):
        try:
            o1 = pickle.load(open(self.o1_name, "rb"))
            return o1.train.epoch >= self.o.train.epochs  # nothing to do
        except:
            return False
    
    def training_running(self):
        out_name = self.o.file.log_dir + 'output.txt'
        if os.path.exists(self.lock_name) and ((os.path.getmtime(self.lock_name) > time.time() - 5) or (
            os.path.exists(out_name) and os.path.getmtime(out_name) > time.time() - 5 * 60)):  # recent output
            # lock exists and is fresh or some activity less than a 5 min ago
            return True
        else:
            return False
    
    def train(self):
        #  lock the dir
        if self.training_completed():
            print('Completed ', self.o.file.log_dir)
            return
        if self.o.debug:
            self.train_protected()
        if self.training_running():
            print('Training may be running or just crashed ', self.o.file.log_dir)
        else:
            # now we lock it
            open(self.lock_name, 'w').close()
            try:
                self.train_protected()
                os.remove(self.lock_name)
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):  # rethrow KeyboardInterrupt
                    raise e
                #
                os.remove(self.lock_name)
                msg = 'ERROR: ' + str(e)
                msg += '\n When training model ' + self.o.file.log_dir
                errf = open(self.err_name, 'w')
                errf.write(msg)
                traceback.print_exc(file=errf)
                errf.close()
                #
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
                frame = last_frame().tb_frame
                ns = dict(frame.f_globals)
                ns.update(frame.f_locals)
                import pdb
                import inspect
                def isdebugging():
                    for frame in inspect.stack():
                        if frame[1].endswith("pydevd.py") or frame[1].endswith("pdb.py"):
                            return True
                    return False
                
                if not isdebugging():
                    pdb.post_mortem()
                else:
                    raise e
            
            # except BaseException as e:
            #     os.remove(self.lock_name)
            #     msg = 'ERROR: ' + str(e)
            #     msg += '\n When training model ' + self.o.file.log_dir
            #     print(msg)
            #     traceback.print_exc(file=sys.stdout)
            #     # create 'error file'
            #     errf = open(self.err_name, 'w')
            #     errf.write(msg)
            #     traceback.print_exc(file=errf)
            #     errf.close()
            #     if isinstance(e, KeyboardInterrupt):  # rethrow KeyboardInterrupt
            #         pass
            #     else:
            #         raise e


if __name__ == "__main__":
    import inspect
    from experiments.config import collect_options
    
    log_dir = '../runs/CIFAR/fc-2-200-Disitl-MSE-p/train.method=AP2/'
    scriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
    
    if len(sys.argv) > 1:  # have 1 command line arguments
        log_dir = sys.argv[1] + '/'
        print(log_dir)
    
    if len(sys.argv) > 2:  # have 2 command line arguments
        epochs = int(sys.argv[2])
        print(f'Continue to #epochs={epochs}')
    else:
        epochs = 0
    
    if len(sys.argv) > 3:  # have 3 command line arguments
        debug = bool(sys.argv[3])
        print('Debug mode')
    else:
        debug = False
    
    oo = []
    o_file = log_dir + 'o.pkl'
    if os.path.exists(o_file):
        o = pickle.load(open(o_file, "rb"))
        # update location in config
        o.file.log_dir = log_dir
        oo += [o]
        print(f'Loaded config from {o_file}')
    else:
        print(f'No config found: {o_file}')
        print(f'Looking in subdirectories')
        fo = collect_options(log_dir)
        print('Found:')
        for d in fo.keys():
            print(d)
            # update location in config
            fo[d].file.log_dir = d + '/'
        oo = [*fo.values()]
    
    os.chdir(scriptdir)
    
    for o in oo:
        o.train.epochs = max(o.train.epochs, epochs)
        if debug:
            o.debug = True
        TrainControl(o).train()
