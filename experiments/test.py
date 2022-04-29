#import models
from .data_loader import *
from .construct import *
import numpy as np
import scipy.optimize
import math
from gradeval.utils import *
from models.methods import *
from .data_loader import *

class Test:
    def __init__(self, o, net, data):
        self.o = o
        self.net = net
        self.data = data
        self.new_loaders()
        
    def new_loaders(self):
        torch.manual_seed(self.o.train.init_seed)
        self.train_loader, self.val_loader = get_train_valid_loader(self.data.train_dataset, train_batch_size=self.o.train.batch_size,
                                                                    val_batch_size=1,
                                                                    random_seed=self.o.train.data_seed,
                                                                    valid_size=self.o.train.val_size, shuffle=True, num_workers=0,
                                                                    pin_memory=False, train_transform=self.data.train_transform,
                                                                    val_transform=self.data.test_transform)

        self.test_loader1 = DataLoaderIndex(self.data.test_dataset, batch_size=1, num_workers=0, transform=self.data.test_transform)

    def test_accuracy_samples(self, method, n_samples=128):
        """
        :return: test accuracy of the model using varying number of samples
        """
        loader = self.test_loader1
        n_data = len(loader.sampler)
        batch_size = 1  # will use samples instead
        n_batches = divup(n_data, batch_size)
        device = next(self.net.parameters()).device
        #
        iter = enumerate(loader)
        if self.o.train.norm == 'AP2':
            method = MethodWithNorm(method=method)

        total_acc = 0
        total_data = 0
        while True:
            try:
                batch_idx, (data_idx, data, ctarget, *targets) = next(iter)
            except StopIteration:
                # if StopIteration is raised, break from loop
                break
            data = data.to(device=device)
            target = ctarget.to(device=device)
            with torch.no_grad():
                self.net.train(False)  # test mode on
                    
                sz = list(data.size())
                sz[0] = n_samples
                data = data.expand(sz)
                p = self.net.forward(data, method=method, objs=obj_softmax)
                c1 = torch.arange(1, n_samples + 1, dtype=p.dtype, device=p.device)
                cp = p.cumsum(dim=0)
                mp = cp / c1.view([-1, 1])
                pred = mp.argmax(dim=-1)
                acc = (pred == target).to(dtype=p.dtype)  # [S]
                total_acc = acc + total_acc
                total_data += 1
            # if batch_idx >= 100:
            #     break
        acc = total_acc / total_data
        return acc


def cache_test(o, res_name, compute):
    cache_name = o.file.log_dir + res_name + '.pkl'
    model_name = o.file.log_dir + 'model_state.pkl'
    if os.path.exists(cache_name) and os.path.getmtime(cache_name) > os.path.getmtime(model_name):
        # cache is up to date
        print('Loading ', res_name)
        res = pickle.load(open(cache_name, "rb"))
    else:
        print('Computing ', res_name)
        res = compute()
        save_object(cache_name, res)
    return res
