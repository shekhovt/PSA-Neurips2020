import context

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from experiments.args import *
from experiments.default import *
from experiments.data_loader import *
from experiments.construct import *
from gradeval.utils import *
from models.losses import *
from experiments.test import Test

import models


# from PIL import Image


def default_options():
    ROOT = "../"
    
    o = odict()
    #
    o.file = odict()
    o.file.root = ROOT
    o.file.dataset = odict()
    o.file.dataset.name = 'MNIST'
    o.file.dataset.path = ROOT + 'data/'
    o.file.dataset.class_name = MNIST_dataset
    o.file.dataset.construct_args = odict()
    #
    #
    o.file.base_dir = ROOT + 'runs/'
    o.file.cuda = True
    o.file.verbose = True
    o.file.log_dir = o.file.base_dir + 'test/'
    #
    o.create = odict()
    o.create.network = 'fc'
    o.create.n_layers = 2
    o.create.hidden_size = 100
    o.create.add_batch_norm = False
    
    o.methods = odict()
    o.methods.init_temp = 1.0
    o.methods.activation = 'tanh'
    #
    o.train = odict()
    o.train.method = 'standard'
    # o.train.method = 'sah'
    # o.train.method = 'score'
    # o.train.method = 'gumbel'
    o.train.epochs = 100
    o.train.batch_size = 64
    # o.train.lr = 0.03
    o.train.lr = None
    o.train.lr_estimate_epochs = 5
    # o.train.samples = 1 # depricated
    o.train.init_seed = 1
    o.train.data_seed = 1
    o.train.val_size = 0.1
    o.train.optimizer = 'sgd'
    o.train.momentum = 0.9
    o.train.Nesterov = True
    o.train.train_samples = 1
    o.train.stat_samples = 10
    o.train.stochastic = False
    #
    #
    o.test = odict()
    o.test.batch_size = 128
    o.test.samples = 1
    o.test.stat_samples = 10
    #
    o.train.lr = None
    o.train.lr_estimate_epochs = 5
    o.train.lr_estimate_trials = 10
    o.train.step_lr = 100
    #
    # #data = MNIST_dataset(o.file.dataset_path)
    # data = new_dataset(o.file.dataset)
    #
    #
    o.create.input_shape = None
    o.create.input_size = None
    o.create.n_classes = None
    #
    o.train.epoch = None
    #
    o.lock()
    return o


def save_config(o, name):
    save_object(o.file.log_dir + name + '.pkl', o)
    with open(o.file.log_dir + name + '.txt', 'w') as f:
        print(*(s + os.linesep for s in o.strings()), file=f)


def edit_on_load(o):
    if o.train.optimizer == 'Adam' and o.train.Adam is not None and o.train.Adam.amsgrad:
        o.train.optimizer = 'Amsgrad'
        # del o.train.Adam
    return o


def collect_options(root_dir, exclude={}, must_include={}):
    #
    subd = [os.path.join(root_dir, o) for o in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, o))]
    subd.sort()
    ALL = odict()
    for d, dir_name in reversed(list(enumerate(subd))):
        info_name = dir_name + '/o.pkl'
        if not os.path.exists(info_name):
            print('Skipping (not exists) {}'.format(dir_name))
            del subd[d]
            continue
        o = pickle.load(open(info_name, "rb"))
        params = ', '.join(o.strings())
        if any(word in params for word in exclude) or not all(word in params for word in must_include):
            print('Skipping (by exclude/must_include) {}'.format(dir_name))
            del subd[d]
            continue
           
        
        o = edit_on_load(o)
        ALL[dir_name + '/'] = o
    return ALL


def uncommon_important(o, common):
    uncommon = o - common
    del uncommon.file
    del uncommon.train.epochs
    del uncommon.train.epoch
    if uncommon.create:
        del uncommon.create.input_shape
        del uncommon.create.n_classes
    del uncommon.test
    if uncommon.train:
        del uncommon.train.samples
    del uncommon.code
    del uncommon.debug
    if uncommon.train:
        del uncommon.train.init
        if not uncommon.train.step_lr:
            del uncommon.train.step_lr
            del uncommon.train.lr_estimate_trials
        
    if uncommon.train and not uncommon.train.backtrack:
        del uncommon.train.backtrack
    return uncommon


def configure(base_dir, oo):
    force_path(base_dir)
    # collect options from existing directories
    fo = collect_options(base_dir)
    # find common
    common = odict.intersection(*oo, *fo.values())
    del common.train.method
    # update existing items
    for [dir, o] in fo.items():
        # find uncommon
        uncommon = uncommon_important(o, common)
        # update dir name, if nothing is running...
        new_dir = os.path.join(base_dir, str(uncommon)) + '/'
        # rename
        c = True
        if False:
            if dir != new_dir:
                c = query_yes_no(f'Rename:? \n    {dir}\n to {new_dir}\n', default="yes")
            if c:
                os.rename(dir, new_dir)
                o.file.log_dir = new_dir
                save_config(o, 'o')
    
    # add new items
    for o in oo:
        # find uncommon
        uncommon = uncommon_important(o, common)
        new_dir = os.path.join(base_dir, str(uncommon)) + '/'
        c = True
        # if os.path.exists(new_dir):
        #     c = query_yes_no(f'Path {dir} exists, overwrite config?'.format(), default="no")
        if c:
            force_path(new_dir)
            o.file.log_dir = new_dir
            save_config(o, 'o')
    # return full list of configs
    
    fo = collect_options(base_dir)
    return [*fo.values()]
