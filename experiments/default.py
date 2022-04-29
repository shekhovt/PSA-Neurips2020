from .options import *


def all_options():
    o = multioption()
    #
    # 'file' options do not affect train or test results
    #
    o.file = []
    o.file.version = '1.0'
    o.file.dataset = ''
    o.file.dataset.path = ''
    o.file.model = ''
    o.file.model.path = ''
    o.file.num_workers = 0
    o.file.allow_cuda = True
    o.file.verbose = True
    #
    #
    # training options
    #
    #
    o.train = []
    o.train.seed = 0
    o.train.optimizer = ['SGD', 'Adam']
    o.train.optimizer.SGD.Momentum = 0.9
    o.train.optimizer.SGD.Nesterov = True
    o.train.weight_decay = 0
    o.train.batch_size = 32
    o.train.init_batch_size = 128
    #
    o.train.reg = [None, 'VB']
    o.train.init = [None, 'BN', 'AP2']
    o.train.lr = [None, float]
    o.train.lr[None].lr_estimate_epochs = 5
    o.train.lr_down = ['exp', 'sqrt', 'const']
    o.train.lr_down.exp.base = 0.99426
    #
    #
    # # test options
    #
    #
    o.test = []
    o.test.inference = ['AP1']
    return o
