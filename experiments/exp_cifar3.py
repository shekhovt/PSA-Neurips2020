import context
from experiments.config import *
from models import methods
from experiments.construct import *
from experiments.train import *

o = default_options()
o.unlock()
#
o.file.dataset.name = 'CIFAR'
o.file.dataset.class_name = CIFAR10_dataset_logits.__name__
o.file.dataset.construct_args = odict(scaling=0.5)
#
o.create.network = 'CIFAR_S1'
#
o.train.stochastic = True
o.train.epochs = 2000
o.train.loss = ExpectedLoss_NLL.__name__

# o.train.preproc = None  #'Affine'
# o.train.preproc = 'Affine'
o.train.preproc = 'FlipCrop'

o.train.norm = None

o.train.optimizer = 'sgd'
o.train.step_lr = False
#
cases = []
mm = ['sah', 'gumbel', 'standard', 'score', 'AP2', 'LocalReparam', 'ST', 'HardST']
data_net_dir = o.file.base_dir + o.file.dataset.name + '/' + o.create.network + '-' + data_net_dir + 'NLL-' + o.train.preproc + '/'

# configure
oo = []  # list of options to run
for o.train.method in mm:
    oo += [o.clone()]

oo1 = copy.copy(oo)
oo = configure(o.file.base_dir, oo)

for o in oo1:
    tc = TrainControl(o)
    tc.train()
