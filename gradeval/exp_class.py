import context

from model import *
from drawing import *
from options import *

import scipy.stats
import pickle
from gradeval import plot_regression
import torch.utils.data
from gradeval.config import *

# setup regression problem
seed = 1
data_seed = 1
torch.cuda.manual_seed_all(data_seed)
torch.manual_seed(data_seed)
dtype = torch.float64

N = 100

X = torch.empty(0, 2)
Y = torch.empty(0, 1)

X = torch.empty([N, 2])
X[:, 0].uniform_(-math.pi / 2, math.pi / 2)
X[:, 1].uniform_(0, 1)
Y = torch.ones(N, 1) * 1
X2 = torch.empty([N, 2])
i = 0
while i < N:
    x = torch.empty([1]).uniform_(-math.pi / 2, math.pi / 2)
    y = torch.empty([1]).uniform_(-1, 1)
    if y < x.cos():
    # if y < (2 * x + math.pi / 2).cos().abs():
        X2[i, 0] = x
        X2[i, 1] = y
        i = i + 1
# X2[:, 0].uniform_(-1, 1)
# X2[:, 1] = torch.empty(N).uniform_(0, 1) * (1 + torch.cos(X2[:, 0] * math.pi / 2)) - 1

X = torch.cat([X, X2], dim = 0)
Y = torch.cat([Y, torch.ones(N, 1) * (-1)], dim=0)

# for y in [-1, 1]:
#     Xy = torch.empty(N, 2)
#     rho = torch.empty(N).uniform_().sqrt()
#     theta = torch.empty(N).uniform_(0, 2 * math.pi)
#     Xy[:, 0] = rho * theta.cos()
#     Xy[:, 1] = rho * theta.sin() + y * 0.5
#     X = torch.cat([X, Xy], dim=0)
#     Yy = torch.ones(N, 1) * y
#     Y = torch.cat([Y, Yy], dim=0)  # target class +-1 encoding

# Logistic distribution on the output
# def likelihood(eta, y):
#     return torch.sigmoid(eta * y)
#
# NLL = lambda eta, y: log1p_exp(-eta * y)
# NLL = lambda eta, y: -torch.log(torch.sigmoid(eta * y))

# training

def test2():
    o = odict()
    o.n_samples = 1
    o.n_samples_ref = 10
    # o.lr0 = 0.03
    # o.lr0 = 0.001
    # o.lr0 = 3
    # o.lr0 = 3  # 0.1
    # o.lr0 = 0.1
    o.lr0 = 0.03
    o.try_load = False
    # o.N = [2, 10, 3]
    # o.N = [2, 5, 3, 3]
    # o.N = [2, 5, 3, 3]
    # o.N = [2, 5, 3, 3, 3, 3]
    # o.N = [2, 5, 3, 3, 3, 3, 3]
    o.N = [2, 5, 5, 5]
    # o.N = [2, 5, 3, 3, 3]
    o.root_dir = '../../exp/class/'
    # o.exp_name = 'cos-s10-deep3-LB-EM'
    # o.exp_name = 'cos-s10-deep3-LB-l003'
    # o.exp_name = 'cos-s10-deep5-LB-l003'
    # o.exp_name = 'cos-s10-deep6-LB'
    # o.exp_name = '5-3-3-3'
    o.exp_name = '5-5-5'
    o.ML = False
    # o.ML_grad = 'LB_running'
    o.ML_grad = 'LB'
    o.batch_size = 10  # X.size(0)  # full batch
    o.optimizer = 'SGD'
    o.SGD_momentum = 0.9  # 0.9
    o.SGD_Nesterov = False
    o.epochs = 2000
    o.checkpoints = [1, 100, 200, 500, 1000, 1500, 2000]
    
    methods = []
    # methods.append(odict(method='score'))
    # methods.append(odict(method='ARM'))
    # methods.append(odict(method='ST'))
    # methods.append(odict(method='SA', t=0))
    methods.append(odict(method='SAH1'))
    # methods.append(odict(method='SA', t=1))
    # methods.append(odict(method='concrete', t=1))
    # methods.append(odict(method='AP1'))
    # methods = [odict(method='enumerate')]
    # methods.append(odict(method='concrete', t=0.1))
    # methods = [odict(method='AP1'), odict(method='ST'), odict(method='score'), odict(method='concrete', t=0.1), odict(method='SA', t=0)]
    # methods = [odict(method='score'), odict(method='concrete', t=0.1), odict(method='ST'), odict(method='AP1'), odict(method='SA', t=0), odict(method='SAH')]
    # methods = [odict(method='ST'), odict(method='score'), odict(method='concrete', t=0.1)]

    methods = select_methods(methods)
    
    o0 = o.copy()
    for kwargs in methods:
        o = o0
        o = odict(o, **kwargs)
        o.t = kwargs.t
        o.tau = kwargs.t
        model_name = o.method
        if o.method == 'concrete' or o.method == 'SA':
            model_name += '-t={}'.format(o.t)
        if o.method == 'enumerate' or o.method == 'AP1':
            o.n_samples = 1
        o.model_name = model_name
        print("_______________________ Method: %s" % o.method)
        # if o.method == 'score':
        #     o.lr0 = 0.0001
        # create and initialize network
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        l0 = InputLayer(out_units=o.N[0])
        ll = []
        li = l0
        it = enumerate(o.N)
        next(it)
        for (i, n) in it:
            li = LogisticBernoulliLayer(out_units=o.N[i], prev=li)
            ll.append(li)
        lN = OutputLayer(prev=li)
        net = l0
        #
        # net.cuda()
        # Xc = X.cuda()
        # Yc = Y.cuda()
        Xc = X.to(dtype=dtype)
        Yc = Y.to(dtype=dtype)
        net.to(dtype=dtype)
        
        # data loader
        class IndexTensorDataset(torch.utils.data.TensorDataset):
            def __getitem__(self, index):
                data, target = torch.utils.data.TensorDataset.__getitem__(self, index)
                return data, target, index
        train_dataset = IndexTensorDataset(Xc, Yc)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=o.batch_size, shuffle=True)
        # init weights
        NX = RandomVar(Xc.mean(dim=0, keepdim=True), Xc.var(dim=0, keepdim=True))
        net.forward(NX, method='AP2-init')
        #
        # callback
        call_back = lambda: (plot_regression.plot_class(o.root_dir, o.exp_name, o.model_name),
                                            plot_regression.plot_trainig(o.root_dir, o.exp_name))
        # train
        train_ML(net, Xc, Yc, train_loader, logistic_likelihood, logistic_NLL, o, call_back)

test2()
