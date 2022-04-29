import context

from gradeval.model import *
from gradeval.drawing import *
from gradeval.options import *

import scipy.stats
import pickle
from gradeval.config import *

# figsize = (3.0, 3.0 * 3 / 4)
figsize = (6.0, 6.0 * 3 / 4)


# root_dir = './exp/3x8/'

# simple test -- usage example


def test_grad(root_dir, exp_name, point):
    print('Test grad, point:{}'.format(point))
    # load some model
    # res_name = root_dir + exp_name + '/' + 'score' + '.pkl'
    res_name = root_dir + exp_name + '/' + point + '.pkl'
    res = pickle.load(open(res_name, "rb"))
    net = res.net
    X = res.X
    Y = res.Y
    o = res.o
    
    # X = X.cuda()
    # Y = Y.cuda()
    # net.cuda()
    
    # exact gradient
    # obj_batch = lambda eta: likelihood(eta, Y)
    obj_batch = lambda eta: NLL(eta, Y)
    net.zero_grad()
    EE = net.forward(X, method='enumerate', obj=obj_batch).mean()  # mean over training data
    EE.backward()
    G_t = net.grad_list()
    for g in G_t:
        print(g)
    
    # sampling based gradients
    n_samples = 20000
    sample_batch = 1  # do not use this to not mess up with the statistics

    # for metric in [odict(label='Cosine similarity', name='cos')]:
    # for metric in [odict(label='Relative RMSE', name='rmse')]:
    for metric in [odict(label='Relative RMSE', name='rmse'), odict(label='Cosine similarity', name='cos')]:
        
        # methods = [odict(method='score'), odict(method='concrete', t=0.1), odict(method='concrete', t=1), odict(method='AP1'), odict(method='ST'), odict(method='SA', t=0), odict(method='SAH')]
        # methods = [odict(method='score'), odict(method='concrete', t=0.1), odict(method='concrete', t=1), odict(method='AP1'), odict(method='ST'), odict(method='SAH'), odict(method='ARM')]
        methods = [odict(method='score'), odict(method='concrete', t=0.1), odict(method='concrete', t=1), odict(method='AP1'), odict(method='ST'),
                   odict(method='SAH'), odict(method='ARM')]
        # methods = [odict(method='score'), odict(method='SAH1'), odict(method='SA', t=0), odict(method='SAH'), odict(method='ST')]
        # methods = [odict(method='SAH1'), odict(method='SAH')]
        # methods = [odict(method='score'), odict(method='SAH1'), odict(method='ARM')]
        # methods = [odict(method='ARM')]
        
        methods = select_methods(methods)
        for kwargs in methods:
            method = kwargs.method
            t = kwargs.t
            print("_______________________ Method: %s" % method)
            # ngroups = [1, 5,10, 50, 100, 500, 1000]
            # ngroups = np.ceil(10 ** np.arange(0, math.log10(n_samples), 0.1))
            ngroups = np.arange(1, 10, 1)
            if n_samples >= 100:
                ngroups = np.append(ngroups, range(10, 100, 10))
            if n_samples >= 1000:
                ngroups = np.append(ngroups, range(100, 1000, 100))
            if n_samples >= 10000:
                ngroups = np.append(ngroups, range(1000, 10000, 1000))
            
            # small accumulator for M-sample estimate
            G_acc = [[RunningStatAvg() for n in np.nditer(ngroups)] for l in range(len(G_t))]
            # large accumulator of different trials of M-sample estimates
            G_s = [[RunningStatAvg() for n in np.nditer(ngroups)] for l in range(len(G_t))]
            G_trials = [[ [] for n in np.nditer(ngroups)] for l in range(len(G_t))]
            for sample in range(n_samples):
                # draw grad sample
                net.zero_grad()
                EE = net.forward(X, method=method, sample_batch=1, tau=t, t=t, obj=obj_batch).mean()  # mean over training data
                EE.backward()
                G = net.grad_list()
                for l in range(len(G_t)):
                    g_t = G_t[l]
                    g_s = G[l]
                    for gr in range(len(ngroups)):
                        G_acc[l][gr].update(g_s.detach())
                        if G_acc[l][gr].n == ngroups[gr]:  # accumulator full
                            g = G_acc[l][gr].mean  # M-sample estimator
                            # compute its error
                            if metric.name == 'cos':
                                v1 = F.cosine_similarity(g, g_t, dim=0).item()
                            elif metric.name == 'rmse':
                                v1 = torch.sum((g - g_t) ** 2) / (g_t.norm(2) ** 2)
                            G_s[l][gr].update(v1)
                            G_trials[l][gr].append(v1)
                            # reset accumulator
                            G_acc[l][gr] = RunningStatAvg()
            
            for l in range(len(G_t)):
                x = ngroups
                if metric.name == 'cos':
                    y = np.array([s.mean for s in G_s[l]])
                    yerr = np.array([math.sqrt(s.std_of_mean ** 2 + s.var) for s in G_s[l]])
                elif metric.name == 'rmse':
                    y = np.array([s.mean for s in G_s[l]]) ** 0.5
                    yerr = np.array([math.sqrt(s.std_of_mean ** 2 + s.var) for s in G_s[l]]) ** 0.5
                y1 = np.zeros(len(G_s[l]))
                y2 = np.zeros(len(G_s[l]))
                for gr in range(len(ngroups)):
                    yy = np.array(G_trials[l][gr])
                    yy = np.sort(yy)
                    ny = len(yy)
                    y1[gr] = yy[math.ceil(ny * 0.15)]
                    y2[gr] = yy[math.ceil((ny - 1) * 0.85)]
                # plot comparison
                plt.figure(l, figsize=figsize)
                
                label = method
                if method == 'concrete' or method == 'SA':
                    label += '-t={}'.format(t)
                # plt.plot(n, n_rmse, label=label)
                
                label = label.replace('score', 'REINFORCE')
                label = label.replace('AP1', 'tanh')
                label = label.replace('AP2', 'ADF')
                label = label.replace('concrete', 'Concrete')
                label = label.replace('SAH', '$\\bf PSA$')
                label = label.replace('SA-t=0', '$\\bf SA$-t=0')
                label = label.replace('ST', '$\\bf ST$')
                
                ylog = metric.name == 'rmse'
                
                # fmt = '-'
                fmt = '-' + kwargs.marker
                
                # if method == 'SAH':
                #     fmt += '+'
                # if method == 'ST':
                #     fmt += '*'
                
                pargs = {'markersize': 5, 'linewidth': 2, 'label': label, 'fmt': fmt, 'color': kwargs.color, 'ylog': ylog}

                if metric.name == 'rmse':
                    errorbar_fency(x, y, yerr=None, **pargs)
                else:
                    errorbar_fency(x, y, yl=y1, yu=y2, **pargs)
                
                ax = plt.gca()
                plt.xlabel("# samples")
                plt.ylabel(metric.label)
                if metric.name == 'cos':
                    plt.legend(loc=4)
                ylog = False
                if metric.name == 'rmse':
                    ax.set_yscale("log")
                    plt.legend(loc=1)
                    ylog = True
                
                ax.set_xscale("log")
                plt.xlim(1, ngroups.max())
                # plt.title('Layer {}'.format(l))
                plt.tick_params(axis='y', which='both', left='on', right='on', labelleft='on')
                # res_file = root_dir + exp_name + '/test_grad/' + point +'-e{}'.format(o.epochs) + '/'
                res_file = root_dir + exp_name + '/test_grad/' + point + '/' + metric.name + '-layer-{}.pdf'.format(l)
                save_pdf(res_file)
        
        plt.close('all')


if __name__ == "__main__":
    # root_dir = '../../exp/gradeval/'
    # test_grad(root_dir, 'chk-1', 'score')
    # test_grad(root_dir, 'chk-200', 'score')
    # test_grad(root_dir, 'chk-1000', 'score')
    # root_dir = '../../exp/class/5-3-3-3/'
    # root_dir = '../../exp/class/5-5-5/'
    #
    root_dir = '../../exp/gradeval/5-5-5/'
    chk = 'score'
    test_grad(root_dir, 'chk-1', chk)
    test_grad(root_dir, 'chk-100', chk)
    test_grad(root_dir, 'chk-2000', chk)
