from model import *
from drawing import *
from options import *

import scipy.stats
import pickle

from config import *


def plot_regression(root_dir, exp_name, model_name):
    figsize = (6.0, 6.0 * 3 / 4)
    #figsize = (3.0, 2)
    
    #
    res_name = root_dir + exp_name + '/' + model_name + '.pkl'
    res = pickle.load(open(res_name, "rb"))
    
    plt.figure(1, figsize=figsize)
    
    plt.plot(res.X.numpy(), res.Y.numpy(), '.k')
    # draw model samples
    n_samples = 100
    for s in range(n_samples):
        obj = lambda eta: eta  # just output prediction
        Y = res.net.forward(res.X, **dict(res.kwargs, method='score'), obj=obj)
        plt.plot(res.X.numpy(), Y.detach().numpy(), '.', markerfacecolor='r', markeredgewidth=0, marker='o', alpha=1 / n_samples, ms=10)
    
    plt.legend(loc=1)
    ax = plt.gca()
    plt.xlabel(" input ")
    plt.ylabel(" prediction ")
    res_file = root_dir + exp_name + '/plot/' + model_name + '.pdf'
    save_pdf(res_file)
    plt.close(1)
    
    plt.figure(2, figsize=figsize)
    UB, = plt.plot(np.arange(len(res.NLL0)), res.NLL0, '--', label='UB')
    plt.plot(np.arange(len(res.NLL1)), res.NLL1, '-', label='negative log likelihood', color=UB.get_color())
    plt.legend(loc=1)
    ax = plt.gca()
    plt.xlabel(" iterations ")
    res_file = root_dir + exp_name + '/training_' + model_name + '.pdf'
    save_pdf(res_file)
    plt.close(2)


def arange_uniform(X, N):
    m = X.min().item()
    M = X.max().item()
    s = (M - m) / (N - 1)
    return torch.arange(m, M + s - 0.0001, s)


def plot_class(root_dir, exp_name, model_name):
    figsize = (6.0, 6.0 * 3 / 4)
    #
    res_name = root_dir + exp_name + '/' + model_name + '.pkl'
    res = pickle.load(open(res_name, "rb"))
    
    res.net.cpu()
    res.X = res.X.cpu()
    res.Y = res.Y.cpu()
    
    for st in ['sigmoid', 'stoch', 'determ', 1, 2, 3, 4]:
        plt.figure(1, figsize=figsize)
        ax = plt.gca()
        m = res.Y.view([-1]) > 0
        plt.plot(res.X[m, 0].cpu().numpy(), res.X[m, 1].cpu().numpy(), 'or', label='training data')
        m = res.Y.view([-1]) < 0
        plt.plot(res.X[m, 0].cpu().numpy(), res.X[m, 1].cpu().numpy(), 'xb', label='training data')

        ax.axis('equal')
        ax.axis('off')
        res_file = root_dir + exp_name + '/data.pdf'
        save_pdf(res_file)
        
        N = 50
        X0 = arange_uniform(res.X[:, 0], N)
        X1 = arange_uniform(res.X[:, 1], N)
        # X0 = np.arange(-1.5,1.5,0.1)
        # X1 = np.arange(-1.5,1.5,0.1)
        N = len(X0)
        Z = np.zeros([N, N])
        if isinstance(st, numbers.Number) and st != 'stoch':
            n_samples = 1
        else:
            n_samples = 1000
        for i in range(N):
            for j in range(N):
                if isinstance(st, numbers.Number):
                    torch.cuda.manual_seed_all(st)  # use the same noise sample in all data points (ensemble)
                    torch.manual_seed(st)
                Xs = torch.tensor([X0[i], X1[j]]).view([1, -1]).expand(n_samples, -1).type_as(res.X)
                obj = lambda eta: torch.sigmoid(eta)  # output probability of class 1
                if st == 'sigmoid':
                    # determenistic approx, AP1
                    Y = res.net.forward(Xs, **dict(res.kwargs, method='AP1'), obj=obj)
                elif st == 'determ':
                    # hadr thresholds
                    Y = res.net.forward(Xs, **dict(res.kwargs, method='determ'), obj=obj)
                else:  # draw model samples
                    Y = res.net.forward(Xs, **dict(res.kwargs, method='score'), obj=obj)
                Z[j, i] += Y.mean().item()
        
        # plt.plot(res.X.numpy(), Y.detach().numpy(), '.', markerfacecolor='r', markeredgewidth=0, marker='o', alpha=1 / n_samples, ms=10)
        plt.contourf(X0, X1, Z, cmap='coolwarm', levels=10)
        # cset = ax.contour(X0, X1, Z, colors='k')
        # ax.clabel(cset, inline=1, fontsize=10)
        #
        # plt.legend(loc=1)
        ax.axis('equal')
        ax.axis('off')
        res_file = root_dir + exp_name + '/plot/' + model_name + '-{}'.format(st) + '.pdf'
        save_pdf(res_file)
        plt.close(1)


def plot_gen(root_dir, exp_name, model_name):
    figsize = (6.0, 6.0 * 3 / 4)
    #
    res_name = root_dir + exp_name + '/' + model_name + '.pkl'
    res = pickle.load(open(res_name, "rb"))
    
    plt.figure(1, figsize=figsize)
    ax = plt.gca()
    plt.plot(res.Y[:, 0].numpy(), res.Y[:, 1].numpy(), '.k', label='training data')
    
    N = 50
    X0 = arange_uniform(res.Y[:, 0], N)
    X1 = arange_uniform(res.Y[:, 1], N)
    # X0 = np.arange(-1.5,1.5,0.1)
    # X1 = np.arange(-1.5,1.5,0.1)
    N = len(X0)
    Z = np.zeros([N, N])
    n_samples = 1000
    
    Xs = torch.tensor([1.0]).view([1, 1]).expand(n_samples, 1)
    obj = lambda eta: eta  # output point
    Y = res.net.forward(Xs, **dict(res.kwargs, method='score'), obj=obj).detach()
    # for s in range(Y.size(0)):
    #     i = math.floor((Y[s, 0]-X0[0]).item())
    #     j = math.floor(Y[s, 1].item())
    #     Z[j, i] += 1
    
    plt.plot(Y[:, 0].numpy(), Y[:, 1].numpy(), '.', markerfacecolor='b', markeredgewidth=0, marker='o', alpha=1 / 50, ms=10)
    # plt.contourf(X0, X1, Z, cmap='coolwarm', levels=10)
    # cset = ax.contour(X0, X1, Z, colors='k')
    # ax.clabel(cset, inline=1, fontsize=10)
    #
    # plt.legend(loc=1)
    ax.axis('equal')
    # ax.axis('off')
    res_file = root_dir + exp_name + '/plot/' + model_name + '.pdf'
    save_pdf(res_file)
    plt.close(1)


def plot_trainig(root_dir, exp_name):
    np.seterr(over='raise')
    # figsize = (3.0, 2.0)
    figsize = (4.0, 3.0)
    #figsize = (6.0, 6.0 * 3 / 4)
    # figsize = (12.0, 12.0 * 3 / 4)
    plt.figure(2, figsize=figsize)
    
    methods = all_methods
    
    for kwargs in methods:
        method = kwargs.method
        t = kwargs.t
        model_name = method
        if method == 'concrete' or method == 'SA':
            model_name += '-t={}'.format(t)
        try:
            res_name = root_dir + exp_name + '/' + model_name + '.pkl'
            res = pickle.load(open(res_name, "rb"))
        except FileNotFoundError:
            continue
        model_name = model_name.replace('SAH1','PSA')
        
        UB, = plt.plot(np.arange(len(res.NLL0)), res.NLL0, '--', label=None, color=kwargs.color)
        # plt.plot(np.arange(len(res.NLL1)), res.NLL1, '-', label=model_name, color=UB.get_color())
        x = np.arange(len(res.NLL1))
        y = res.NLL1  # adaptively averaged
        yerr = np.zeros_like(y)
        mse = RunningStat(0.95)  # filter to compute error bounds
        for i in range(len(res.NLL2)):
            mse.update((res.NLL2[i] - y[i]) ** 2)
            yerr[i] = math.sqrt(mse.mean)
        errorbar_fency(x, y, yerr=yerr, fmt='-', label=model_name, color=UB.get_color())
        # plt.plot(np.arange(len(res.NLL2)), res.NLL2, '-', label=None, color=UB.get_color(), linewidth=0.1)
    
    plt.legend(loc=1)
    ax = plt.gca()
    ax.set_yscale("log")
    plt.xlabel(" epochs ")
    # plt.ylim(0.2, 0.7)
    res_file = root_dir + exp_name + '/training.pdf'
    save_pdf(res_file)
    plt.close(2)


if __name__ == "__main__":
    root_dir = '../../exp/class/'
    plot_trainig(root_dir, '5-5-5')
    # root_dir = './exp/regression/'
    # exp_name = 'train-10x5'
    # model_name = 'SA-t=0'
    # model_name = 'score'
    # root_dir = './exp/determenistic/'
    # exp_name = 'train-3x3'
    # model_name = 'SA-t=0'
    
    # plot_model(root_dir, exp_name, model_name)
    # plot_trainig(root_dir, exp_name)
    
    # root_dir = './exp/class/'
    # exp_name = 'train-3x3'
    # model_name = 'SA-t=0'
    # model_name = 'AP1'
    # plot_class(root_dir, '5-5-5', model_name)
    # plot_trainig(root_dir, 'cos-s10-b10-m')
    # plot_class(root_dir, 'cos-s10-b10-t', 'score')
    # plot_class(root_dir, 'cos-s10-b10-m', 'score')
    # plot_class(root_dir, 'cos-s10-running', 'SAH')
    # plot_trainig(root_dir, 'cos-s10-running')
    # plot_trainig(root_dir, 'cos-s10-deep5-LB')
    # plot_trainig(root_dir, 'cos-s10-wide5-LB')
    # plot_class(root_dir, 'cos-s10-deep5-LB','AP1')
    # plot_trainig(root_dir, 'cos-s10-running-slo')
    # plot_class(root_dir, 'cos-s10-b10-t', 'AP1')
    # plot_trainig(root_dir, 'cos-s10-deep3-LB-l003')
    # plot_trainig(root_dir, 'cos-s10-deep5-LB-l003')
    # plot_trainig(root_dir, 'cos-s10-deep6-LB')
    
    # root_dir = './exp/gen/'
    # exp_name = 'train-3x3'
    # exp_name = 'mix'
    # model_name = 'SA-t=0'
    # model_name = 'AP1'
    # plot_gen(root_dir, exp_name, model_name)
    # plot_trainig(root_dir, exp_name)
