import context
import os
import inspect
import sys
from experiments.drawing import *
from experiments.options import *

from experiments.options import odict
from experiments.data_loader import *
from experiments.save_load import *
from experiments.test import *
from experiments.train_hist import *
from experiments.config import *
from itertools import cycle
from gradeval.utils import *

np.seterr(over='raise')

figsize = (6.0, 6.0 * 3 / 4)
# figsize = (12.0, 12.0 * 3 / 4)
# figsize = (24.0, 24.0 * 3 / 4)
show_legend_loss = False
show_legend_acc = True
show_legend_entropy = False

# exclude = {'PSGD2', 'HingeSQ', 'LocalReparam'}
exclude = {'PSGD2', 'HingeSQ', 'lr=0.'}  # exclude manual lr trials
# exclude = {'HingeSQ', 'LocalReparam'}
# exclude = {'HingeSQ', 'LocalReparam', 'HardST', 'standard', 'AP2', 'gumbel', 'sah', 'score'}
# exclude = {'HingeSQ', 'LocalReparam'}
# must_include = {'AP2'}
must_include = {}

show_conf = True
#show_conf = False

all_methods = [
    odict({'train.method': 'standard', 'name': 'tanh'}),
    odict({'train.method': 'score', 'name': 'REINFORCE'}),
    odict({'train.method': 'gumbel', 'methods.init_temp': 1.0, 'name': 'Concrete'}),
    odict({'train.method': 'ST', 'name': '$\\bf ST$'}),
    odict({'train.method': 'sah', 'name': '$\\bf PSA$'}),
    odict({'train.method': 'AP2', 'name': 'ADF'}),
    odict({'train.method': 'LocalReparam', 'name': 'LocalReparam'}),
    odict({'train.method': 'enumerate', 'name': 'Exhaustive'}),
    odict({'train.method': 'gumbel', 'methods.init_temp': 0.1, 'name': 'Concrete(0.1)'}),
    odict({'train.method': 'HardST', 'name': 'Hard ST'}),
]

prop_cycle = plt.rcParams['axes.prop_cycle']
ccolors = [*prop_cycle.by_key()['color']]
ccycle = cycle(ccolors)
markers = 'oxd*+v^><s'
mcycle = cycle(markers)


# attach colors to all methods
for (i, m) in enumerate(all_methods):
    color_s = ccolors[i % len(ccolors)]
    m.color = color_s
    m.marker = markers[i % len(markers)]


def get_color(o):
    for m in all_methods:
        S = set((k, v) for (k, v) in m.items() if k not in {'color', 'name', 'marker'})
        So = set((v[0], tuple(v[1])) if isinstance(v[1], list) else v for v in o.items())
        if S <= So:  # subset
            return m.color, m.name, m.marker
            # return next(ccycle), m.name, next(mcycle)
    for m in all_methods:
        if m['train.method'] == o['train.method']:
            return next(ccycle), m.name, m.marker
    return next(ccycle), 'Unknown', ''


def format_name(method_name, uncommon):
    uncommon = uncommon.clone()
    del uncommon.train.method
    del uncommon.train.train_samples
    flat = odict(uncommon.flatten())
    if 'train.lr' in flat and flat['train.lr'] is not None:
        flat['train.lr'] = format_digits(flat['train.lr'], 2)  # string
    params = ', '.join(flat.strings())
    # params = params.replace('train.lr', 'lr')
    params = params.replace('train.', '')
    params = params.replace('ExpectedLoss_', '')
    params = params.replace('precond.optimizer=', '')
    return method_name + ' ' + params


def get_color_method(o, uncommon):
    of = odict(o.flatten())
    color, method_name, marker = get_color(of)
    name = format_name(method_name, uncommon)
    return color, name, marker

def collect_results(root_dir, jobs):
    #
    plt.close("all")
    
    ALL = collect_options(root_dir, exclude, must_include)
    #
    # subd = [os.path.join(root_dir, o) for o in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, o))]
    # subd.sort()
    # ALL = odict()
    # for d, dir_name in reversed(list(enumerate(subd))):
    #     info_name = dir_name + '/o.pkl'
    #     if not os.path.exists(info_name):
    #         print('Skipping {}'.format(dir_name))
    #         del subd[d]
    #         continue
    #     o = pickle.load(open(info_name, "rb"))
    #     # args = odict(o.flatten())
    #     # for k,v in args.items():#
    #     #     if isinstance(v, list) or isinstance(v, tuple) and len(v)>0 and isinstance(v[0], list):
    #     #         all_args[k] = tuple(tuple(l) for l in v)
    #     # apairs = set(args.items())
    #     # if apairs & set(skip_exp.items()) != set() or not set(select_exp.items()) <= apairs or all_args.epochs < 2 or '-X' in dir_name:
    #     #     print('Skipping {} - epochs {}'.format(dir_name, all_args.epochs))
    #     #     del subd[d]
    #     # ARGS[dir_name] = args
    #     ALL[dir_name] = o
    #
    # take out common params
    common = odict.intersection(*ALL.values())
    print("Common params: ")
    print(*common.flatten())
    subd = [*ALL.keys()]
    
    global axes
    axes = []
    
    for d in range(len(subd)):
        dir_name = subd[d]
        print("Case: {}".format(dir_name))
        try:
            o = pickle.load(open(dir_name + '/o1.pkl', "rb"))
        except:
            o = pickle.load(open(dir_name + '/o.pkl', "rb"))
        
        o = edit_on_load(o)
        
        uncommon = uncommon_important(o, common)
        uncommon.file = odict()
        print("Uncommon params: ")
        print(*uncommon.flatten())
        # plot cases
        o.file.log_dir = dir_name + '/'
        #
        color, model_name, marker = get_color_method(o, uncommon)
        for (i, job) in enumerate(jobs):
            plt.figure(i, figsize=figsize)
        #
        for (i, job) in enumerate(jobs):
            plt.figure(i)
            job(dir_name, o, common, uncommon, color=color, model_name=model_name, marker=marker)
    
    for (i, job) in enumerate(jobs):
        plt.figure(i)
        res_file = root_dir + '/' + job.__name__ + '.pdf'
        save_pdf(res_file)


def smooth(x, y, r=None):
    # r = RunningStatAdaptive(0.5)
    if r is None:
        if len(x) > 2 * x.max():  # subepoch values, do more smoothing
            r = RunningStatAdaptive(0.5)
        else:
            r = RunningStatAdaptive(0.3)
        # r = RunningStat(q=0.95)
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    for i, l in enumerate(y):
        r.update(y[i])
        m[i] = r.mean
        v[i] = r.var
    return m, v


def subsample(x, *vars):
    # subsample
    mask = x == np.floor(x)
    return (x[mask],) + tuple(v[mask] for v in vars)


def train_loss(dir_name, o, common, uncommon, train=True, color=None, model_name=None, marker='', **kwargs):
    # flat = odict(uncommon.flatten())
    # if 'train.lr' in flat:
    #     flat['train.lr'] = format_digits(flat['train.lr'], 2)  # string
    # model_name = ', '.join(flat.strings())
    #print('model_name=', model_name)
    # surpress some warnings
    
    res_name = dir_name + '/hist.pkl'
    try:
        res = pickle.load(open(res_name, "rb"))
    except FileNotFoundError:
        return
    # color by method
    # of = odict(o.flatten())
    #
    epochs = o.train.epochs
    if train:
        res = res.train
    else:
        res = res.val
    # for k in res.keys():
    #     v = res[k]
    #     step = max(1, len(v) // epochs)
    #     res[k] = v[::step]
    #
    # x = np.arange(len(res.r_loss))
    if len(res.r_loss) < 20:
        return
    x = res.epoch
    ci_r_loss = res.loss_var_params_samples ** 0.5
    v_e = np.zeros_like(res.u_loss)
    v_e[1:] = (res.u_loss[1:] - res.u_loss[:-1]) ** 2 / 2
    # ci_u_loss = (res.loss_var_samples + res.loss_var_params) ** 0.5
    # ci_u_loss = (res.loss_var_samples + v_e) ** 0.5
    ci_u_loss = (res.loss_var_samples) ** 0.5
    #
    (y_m, y_v) = smooth(x, res.r_loss)

    (xs, y_m, y_v) = subsample(x, y_m, y_v)
    #
    kwargs = dict(markevery=int(math.ceil(len(xs) / 10)), label=None, color=color)
    UB, = plt.plot(xs, y_m, ':' + marker, **kwargs)
    # plt.plot(np.arange(len(res.NLL1)), res.NLL1, '-', label=model_name, color=UB.get_color())
    # plt.plot(x, res.u_loss, '-', label=model_name, color=color)

    (y_m, y_v) = smooth(x, res.u_loss)

    v_m, _ = smooth(x, res.loss_var_samples)
    ci = (v_m + y_v) ** 0.5 * 3

    (xs, y_m, ci) = subsample(x, y_m, ci)
    plt.gca().set_ylim(auto=True)
    errorbar_fency(xs, y_m, ci, fmt='-' + marker, label=model_name, color=UB.get_color(), show_conf=show_conf)
    plt.legend(loc=1)
    ax = plt.gca()
    ax.set_yscale("log")
    plt.xlabel(" epochs ")
    # plt.ylim(0.2, 0.7)
    ymin, ymax = plt.ylim()
    plt.ylim(bottom=max(1e-4, ymin))
    if not show_legend_loss:
        plt.gca().get_legend().remove()


def train_acc(dir_name, o, common, uncommon, train=True, color=None, model_name=None, marker='',**kwargs):
    # surpress some warnings
    np.seterr(over='raise')
    res_name = dir_name + '/hist.pkl'
    try:
        res = pickle.load(open(res_name, "rb"))
    except FileNotFoundError:
        return
    epochs = o.train.epochs
    if train:
        res = res.train
    else:
        res = res.val
    # for k in res.keys():
    #     v = res[k]
    #     step = max(1, len(v) // epochs)
    #     res[k] = v[::step]
    # #
    if len(res.acc1) < 20:
        return
    x = res.epoch
    #
    (y_m, y_v) = smooth(x, res.acc1)

    (xs, y_m, y_v) = subsample(x, y_m, y_v)

    kwargs = dict(markevery=int(math.ceil(len(xs) / 10)), label=None, color=color)
    UB, = plt.plot(xs, y_m * 100, ':' + marker, **kwargs)
    # UB, = plt.plot(x, res.acc1 * 100, ':', label=None)
    # plt.plot(np.arange(len(res.NLL1)), res.NLL1, '-', label=model_name, color=UB.get_color())
    # plt.plot(x, res.u_loss, '-', label=model_name, color=color)
    (y_m, y_v) = smooth(x, res.acc_e)
    y_v, _ = smooth(x, y_v, r=RunningStat(0.98))
    v_m, _ = smooth(x, res.acc1_var_samples)  # + res.acc1_var_params)
    ci = (v_m + y_v) ** 0.5 * 3
    ci1 = v_m ** 0.5 * 3
    ci2 = y_v ** 0.5 * 3  # variance from smoothing to represent the smoothed oscillations

    (xs, y_m, ci1, ci2, ci) = subsample(x, y_m, ci1, ci2, ci)

    errorbar_fency(xs, y_m * 100, ci * 100, fmt='-' + marker, label=model_name, color=UB.get_color(), show_conf=show_conf)
    # errorbar_fency(xs, y_m * 100, ci1 * 100, fmt='-', label=None, color=UB.get_color(), show_conf=show_conf)
    plt.legend(loc=4)
    ax = plt.gca()
    plt.xlabel(" epochs ")
    if "tanh" in model_name:
        ym = y_m[20] * 100 if len(y_m) > 20 else 10
        # plt.ylim(ym, 100)
        # plt.ylim(bottom=ym, auto=True)
        ym = 40
        top = 100 if train else 90
        plt.ylim(bottom=ym, top=top)
    # plt.ylim(bottom=ym)
    if not show_legend_acc:
        plt.gca().get_legend().remove()


def train_entropy(dir_name, o, common, uncommon, train=True, color=None, model_name=None, marker='',**kwargs):
    res_name = dir_name + '/hist.pkl'
    try:
        res = pickle.load(open(res_name, "rb"))
    except FileNotFoundError:
        return
    # color, model_name = get_color_method(o, uncommon)
    #
    epochs = o.train.epochs
    if train:
        res = res.train
    else:
        res = res.val
    x = res.epoch
    #
    (y_m, y_v) = smooth(x, res.entropy)
    
    (xs, y_m, y_v) = subsample(x, y_m, y_v)
    
    ci = y_v ** 0.5 * 3
    
    errorbar_fency(xs, y_m, ci, fmt='-' + marker, label=model_name, color=color, show_conf=show_conf)
    plt.legend(loc=1)
    ax = plt.gca()
    plt.xlabel(" epochs ")
    if not show_legend_entropy:
        plt.gca().get_legend().remove()


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def train_lr(dir_name, o, common, uncommon, train=True, color=None, model_name=None, marker='', **kwargs):
    res_name = dir_name + '/hist.pkl'
    try:
        res = pickle.load(open(res_name, "rb"))
    except FileNotFoundError:
        return
    # color, model_name = get_color_method(o, uncommon)
    #
    res = res.precond
    if res is not None:
        fig = plt.gcf()
        n = len(axes)
        fig.set_figheight(figsize[1] * n)
        fig.set_figwidth(figsize[0] * 3)
        if n == 0:
            # f, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            for i in range(n):
                axes[i].change_geometry(n + 1, 1, i + 1)
            ax = fig.add_subplot(n + 1, 1, n + 1)
        axes.append(ax)
        legend = res.legend.lr
        N = len(legend)
        # vals = np.log(res.lr.reshape([-1, N]))
        vals = res.lr.reshape([-1, N])
        epochs = vals.shape[0]
        x = np.arange(epochs)
        im = ax.imshow(vals.transpose(), extent=[0, 30, 0, N], origin='lower', norm=matplotlib.colors.LogNorm())
        plt.yticks(np.arange(0.5, N + 0.5, 1).tolist(), legend)
        i = np.arange(0, 11) * 3
        plt.xticks(i, (np.ceil(((epochs / i.max()) * i)).astype(int) // 10) * 10)
        ax.grid(False)
        ax.set_title(model_name)
        colorbar(im)

def test_acc(dir_name, o, common, uncommon, color=None, model_name=None, marker='', **kwargs):
    # flat = odict(uncommon.flatten())
    # if 'train.lr' in flat:
    #     flat['train.lr'] = format_digits(flat['train.lr'], 2)  # string
    # model_name = ', '.join(flat.strings())
    #print('model_name=', model_name)
    # surpress some warnings
    res_name = dir_name + '/hist.pkl'
    try:
        res = pickle.load(open(res_name, "rb"))
    except FileNotFoundError:
        return
    # color by method
    # of = odict(o.flatten())
    # color, model_name = get_color_method(o, uncommon)
    #
    epochs = o.train.epochs
    if max(res.train.epoch) < 200:
        print('Skipping not fully trained method')
        return
    
    np.seterr(over='raise')
    try:
        net = load_model_from_state(o)
    except BaseException as e:
        print("Cannot load model", e)
        return
    # color by method
    # of = odict(o.flatten())
    # color = get_color(of)
    # color, model_name = get_color_method(o, uncommon)
    #
    # data = MNIST_dataset(o.file.dataset_path)
    def compute():
        data = new_dataset(o)
        test = Test(o, net, data)
        method = models.method_class("sample")()
        acc = test.test_accuracy_samples(method=method).cpu().numpy()
        return acc

    acc = cache_test(o, 'test_acc', compute)
    acc = acc[0:32]
    x = np.arange(len(acc)) + 1
    #
    plt.plot(x, acc * 100, '-', label=model_name, color=color)
    #
    if o.train.method == 'standard':
        def compute_tanh():
            data = new_dataset(o)
            test = Test(o, net, data)
            method = models.method_class("standard")(odict(activation='tanh'))
            acc = test.test_accuracy_samples(method=method, n_samples=1).cpu().numpy()
            return acc
        acc_tanh = cache_test(o, 'test_acc_tanh', compute_tanh)
        plt.plot(1, acc_tanh * 100, '-o', label='Sigmoid NN', color=color)
    #
    plt.legend(loc=4)
    ax = plt.gca()
    plt.xlabel("samples")
    ym = acc[0]
    plt.ylim(ym, 100)


def val_loss(dir_name, o, common, uncommon, **kwargs):
    return train_loss(dir_name, o, common, uncommon, train=False, **kwargs)


def val_acc(dir_name, o, common, uncommon, **kwargs):
    return train_acc(dir_name, o, common, uncommon, train=False, **kwargs)


def val_entropy(dir_name, o, common, uncommon, **kwargs):
    return train_entropy(dir_name, o, common, uncommon, train=False, **kwargs)


def draw_all(root_dir):
    # collect_results(root_dir, (train_loss, val_loss, train_acc, val_acc, train_entropy, val_entropy, train_lr))
    collect_results(root_dir, (train_loss, val_loss, train_acc, val_acc, train_entropy, val_entropy))


if __name__ == "__main__":
    # root_dir = '../runs/MNIST-2L-n100-lr0.03/'
    # root_dir = '../runs/MNIST-2L-n100-lrO/'
    # root_dir = '../runs/MNIST-3L-n100-lrO/'
    # root_dir = '../runs/MNIST-2L-n100-lrO-Affine1/'
    # root_dir = '../runs/MNIST-2L-n200-lrO-Affine1/'
    # root_dir = '../runs/MNIST-Distil-2L-n100-lrO/'
    # root_dir = '../runs/MNIST-Distil-2L-n100-lrO-long1/'
    # root_dir = '../runs/MNIST-Distil-soft-2L-n100/'
    # root_dir = '../runs/CIFAR/Distil-2L-n200-lrO/'
    # root_dir = '../runs/CIFAR/CIFAR_S1/Distil/'
    # root_dir = '../runs/CIFAR/fc-2-200-NLL1-N1/'
    # root_dir = '../runs/CIFAR/fc-2-200-Affine-N/'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-Distil-logp-Affine-N'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-Distil-logp-Affine'
    # root_dir = '../runs/CIFAR/CIFAR_S1F-2-100-NLL-Affine'
    # root_dir = '../runs/CIFAR/CIFAR_S1F-2-100-ACC-Affine'
    # root_dir = '../runs/CIFAR/fc-2-200-Affine/'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-NLL-Affine/'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-Distil-logp-Affine-N'
    # root_dir = '../runs/CIFAR/fc-2-200-temp/'
    # root_dir = '../runs/CIFAR/new/'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-NLL-Affine-Init/'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-NLL-Affine-Auto-lr/'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-NLL-Affine-Auto-Adam/'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-NLL-FlipCrop-Auto-Adam/'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-NLL-Affine'
    # root_dir = '../runs/CIFAR/CIFAR_S1-2-100-NLL-FlipCrop/'
    root_dir = '../runs/CIFAR/CIFAR_S1-2-100-NLL/'
    scriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
    cdw = os.getcwd()
    
    if len(sys.argv) > 1:  # have command line arguments
        root_dir = sys.argv[1]

    draw_all(root_dir)
    collect_results(root_dir, (test_acc,))