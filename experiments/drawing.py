import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
from gradeval.utils import *
import matplotlib.pyplot as plt
import matplotlib
import pickle
from matplotlib.ticker import MaxNLocator


prop_cycle = plt.rcParams['axes.prop_cycle']
ccolors = prop_cycle.by_key()['color']

def save_pdf(file_name):
    force_path(file_name)
    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)


def errorbar_fency(x, y, yerr=None, yl=None, yu=None, fmt=None, color=None, show_conf=True, **kwargs):
    if yl is None:
        yl = y - yerr
        yl = np.maximum(yl, 0.1 * y)
    if yu is None:
        yu = y + yerr
    yl = np.minimum(yl, y)  # - 1e-5*np.abs(y)
    yu = np.maximum(yu, y)  # + 1e-5*np.abs(y)
    kwargs['markevery'] = int(math.ceil(len(x) / 10))
    # kwargs['markevery'] = 10
    # plt.fill_between(x, yl, yu, alpha=0.1, facecolor=color)
    if show_conf:
        plt.fill_between(x, yl, yu, alpha=0.2, facecolor=color)
    plt.plot(x, y, fmt, color=color, **kwargs)
