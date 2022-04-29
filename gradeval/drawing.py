import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
from utils import *
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MaxNLocator


def save_pdf(file_name):
    force_path(file_name)
    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)


def errorbar_fency(x, y, yerr=None, yl=None, yu=None, fmt=None, color=None, ylog=False, **kwargs):
    if yerr is not None:
        if yl is None:
            yl = y - yerr
            if ylog:
                yl = np.maximum(yl, 0.01 * y)
        if yu is None:
            yu = y + yerr
        yl = np.minimum(yl, y) - 1e-5 * np.abs(y)
        yu = np.maximum(yu, y) + 1e-5 * np.abs(y)
    kwargs['markevery'] = int(math.ceil(len(x) / 10))
    # kwargs['markevery'] = 10
    # plt.fill_between(x, yl, yu, alpha=0.1, facecolor=color)
    if yl is not None and yu is not None:
        plt.fill_between(x, yl, yu, alpha=0.2, facecolor=color)
    plt.plot(x, y, fmt, color=color, **kwargs)
