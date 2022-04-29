from gradeval.options import *
from gradeval.drawing import *
from experiments.options import odict

all_methods = [
    odict(method='AP1'),
    odict(method='score'),
    odict(method='concrete', t=1),
    odict(method='ST'),
    odict(method='SAH'),
    # odict(method='AP2'),
    odict(method='ARM'),
    odict(method='LocalReparam'),
    odict(method='enumerate'),
    odict(method='HardST'),
    odict(method='concrete', t=0.1),
    odict(method='SA', t=0),
    odict(method='SA', t=100),
    odict(method='SAH1'),
    odict(method='SA', t=1),
    ]

prop_cycle = plt.rcParams['axes.prop_cycle']
ccolors = prop_cycle.by_key()['color']
markers = 'oxd*+v^s<'
for (i, m) in enumerate(all_methods):
    color_s = ccolors[i % len(ccolors)]
    m.color = color_s
    m.marker = markers[i % len(markers)]


# def select_methods(methods):
#     r = []
#     for (i, m) in enumerate(all_methods):
#         try:
#             x = next(x for x in methods if x.method == m.method and x.t == m.t)
#             r.append(m)
#         except StopIteration:
#             pass
#     return r

def select_methods(methods):
    r = []
    for (i, m) in enumerate(methods):
        try:
            x = next(x for x in all_methods if x.method == m.method and x.t == m.t)
            r.append(x)
        except StopIteration:
            pass
    return r
