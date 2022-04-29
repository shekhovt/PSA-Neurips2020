from .construct import *
import pickle
from models.architectures import *


def load_model_from_state(o):
    net = new_model(o)
    if o.train.norm == 'AP2':
        reparam_with_norm(net)
    net_state = pickle.load(open(o.file.log_dir + 'model_state.pkl', "rb"))
    net.load_state_dict(net_state)
    for m in net.modules():
        if isinstance(m, LinearWithNorm):
            m.initialized = True
    return net
