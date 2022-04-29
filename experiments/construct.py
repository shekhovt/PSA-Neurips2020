#import models
from .data_loader import *
from models.losses import *
from models.methods import *
from models.architectures import *
from PIL import Image


def new_model(o):
    method = method_class(o.train.method)(o.methods)
    model_cls = model_class(o.create.network)
    net = model_cls(method, o.create)
    if o.file.cuda:
        net = net.cuda()
    return net


def new_dataset(o):
    path = o.file.dataset.path
    cls = globals()[o.file.dataset.class_name]
    data = cls(path, **o.file.dataset.construct_args)
    #
    return data

def new_loss(o):
    cls = globals()[o.train.loss]
    loss = cls()
    return loss


def new_preproc(o):
    if o.train.preproc == 'Affine':
        transform = RandomAffineTensor(10, translate=(0.05, 0.05), scale=None, shear=None, resample=Image.BILINEAR, fillcolor=0)
        p = PreprocTransformData(transform)
    if o.train.preproc == 'FlipCrop':
        # transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor()])
        transform = FLIPCropTensor()
        p = PreprocTransformData(transform)
    elif o.train.preproc == 'logp':
        p = Preproc_logp(num_classes=10)
    else:
        p = Preproc()
    return p
