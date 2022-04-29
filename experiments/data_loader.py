from torchvision import datasets, transforms
import torch.utils.data
import torch.utils
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from PIL import Image
from .options import odict


class LogitsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, logits_file, scaling=1.0):
        self.dataset = dataset
        self.logp = torch.tensor(np.load(logits_file).astype(np.float32))
        self.logp = F.log_softmax(self.logp * scaling, dim=-1)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        target_logp = self.logp[index]
        return img, target, target_logp

    def __len__(self):
        return len(self.dataset)


class RandomAffineTensor(transforms.RandomAffine):
    def __call__(self, sample):
        t_sample = torch.empty_like(sample)
        for idx in range(sample.size(0)):
            t_sample[idx] = transforms.ToTensor()(super().__call__(transforms.ToPILImage()(sample[idx])))
        return t_sample


class FLIPCropTensor(object):
    def __call__(self, sample):
        t_sample = torch.empty_like(sample)
        T = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor()])
        for idx in range(sample.size(0)):
            t_sample[idx] = T(sample[idx])
        return t_sample

class NormalizeTensor(transforms.Normalize):
    """ Make Normalize applicable to batch of Tensors
    """
    def __call__(self, tensor):
        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view([1, -1, 1, 1])
        std = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device).view([1, -1, 1, 1])
        return (tensor - mean) / std


class Identity(object):
    """ Make Normalize applicable to batch of Tensors
    """
    
    def __call__(self, x):
        return x


def MNIST_dataset(path):

    load_transform = transforms.ToTensor()
    train_transform = NormalizeTensor((0.1307,), (0.3081,))
    test_transform = NormalizeTensor((0.1307,), (0.3081,))

    r = odict()
    r.train_transform = train_transform
    r.test_transform = test_transform

    r.train_dataset = datasets.MNIST(root=path, train=True, download=True, transform=load_transform)
    r.test_dataset = datasets.MNIST(root=path, train=False, download=True, transform=load_transform)

    r.input_size = 784
    r.n_classes = 10
    r.input_shape = (1, 28, 28)
    
    return r


def CIFAR10_dataset(path, aug=None):
    r = odict()
    load_transform = transforms.ToTensor()
    # this works on Tensors, just remember it if we ever want to do it, do it in the very end, not at the loaders
    # norm_transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                       (0.2023, 0.1994, 0.2010))

    r.train_transform = Identity()
    r.test_transform = Identity()
    aug_transform = Identity()
    # load_transform = Identity()
    #
    # if aug == 'CropFlip':
    #     aug_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        # elif aug == 'Affine':
    #     aug_transform = transforms.RandomAffine(10, translate=(0.05, 0.05), scale=None, shear=None, resample=Image.BILINEAR, fillcolor=0)
    # else:
    #     aug_transform = Identity()
    #
    # r.train_transform = transforms.Compose([
    #     # aug_transform,
    #     # transforms.ToTensor(),
    #     norm_transform,
    # ])
    # r.test_transform = transforms.Compose([
    #     # transforms.ToTensor(),
    #     norm_transform,
    # ])

    r.train_dataset = datasets.CIFAR10(root=path, train=True, download=False, transform=load_transform)
    r.test_dataset = datasets.CIFAR10(root=path, train=False, download=False, transform=load_transform)
    
    r.input_size = None
    r.n_classes = 10
    r.input_shape = (3, 32, 32)
    
    return r


def CIFAR10_dataset_logits(path, scaling=1.0, aug=None):
    r = CIFAR10_dataset(path, aug=aug)
    
    CIFAR10_TRAIN_LOGITS = f'{path}/CIFAR/enspreds/logits_CIFAR10-VGG16BN-deepens-100-0_train'
    CIFAR10_TEST_LOGITS  = f'{path}/CIFAR/enspreds/logits_CIFAR10-VGG16BN-deepens-100-0_test'

    r.train_dataset = LogitsDataset(r.train_dataset, CIFAR10_TRAIN_LOGITS, scaling)
    r.test_dataset = LogitsDataset(r.test_dataset,  CIFAR10_TEST_LOGITS, scaling)
    
    return r


class SingleProcessDataLoaderIterwIndex(torch.utils.data.dataloader._SingleProcessDataLoaderIter):
    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __next__(self):
        index = self._next_index()  # may raise StopIteration
        data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
        if self.transform is not None:
            data[0] = self.transform(data[0])
        if self.pin_memory:
            data = torch.utils.data._utils.pin_memory.pin_memory(data)
        return (index, *data)


class DataLoaderIndex(torch.utils.data.DataLoader):
    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        
    def __iter__(self):
        data_iter = SingleProcessDataLoaderIterwIndex(self, transform=self.transform)
        return data_iter


class Preproc:
    def preproc(self, data_idx, data, *targets):
        return (data_idx, data, *targets)
    
    def __call__(self, generator):
        for batch_idx, (data_idx, data, *targets) in enumerate(generator):
            yield self.preproc(data_idx, data, *targets)


class PreprocToDevice(Preproc):
    def __init__(self, device):
        self.device = device
        
    def preproc(self, data_idx, data, *targets):
        data = data.to(self.device)
        targets = [t.to(self.device) for t in targets]
        return (data_idx, data, *targets)


class PreprocTransformData(Preproc):
    def __init__(self, transform):
        self.transform = transform

    def preproc(self, data_idx, data, *targets):
        data = self.transform(data)
        return (data_idx, data, *targets)


class PreprocDistilNet(Preproc):
    def __init__(self, net):
        self.net = net
    
    def preproc(self, data_idx, data, *targets):
        with torch.no_grad():
            target_logp = self.net(data)
        return (data_idx, data, *targets, target_logp)


class Preproc_logp(Preproc):
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def preproc(self, data_idx, data, target_idx, *targets):
        p = F.one_hot(target_idx, num_classes=self.num_classes).to(device=data.device, dtype=float)
        p = p + 0.05 / self.num_classes
        p = p / p.sum(dim=-1, keepdim=True)
        target_logp = p.log()
        return (data_idx, data, target_idx, *targets, target_logp)


def get_train_valid_loader(dataset,
                           train_batch_size,
                           val_batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=False, train_transform=None, val_transform=None):
    assert ((valid_size >= 0) and (valid_size <= 1)), "[!] valid_size should be in the range [0, 1]."
    
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.ceil(valid_size * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # print(valid_idx)
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoaderIndex(dataset, batch_size=train_batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory,
                                   transform=train_transform)
    train_loader.idx = train_idx
    valid_loader = DataLoaderIndex(dataset, batch_size=val_batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory,
                                   transform=val_transform)
    valid_loader.idx = valid_idx
    
    return train_loader, valid_loader


