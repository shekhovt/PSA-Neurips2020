# import os
# import numbers
from collections import OrderedDict
import inspect
import copy


# class odict(OrderedDict):
#     def __getattr__(self, name):
#         if name not in self:
#             return None
#         else:
#             return self[name]
#
#     def __setattr__(self, name, value):
#         self[name] = value
#
#     def __delattr__(self, name):
#         del self[name]
#

class Uninitialized:
    pass


def first(s):
    '''Return the first element from an ordered collection
       or an arbitrary element from an unordered collection.
       Raise StopIteration if the collection is empty.
    '''
    return next(iter(s))


class multioption:
    """ represent possible choices for options and hieararchy of suboptions for specific option choices
        validate if a given option is a member of the family
        iterate over the family
    """
    
    def __new__(cls, *args, **kwargs):
        obj = super(multioption, cls).__new__(cls)
        super(multioption, obj).__setattr__('suboptions', OrderedDict())  #
        super(multioption, obj).__setattr__('choices', OrderedDict())
        super(multioption, obj).__setattr__('index', None)
        super(multioption, obj).__setattr__('c', None)
        super(multioption, obj).__setattr__('locked', False)
        super(multioption, obj).__setattr__('super', None)
        return obj
    
    def __init__(self, *args):
        super().__init__()
        self.suboptions = OrderedDict()  #
        self.choices = OrderedDict()
        self.index = None
        self.c = None
        if len(args) > 0:
            for a in args:
                self.choices[a] = multioption()
    
    def __getattr__(self, name):
        # by the precedence, attributs in the __dict__ and properties are returned first
        try:
            return self[name]  # redirects to __getitem__
        except KeyError:
            raise AttributeError
    
    def __getitem__(self, item):
        assert ('suboptions' in self.__dict__)
        assert ('choices' in self.__dict__)
        if item in self.suboptions:
            return self.suboptions[item]
        elif item in self.choices:
            return self.choices[item]
        else:
            raise KeyError('No suboption {}'.format(item))
    
    def __setattr__(self, name, value):
        if name == 'suboptions' or name == 'choices' or name == 'index' or name == 'c':
            object.__setattr__(self, name, value)
        else:
            if isinstance(value, list):
                self.suboptions[name] = multioption(*value)
            elif isinstance(value, multioption):
                self.suboptions[name] = value
            else:
                self.suboptions[name] = multioption(value)
    
    def __delattr__(self, name):
        if name in self.suboptions:
            del self.suboptions[name]
        elif name in self.choices:
            del self.choices[name]
    
    def lock(self, locked=True):
        self.__dict__.locked = locked
    
    def is_empty(self):
        return len(self.suboptions) == 0 and len(self.choices) == 0
    
    def __str__(self, name=Uninitialized, tab=0):
        try:
            tmpstr = ''
            if name is not Uninitialized:
                tmpstr = name
                tab += 1
            else:
                tmpstr = self.__class__.__name__ + ':'
            if len(self.choices) > 0:
                tmpstr += '= '
            if len(self.choices) == 1:
                tmpstr += str(list(self.choices.keys())[0])
            elif len(self.choices) > 1:
                tmpstr += str(list(self.choices.keys()))
            tmpstr += '\n'
            # list choices
            for i, (o, v) in enumerate(self.choices.items()):
                if not v.is_empty():
                    tmpstr += '  ' * tab
                    tmpstr += v.__str__(name=':' + str(o), tab=tab)
            # list suboptions
            for i, (o, v) in enumerate(self.suboptions.items()):
                if not v.is_empty():
                    # tmpstr += '│ ' * tab
                    tmpstr += '  ' * tab
                    # if i < len(self)-1:
                    #     tmpstr += '├─'
                    # else:
                    #     tmpstr += '└─'
                    tmpstr += v.__str__(name=str(o), tab=tab)
            return tmpstr
        except:
            return super().__str__()
    
    def __repr__(self, name=Uninitialized, tab=0):
        try:
            tmpstr = ''
            if name is not Uninitialized:
                tmpstr = name
                tab += 1
            else:
                tmpstr = self.__class__.__name__
            if len(self.choices) > 0:
                tmpstr += '='
            if len(self.choices) == 1:
                v = list(self.choices.keys())[0]
                tmpstr += repr(v)
            elif len(self.choices) > 1:
                tmpstr += repr(list(self.choices.keys()))
            t1 = ''
            if len(self.choices) == 1:
                t1 += first(self.choices.values()).__repr__(name='', tab=tab)
            elif len(self.choices) > 1:
                # list choices
                for i, (o, v) in enumerate(self.choices.items()):
                    if not v.is_empty():
                        t1 += v.__repr__(name=str(o), tab=tab)
                        if i < len(self.choices) - 1:
                            t1 += ', '
            # list suboptions
            for i, (o, v) in enumerate(self.suboptions.items()):
                if not v.is_empty():
                    t1 += v.__repr__(name=str(o), tab=tab)
                    if i < len(self.suboptions) - 1:
                        t1 += ', '
            if len(t1) > 0:
                if name != '':
                    tmpstr += '[{}]'.format(t1)
                else:
                    tmpstr += t1
            return tmpstr
        except:
            return super().__repr__()
    
    def index_reset(self):
        if len(self.choices) > 0:
            self.index = iter(self.choices.items())
            self.c = next(self.index)
        else:
            self.index = None
            self.c = None
        for o in self.choices.values():
            o.index_reset()
        for o in self.suboptions.values():
            o.index_reset()
    
    # def index_feasible(self):
    #     return self.index < len(self.choices)
    
    def index_inc(self):
        # increment suboptions in reversed order
        for k, o in reversed(self.suboptions.items()):
            try:
                o.index_inc()
                return True
            except StopIteration:  # o.index overflows
                o.index_reset()
        # increment inside the chosen option
        if self.c is not None:
            try:
                self.c[1].index_inc()
                return True
            except StopIteration:
                self.c[1].index_reset()
        # else increment self
        # this will retrive correctly the next element or throw StopIteration
        if self.index is not None:
            self.c = next(self.index)
        else:
            raise StopIteration
        return True
    
    # def index_retrive(self) -> 'multioption':
    #     if self.c is not None: # retrive suboptions
    #         so = self.c[1].index_retrive()
    #         r = multioption(self.c[0]) # option with current chosen key
    #         r += so # add suboptions from the choice
    #     else:
    #         r = multioption()
    #     # add suboptions
    #     for k,o in self.suboptions.items():
    #         so = o.index_retrive()
    #         r.suboptions[k] = so
    #     return r
    
    def index_retrive(self) -> 'multioption':
        if self.c is not None:  # retrive suboptions
            so = self.c[1].index_retrive()
            r = multioption(self.c[0])  # option with current chosen key
            r.choices[self.c[0]] = so  # current choice
        else:
            r = multioption()
        # add suboptions
        for k, o in self.suboptions.items():
            so = o.index_retrive()
            r.suboptions[k] = so
        r.__dict__['super'] = self
        return r
    
    def __iter__(self) -> 'multioption':
        self.index_reset()
        return self
    
    def __next__(self) -> 'multioption':
        if self.index is StopIteration:
            raise StopIteration
        o = self.index_retrive()
        try:
            self.index_inc()
        except StopIteration:
            self.index = StopIteration
        return o
    
    def default(self) -> 'multioption':
        return first(self)
    
    def update(self, other: 'multioption'):
        """overwrite / add other options """
        for k, v in other.choices.items():
            if k in self.choices:
                self.choices[k].update(v)
            else:
                self.choices[k] = v
        for k, v in other.suboptions.items():
            if k in self.suboptions:
                self.suboptions[k].update(v)
            else:
                self.suboptions[k] = v
    
    def __iadd__(self, other: 'multioption') -> 'multioption':
        self.update(other)
        return self
    
    def __add__(self, other: 'multioption') -> 'multioption':
        r = multioption()
        r += self
        r += other
        return r
    
    def __isub__(self, other: 'multioption') -> 'multioption':
        for k, v in other.choices.items():
            if k in self.choices:
                self.choices[k] -= v
                if self.choices[k].is_empty():
                    del self.choices[k]
            else:
                pass
        for k, v in other.suboptions.items():
            if k in self.suboptions:
                self.suboptions[k] -= v
                if self.suboptions[k].is_empty():
                    del self.suboptions[k]
            else:
                pass
        return self
    
    #
    
    def copy(self) -> 'multioption':
        r = multioption.__new__(multioption)
        r.__dict__['locked'] = self.locked
        r.__dict__['super'] = self.super
        r.choices = copy.deepcopy(self.choices)
        r.suboptions = copy.deepcopy(self.suboptions)
        return r
    
    def __sub__(self, other: 'multioption') -> 'multioption':
        r = self.copy()
        r -= other
        return r
    
    def includes(self, other: 'multioption') -> bool:
        return (other - self).is_empty()
    
    @property
    def value(self):
        if len(self.choices) == 1:
            return first(self.choices.keys())
        elif len(self.choices) == 0:
            raise ValueError('option does not have a value set')
        else:
            raise ValueError('option is multi-valued')
    
    def __bool__(self):
        """option can evaluate to True / False when value is Boolean"""
        return first(self.choices.keys())
    
    def __eq__(self, other):
        """ test of eqiality with other option object or with the option value"""
        if isinstance(other, multioption):
            return (self - other).is_empty() and (other - self).is_empty()
        elif isinstance(other, str):
            return str(self.value) == other
        else:
            return self.value == other
    
    def __ne__(self, other):
        """ test two option hierarchies are not equal """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result
    
    # def inclusion(self):
    #     pass
    
    # def check_in_list(self):
    #     # check feasibility
    #     legal = False
    #     for v in self.legal_values:
    #         if inspect.isclass(v) and isinstance(self.value, v):
    #             legal = True
    #             break
    #         elif self.value == v:
    #             legal = True
    #             break
    #     if not legal:
    #         raise ValueError('Option {} is not in the list of legal values {}'.format(self.value, self.legal_values))
    #     self.value = value


def test_multioption():
    o = multioption()
    #
    # 'file' options do not affect train or test results
    #
    o.file = []
    o.file.version = '1.0'
    o.file.dataset = ''
    o.file.dataset.path = ''
    o.file.model = ''
    o.file.model.path = ''
    o.file.num_workers = 0
    o.file.allow_cuda = True
    o.file.verbose = True
    #
    # training options
    #
    o.train = []
    o.train.seed = 0
    o.train.optimizer = ['SGD', 'Adam']
    o.train.optimizer.SGD.Momentum = 0.9
    o.train.optimizer.SGD.Nesterov = True
    o.train.weight_decay = 0
    o.train.batch_size = 32
    o.train.init_batch_size = 128
    #
    o.train.inferece = ['AP1', 'AP2', 'sample']
    o.train.inferece.AP2.softmax = ['simplified', 'full']
    # #
    o.train.reg = [None, 'VB']
    o.train.reg.VB.reg_c = 1.0
    o.train.reg.VB.ssb = True
    o.train.reg.VB.SConv = False
    o.train.weight_init = ['uniform', 'orthogonal']
    o.train.init = [None, 'BN', 'AP2']
    o.train.norm = [None, 'BN', 'AP2', 'WeightNorm']
    o.train.norm.BN.project = True
    o.train.norm.AP2.project = [True, False]
    o.train.noise_augment = 0.0
    o.train.input_var = 0.0
    o.train.lr = [None, float]
    o.train.lr[None].lr_estimate_epochs = 5
    o.train.dropout_noise = [None, 'bernoulli', 'normal']
    o.train.dropout_noise.bernoulli.var = 1 / 4
    o.train.dropout_noise.normal.var = 1 / 4
    o.train.lr_down = ['exp', 'sqrt', 'const']
    o.train.lr_down.exp.base = 0.99426
    # # test options
    o.test = []
    o.test.inference = ['AP1', 'AP2']
    o.test.inference.AP2.softmax = ['simplified', 'full']
    
    print(o)
    io = iter(o)
    a = next(io)
    print(repr(a))
    i = 1
    # loop over train options
    l = len(list(iter(o.train)))
    print('l={}'.format(l))
    assert (l == 8640)
    # loop over test options and merge with non-test options
    O = o.default()
    for r in o.test:
        i += 1
        O.test = r
        print(str(i) + ':' + repr(r))
        print(str(i) + ':' + repr(O))
        assert (o.includes(O))
        assert (O.super.includes(O))
    
    assert O.file.allow_cuda
    O.train.optimizer = 'SGD'
    assert O.train.optimizer == 'SGD'
    
    O.train.optimizer.SGD = o.train.optimizer.SGD
    
    print(repr(o.default()))
    
    # print(str(i) + ':' + repr(a))
    # while True:
    #     try:
    #         b = next(io)
    #         i += 1
    #         print(str(i) + ':' + repr(b))
    #         assert(a != b)
    #     except StopIteration:
    #         break
    #


# def test_option():
#     # test code
#     o = option()
#     # code version stamp
#     o.code = option()
#     o.code.version = '1.0'
#     # file options, do not affect train or test
#     o.file = option()
#     o.file.dataset = 'CIFAR-10'
#     o.file.dataset.path = '../../data/train/'
#     o.file.model = 'Spingenberg v. S3'
#     o.file.model.path = '../../train/CIFAR_S3/l1'
#
#     # train options
#     o.train = option()
#     o.train.optimizer = 'SGD'
#     o.train.optimizer.Nesterov = True
#     o.train.optimizer.Momentum = 0.9
#     o.train.batch_size = 32
#
#     # test options
#     o.test = option()
#     o.test.batch_size = 32
#     o.test.inference = 'AP1'
#
#     o1 = o.copy()
#     o1.train.optimizer.Nesterov = False
#     o1.train.optimizer.eps = 1e-8
#     o1.train.optimizer.step_rule = 'exp'
#     o1.train.optimizer.step_rule.base = 0.99
#
#     print(o)
#     print(repr(o))
#     print(repr(o - o1))
#     print(repr(o1 - o))
#
#     print(o.train.optimizer == 'SGD')
#
#     print(bool(o.train.optimizer.Nesterov))
#
#     print(o == o1)
#     print(o.test == o1.test)
#
#     o.update(o1)
#     print(repr(o))


if __name__ == "__main__":
    test_multioption()
