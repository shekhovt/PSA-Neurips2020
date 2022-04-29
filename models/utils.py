import math
from collections.abc import Iterable
import types
import torch

all_checks = True


def sign_bernoulli(x):
    u = torch.empty_like(x).uniform_()
    return (x.detach() - u).sign_()


def to_tuple(z):
    if isinstance(z, tuple):
        return z
    elif isinstance(z, types.GeneratorType):
        return tuple(z)
    else:
        return (z,)


def untuple(z):
    if isinstance(z, tuple) and len(z) == 1:
        return z[0]
    else:
        return z


class NumericalProblem(BaseException):
    pass

def check_var(x):
    """
    Check variance values are elligible
    """
    # print(torch.min(x.data))
    if all_checks:
        if not (x.data >= 0).all():
            raise NumericalProblem('variance is negative or nan')


def check_real(x):
    """
       Check for NaN / Inf
    """
    if all_checks:
        if not (x == x).all():
            raise NumericalProblem('NaN encountered')
        if not (x.abs() < float('inf')).all():
            raise NumericalProblem('+-Inf encountered')


def nan_to_zero(x):
    if math.isnan(x):
        x = 0.0
    return x
