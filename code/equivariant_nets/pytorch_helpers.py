import torch
from torch import nn
import torch.nn.functional as F


# shortcut to make a non-trainable parameter
def NonTrainable(data):
    return nn.Parameter(torch.as_tensor(data), requires_grad=False)


# reset running stats of batch norm for a model
def reset_running_stats(net):
    return [m.reset_running_stats() for m in net.modules() if hasattr(m, 'reset_running_stats') and m.momentum is None]


# tensor functions
def arange_like(tensor, repeat=None, **kwargs):
    arange = torch.arange(len(tensor), device=tensor.device, **kwargs)
    return arange if repeat is None else arange.repeat_interleave(tensor if repeat is True else repeat)


def inverse_order(order, target=None):
    # this scatter op should be faster than doing a second argsort on the indices
    if target is None:
        target = torch.empty_like(order)
    target[order] = arange_like(order)
    return target


def fix_stride(x):  # set stride to 0 for all axes with shape 1
    # original call: 'x.as_strided(x.shape, tuple((np.array(x.shape) != 1) * x.stride()))'
    return x.reshape_as(x)  # but this works too and is much simpler


# geometry functions
def orthonormal_system(z):
    h = F.one_hot(z.abs().min(dim=-1)[1], 3).type(z.dtype)  # choose a unit vector != z
    y = F.normalize(torch.cross(z, h, dim=-1), dim=-1)
    x = F.normalize(torch.cross(y, z, dim=-1), dim=-1)
    return x, y
