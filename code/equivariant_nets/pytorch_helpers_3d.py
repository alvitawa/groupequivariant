import numpy as np
import torch
from torch import nn
from itertools import chain
from .python_helpers import DotDict, check_single_value, pairwise
from .pytorch_helpers import fix_stride


# alignment helpers
def _expand_name(e, name, avail):
    ep, ex = ['', *e.split('_', 1)][-2:]
    ep = '' if name == '*' else ep if name == ':' else ep + name
    matches = [a for a in avail for ap, ax in [['', *a.split('_', 1)][-2:]] if ex == ax and set(ep).issubset(ap)]
    return check_single_value(matches or [e])


def _align_element(element, n, order, skip, end):
    if isinstance(element, (dict, nn.ParameterDict)):
        aligned = {k: _align_element(v, n, order, skip, end) for k, v in element.items() if f'{n}_{k}' not in skip}
        return DotDict(aligned)
    if isinstance(element, list):
        return [align_element(v, n, order) for v in element]
    element = element.rename(*[_expand_name(x, n, order) for x in element.names])
    surplus_axes = [x for x in element.names if x not in order]
    assert set(surplus_axes).issubset(end)
    return element.align_to(*order, *surplus_axes).rename(None)


def align_elements(elements, names, order, skip=set(), end=set()):
    return [_align_element(e, n, [*chain(*order)], skip, end) for e, n in zip(elements, names)]


def aligned_matmul(a, b, order):
    cuts = np.cumsum([0, *map(len, order)])
    rs_a, rs_b = [[np.prod(x.shape[s: e], dtype=int) for s, e in pairwise(cuts)] for x in [a, b]]
    #print(a.shape, b.shape, '->', a.reshape(rs_a).squeeze(-1).shape, b.reshape(rs_b).squeeze(-3).shape)
    result = torch.bmm(a.reshape(rs_a).squeeze(-1), b.reshape(rs_b).squeeze(-3))
    return result.reshape(*a.shape[:cuts[-3]], *b.shape[cuts[-2]:]).rename(*chain(*order[:-2]), *order[-1])


def gather_if_required(x, src, tgt, axis, flip_j=None):
    if not src.equal(tgt):
        flip = (src >= 0).all() and (tgt < 0).any()
        idx = torch.tensor([*map(list(src.flatten()).index, tgt.abs() if flip else tgt)], device=tgt.device)
        x = x.index_select(axis, idx)
        if flip:
            x *= (-1)**(flip_j * tgt.clamp(-1, 0))
    return x


def ax_to_j(x, axis=-1):
    return torch.view_as_complex(fix_stride(x.unsqueeze(-1).transpose(-1, axis - (axis < 0))))


def j_to_ax(x, axis=-1):
    return torch.stack([x.real, x.imag], axis=-1).transpose(-1, axis - (axis < 0)).squeeze(-1)


# make symmetric to simultaneously conjugating axis 'j' and flipping all axes whose name ends on '_k'
def symmetrize_complex(w):
    f = w.transpose('j', w.names[-1])
    f_names = f.names
    f = f.rename(None).flip([i for i, n in enumerate(f_names) if n.endswith('_k')])  # flip k-axes
    f = torch.view_as_complex(f).conj()  # conjugate
    f = torch.view_as_real(f).rename(*f_names).transpose('j', w.names[-1])
    return (w + f) / 2


# exactly calculate 1j**(v) for an integer v (direct calculation results in inaccuracies)
def j_pow(v):
    v = torch.as_tensor(v, device=v.device)
    assert not v.dtype.is_floating_point
    return torch.view_as_complex(torch.view_as_real(1j**v).round())


# applying trainable weights
def apply_weights_bmm(x, weights, filter_desc, data_b_desc, align=None):
    b_ = {k: data_b_desc[k] for k in {'k', 'j'}}
    order = [['bw_k'], ['b_p', 'b_j'], ['adw_c', 'adw_k', 'w_f', 'xw_j'], ['bw_c']]
    x, b, f, w = align_elements([x, b_, filter_desc, weights], ':bww', order, end={'xyz'})  # align inputs
    x = ax_to_j(x, [*chain(*order)].index('xw_j'))  # load stashed complex axis from x_j
    if b.j.numel() > 1 or b.j.item() != 0: x = x * j_pow(-b.j).detach()  # expand final complex output to axis b_j
    x = j_to_ax(x, [*chain(*order)].index('xw_j'))  # project on weights by unpacking complex axis to w_j
    w = gather_if_required(w, f.bk, b.k, [*chain(*order)].index('bw_k'))  # gather if output k's and weight k's differ
    x = aligned_matmul(x, w, order)  # execute by matrix multiplication
    return x if align is None else align_elements([x], '*', [align])[0]  # return aligned result
