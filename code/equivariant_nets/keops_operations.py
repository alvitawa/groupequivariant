import torch
import torch.nn.functional as F
from functools import reduce
from .python_helpers import DotDict, val_map
from .pytorch_helpers import arange_like
from .pytorch_helpers_3d import align_elements, gather_if_required
from pykeops.torch import Vi, Vj, Pm, LazyTensor
from pykeops.torch.cluster import from_matrix


# constants
PI = LazyTensor(-1).acos()


# basic helper functions
def cat(*args):
    as_lazy_tensor = (LazyTensor(a) if isinstance(a, int) else a for a in args)
    return reduce(lambda a, b: a.concat(b), as_lazy_tensor)


def rmat(angle, axes=cat(0, 1)):
    if angle._shape[-1] > 1: angle = cat(1, 1).tensorprod(angle)  # repeat twice
    return ((angle - axes.concat(axes - 1)) * (PI / 2)).cos()


def diag(*args):
    row = lambda i: cat(*[args[i] if i == j else 0 for j in range(len(args))])
    return cat(*[row(i) for i in range(len(args))])


def sign_nozero(v):
    return v.step() * 2 - 1


def angle(vec, plane, rot=None):
    return plane.matvecmult(vec).normalize()[0].clamp(-1, 1).acos() * sign_nozero(vec | plane[3:]) * (2 / PI)


def tangent(normal, plane):
    return angle(normal, plane) - 1


# matrix multiplication
def keops_matmul(a, b):
    (ai, aj), ad = a
    (bi, bj), bd = b
    return (ai, bj), ad.keops_tensordot(bd, (ai, aj), (bi, bj), (1,), (0,))


# chain matrix multiplication
cmm_fwd = lambda matrices: reduce(keops_matmul, matrices)[1]
cmm_bwd = lambda matrices: reduce(lambda a, b: keops_matmul(b, a), matrices[::-1])[1]


# build interaction ranges from objects in batch
def build_batch_range(s):
    return torch.cumsum(torch.stack([F.pad(s, (1, 0))[:-1], s], -1), 0).int()


def build_batch_sparsity(sa, sb):
    ra, rb = map(build_batch_range, [sa, sb])
    cm = torch.ones(len(sb), device=sb.device).diag().bool()
    return ra, rb, cm


# point convolution operation
def point_conv_keops(x, a, b, f, profile):
    # preprocess inputs
    a_, b_ = [{k: x[k] for k in {'b', 'p', 'z', 'x', 'xy', 'k', 'j'}} for x in [a, b]]
    order = ['adw_k', 'abd_p', 'bw_k', 'w_f', 'dw_c', 'ad_j']
    a_, b_, f_, x_ = align_elements([a_, b_, f, x], 'abwd', [order], skip={'b_j'}, end={'xyz'})
    if x_.shape[-1] == 1: x_ = F.pad(x_, (0, 1))
    x = gather_if_required(x_, a_.k, f_.ak, axis=order.index('adw_k'), flip_j=a_.j)
    f = DotDict(**f)
    f.pop('bk')

    # execute point conv
    ranges = build_batch_sparsity(a.obj_size, b.obj_size)
    result = execute_stacked(x.flatten(2), a, b, f, ranges, profile)
    result = result.reshape(*result.shape[:2], x_.shape[-2], len(b.k), len(f.n), 2)
    if a.pool_idx is not b.pool_idx: x = x[:, b.pool_idx]
    result = torch.cat([(x * (f_.ak == b_.k)).moveaxis(4, 2), result], 4)
    return result.rename('adw_k', 'b_p', 'adw_c', 'bw_k', 'w_f', 'xw_j')


def execute_stacked(x, a, b, f, ranges, profile):
    # rebuild ranges for stacked execution
    ranges_a = (ranges[0] + arange_like(x, dtype=torch.int32)[..., None, None] * a.p.shape[0]).flatten(0, 1)
    ranges_b = (ranges[1] + arange_like(x, dtype=torch.int32)[..., None, None] * b.p.shape[0]).flatten(0, 1)
    rs0, rs1 = ranges[2].shape
    ran_keep = torch.zeros(rs0*len(x), rs1*len(x), device=x.device, dtype=torch.bool)
    for i in range(x.shape[0]):
        ran_keep[i*rs0: (i+1)*rs0, i*rs1: (i+1)*rs1] = ranges[2]
    ranges = (ranges_a, ranges_b, ran_keep)

    # run stacked
    ak = f.pop('ak').float().rename(None).repeat_interleave(x.shape[1])[..., None]
    ks = ['p', 'z', 'x', 'pzxy']
    a, b = [DotDict(v, **{k: v[k].rename(None).repeat(x.shape[0], *[1]*(v[k].ndim - 1)) for k in ks}) for v in [a, b]]
    result = process(ak, x.flatten(0, 1), a, b, f, ranges=from_matrix(*ranges), profile=profile)
    return result.reshape(x.shape[0], -1, result.shape[1])


def process(ak, x, a_, b_, f_, **kwargs):
    a, b = [DotDict(
        p=op(v_.p.rename(None).contiguous()),
        z=op(v_.z.rename(None).contiguous()),
        xy=op(v_.pzxy[:, 2:].rename(None).flatten(1).contiguous()),
    ) for (v_, op) in [(a_, Vi), (b_, Vj)]]
    bk = b_.k.rename(None).repeat_interleave(len(f_.n))
    f = DotDict({k: v.rename(None).repeat(len(b_.k)) for k, v in f_.items()})
    f['k'] = bk.float()
    a['k'] = [Pm, Pm, Vi][ak.ndim](ak.float())
    return keops_op_out_split(x, a, b, f, **kwargs)


def keops_op_out_split(x, a, b, f, **kwargs):
    o = 256 // x.shape[-1]
    s = (f.k.shape[-1] - 1) // o + 1
    o = (f.k.shape[-1] - 1) // s + 1
    x = Vi(x)
    shp = (b.p._shape[-2], x._shape[-1] // 2, -1)
    res = [interact(x, a, b, DotDict(val_map(lambda v: v[i*o:(i+1)*o], f)), **kwargs).reshape(shp) for i in range(s)]
    return torch.cat(res, -1).reshape(b.p._shape[-2], -1)


def interact(x, a, b, f, ranges, profile):
    j_out = Pm(torch.tensor([0, 1], device=f.k.device).float().repeat(len(f.k)))
    f = DotDict(val_map(lambda v: Pm(v.repeat_interleave(2)), f))
    i = x._shape[-1] // 2
    o = f.k._shape[-1]
    rp = a.p - b.p
    skip_same_point = 1 - (-rp.sqnorm2()).step()
    filter_reaction = cat(1, 1).tensorprod(profile(f=f, r=b.xy.matvecmult(rp).norm(-1), z=rp | b.z))
    reduction_op = (cmm_fwd if i < o else cmm_bwd)([
        # shape  lazy_tensors
        ((i, 2), x),  # input data
        ((2, 2), cmm_fwd([
            ((2, 2), rmat(tangent(-b.z, a.xy) * (-a.k))),      # rot to tangent frame
            ((2, 2), diag(1, (a.z | b.z)) * skip_same_point),  # project imaginary part
        ])),
        ((2, o), filter_reaction * rmat(tangent(a.z, b.xy) * a.k + angle(rp, b.xy) * (f.k - a.k), axes=j_out)),
    ])
    reduction_op.ranges = ranges  # set ranges before reduction
    return reduction_op.sum(0)  # sum over 'i'
