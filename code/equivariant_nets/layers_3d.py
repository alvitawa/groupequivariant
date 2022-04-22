import torch
import numpy as np
import torch.fft
from torch import nn
import torch.nn.functional as F
from .python_helpers import DotDict, pairwise, val_map
from .pytorch_helpers import NonTrainable, arange_like, orthonormal_system, inverse_order
from .pytorch_helpers_3d import symmetrize_complex, apply_weights_bmm
from .keops_operations import point_conv_keops


# vertical stack of radial filters
class FilterStack(nn.Module):
    def __init__(self, rp, rw, zp, zw, factor=1, obj_dim=2, cutoff_factor=2):
        super().__init__()
        f = DotDict()

        # broadcast shapes
        f.rp, f.rw = torch.broadcast_tensors(*[torch.as_tensor(v) for v in [rp, rw]])
        f.zp, f.zw = torch.broadcast_tensors(*[torch.as_tensor(v) for v in [zp, zw]])

        # mesh on one axis
        f.rp, f.zp = torch.stack(torch.meshgrid(f.rp, f.zp)).flatten(1)
        f.rw, f.zw = torch.stack(torch.meshgrid(f.rw, f.zw)).flatten(1)

        self.cutoff_dist = torch.stack([
            f.rp.abs() + f.rw * cutoff_factor,
            f.zp.abs() + f.zw * cutoff_factor]
        ).norm(dim=0).max()

        # convert given FWHM values to factors
        f.rw = 4 * np.log(2) / f.rw**2
        f.zw = 4 * np.log(2) / f.zw**2

        # for calculating the volume, we either have:
        #  f.rp~=0: a 3d gauss of size (f.zw, f.rw, f.rw) centered at a point (first arg of min func)
        #  f.rp>>0: a 2d gauss of size (f.zw, f.rw) along a circle of length 2*pi*f.rp (second arg of min func)
        # strictly, we would need to blend between the two options, in practice, min should be good enough
        volume = ((f.rw * f.zw) / np.pi ** 2)**0.5 * torch.min((f.rw / np.pi)**0.5, 1 / (2 * np.pi * f.rp))
        f.n = factor * volume**(obj_dim / 3)

        # store as constants
        self.f = nn.ParameterDict(val_map(NonTrainable, f))

    @property
    def n(self):
        return len(self.f.n)

    def named(self, name='f'):
        return DotDict(val_map(lambda v: v.rename(name), self.f))

    def __call__(self, r, z, f=None):
        f = DotDict(f or self.f)
        r2 = f.rw * (r - f.rp) ** 2
        z2 = f.zw * (z - f.zp) ** 2
        return f.n * (-(r2 + z2)).exp()


# function to create simple filter stack with equidistant rings and levels
def simple_filter_stack(nr, nz, w, start=1, **kwargs):
    return FilterStack(**kwargs,
        rw=w, rp=w * (torch.arange(nr) + start),
        zw=w, zp=w * (torch.arange(nz) - (nz - 1) / 2),
    )


class PConv3D(nn.Module):
    def __init__(self, p, k, c, w, nr, nz, rho=300, real=True, bias=True, w_trafo='auto', order='pckj'):
        super().__init__()

        # store parameters
        self.ap, self.bp = p
        self.ak, self.bk = k
        self.ac, self.bc = c
        self.real = real
        self.conv_func = point_conv_keops
        self.order = order

        # input / output properties
        (self.d_ak, self.d_aj), (self.d_bk, self.d_bj) = (map(NonTrainable, [
            torch.arange(-k * (not real), k + 1),
            torch.arange(1 + (k != 0 or not real)),
        ]) for k in [self.ak, self.bk])

        # filter properties
        self.f_ak = NonTrainable(torch.arange(-self.ak, self.ak + 1))
        self.f_bk = NonTrainable(torch.arange(-self.bk, self.bk + 1))
        # adapt width and point density as number of points is divided by 4 for each layer
        self.profile = simple_filter_stack(nr=nr, nz=nz, w=w, factor=4**self.ap / rho)

        # weights_old and bias
        self.w_trafo = w_trafo if w_trafo != 'auto' else symmetrize_complex if real else lambda x: x
        self.w_names = ('b_k', 'a_c', 'a_k', 'w_f', 'b_c', 'j')
        self.weights = nn.Parameter(torch.zeros(len(self.f_bk), self.ac, len(self.f_ak), 1 + self.profile.n, self.bc, 2))
        self.bias = nn.Parameter(torch.zeros(self.bc)) if bias else None

        # init weights (bias already set to zero)
        nn.init.normal_(self.weights, std=np.sqrt(2 / np.prod(self.weights.shape[1:4])))

    def forward(self, inputs):
        data, point_stack = inputs
        data = torch.view_as_real(data) if data.is_complex() else data[..., None]

        # prepare descriptors
        data_a_desc = DotDict(point_stack[self.ap], k=self.d_ak.rename('k'), j=self.d_aj.rename('j'))
        data_b_desc = DotDict(point_stack[self.bp], k=self.d_bk.rename('k'), j=self.d_bj.rename('j'))
        filter_desc = DotDict(self.profile.named(), ak=self.f_ak.rename('a_k'), bk=self.f_bk.rename('b_k'))

        # apply pointwise fourier filters
        x = self.conv_func(data.refine_names(*self.order), data_a_desc, data_b_desc, filter_desc, profile=self.profile)

        # apply trainable weights
        weights_transformed = self.w_trafo(self.weights.rename(*self.w_names))
        x = apply_weights_bmm(x, weights_transformed, filter_desc, data_b_desc, align=self.order)

        # apply bias
        if self.bias is not None:
            x[:, :, self.d_bk == 0, 0] += self.bias[..., None]

        x = x[..., 0] if x.shape[-1] == 1 else torch.view_as_complex(x.contiguous())
        return x, point_stack


class RandRotGeneric(nn.Module):
    STORE = False

    @classmethod
    def grab_rot_matrices(cls, call):
        cls.STORE = []
        func_ret = call()
        rot_matrices, cls.STORE = cls.STORE, False
        return func_ret, rot_matrices

    def forward(self, input):
        p, n, obj_size = input

        # build a random pointwise rotation matrix for each batch element
        R = torch.empty(len(obj_size[0]), 3, 3, device=p.device)
        R[:, 2] = self.get_z_axis(len(R), device=R.device)
        R[:, 0] = torch.rand(len(R), 3, device=R.device)
        R[:, 0] = F.normalize(R[:, 0] - R[:, 2] * (R[:, 0] * R[:, 2]).sum(-1, keepdims=True))
        R[:, 1] = F.normalize(torch.cross(R[:, 2], R[:, 0]))

        # store if requested
        if isinstance(self.STORE, list):
            self.STORE.append(R.detach())

        # inflate to full point size
        R = R[arange_like(obj_size[0], repeat=True)]

        # apply rotation to points and normals
        p = (R @ p[..., None]).squeeze(-1)
        n = (R @ n[..., None]).squeeze(-1)
        return p, n, obj_size

    def get_z_axis(self, n, device):
        raise NotImplementedError('use a specific subclass')


class RandRotZ(RandRotGeneric):
    def get_z_axis(self, n, device): return torch.tensor([0, 0, 1], device=device)


class RandRotSO3(RandRotGeneric):
    def get_z_axis(self, n, device): return F.normalize(torch.rand(n, 3, device=device))


class BuildPointStack(nn.Module):
    def forward(self, input, point_order=lambda p: torch.arange(len(p), device=p.device)):
        p, n, obj_size = input
        assert p.shape == n.shape == (obj_size[0].sum(), 3)
        assert obj_size.shape[1] < 2 ** 15
        idx = torch.stack([arange_like(p), (arange_like(obj_size[0], repeat=True) << 48) + point_order(p)])
        pzxy = torch.stack([p, n, *orthonormal_system(n)], 1)
        stack = []
        for prev, size in pairwise([obj_size[0], *obj_size]):
            if size is not prev:
                idx = torch.cat([idx[:, e - s: e] for s, e in zip(size, prev.cumsum(0))], -1)
            pool = idx[0, (order := idx[1].argsort())]
            pzxy = pzxy[pool]
            stack.append(DotDict(dict(
                zip('pzxy', pzxy.movedim(1, 0).rename('ax', 'p', 'xyz')),
                pzxy=pzxy.rename('p', 'ax', 'xyz'),
                xy=torch.view_as_complex(pzxy[:, 2:].permute(0, 2, 1).contiguous()).rename('p', 'xyz'),
                # xy as complex data
                b=(idx[1] >> 48).rename('p'), pool_idx=pool, obj_size=size, prev_len=prev.sum()
            )))
            inverse_order(order, target=idx[0])
        return stack


class PointInput(nn.Module):
    def __init__(self, p, val):
        super().__init__()
        self.p = 0
        self.val = val

    def forward(self, point_stack):
        in_p = point_stack[self.p].p
        data = torch.ones((len(in_p), 1, 1), device=in_p.device) * self.val
        return data, point_stack


class PPool3D(nn.Module):
    def __init__(self, p, pool_func=torch.mean):
        super().__init__()
        self.p = p
        self.pool_func = pool_func

    def forward(self, input):
        data, point_stack = input
        ends = F.pad(point_stack[self.p].obj_size, (1, 0)).cumsum(0)
        return torch.stack([self.pool_func(data[a: b], 0) for a, b in pairwise(ends)])
