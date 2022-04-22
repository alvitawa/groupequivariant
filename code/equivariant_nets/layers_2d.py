import torch
from torch import nn
import torch.nn.functional as F
from .python_helpers import DotDict, ld_to_dl, val_map
from .pytorch_helpers import NonTrainable


# convert input image to point cloud (flatten inputs and add coordinate grid)
class Img2Pts(nn.Module):
    def forward(self, input):
        grid = torch.stack(torch.meshgrid([torch.arange(s, device=input.device) for s in input.shape[2:]])).float()
        data = input.flatten(2).moveaxis(1, -1)[..., None]
        return data, grid


# pooling based on grid
class GPool(nn.Module):
    def __init__(self, kernel_size, stride=None, pool_func=F.avg_pool2d):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.pool_func = pool_func

    def forward(self, input):
        data, grid = input
        x = data.flatten(2).swapaxes(1, 2)
        x = x.reshape(*x.shape[:-1], *grid.shape[1:])
        grid = grid[:, ::self.stride, ::self.stride] / self.stride
        x = torch.complex(
            self.pool_func(x.real, kernel_size=self.kernel_size, stride=self.stride),
            self.pool_func(x.imag, kernel_size=self.kernel_size, stride=self.stride),
        ).flatten(2).swapaxes(1, 2)
        x = x.reshape(data.shape[0], x.shape[1], *data.shape[2:])
        assert x.shape[1] == grid.flatten(1).shape[1]
        return x, grid


# build same basis as in e2cnn
def e2cnn_basis(kernel_size, inp_rot, out_rot):
    from e2cnn.nn.modules.r2_conv.r2convolution import compute_basis_params
    grid, basis_filter, rings, sigma, max_f = compute_basis_params(kernel_size=kernel_size)
    frequencies = torch.subtract(*torch.meshgrid(out_rot, inp_rot)).unique()
    basis = [dict(radius=r, sigma=s, frequency=f) for r, s in zip(rings, sigma) for f in frequencies if abs(f) <= max_f]
    return ld_to_dl(filter(basis_filter, basis))


# basic 2D point convolution, supports any combination of rotation orders
class FlexChannelPConv2D(nn.Module):
    def __init__(self, size, inp_rot, out_rot, complex_mode=False, basis=e2cnn_basis, fmt='bpc', unpad=0):
        super().__init__()
        self.size = size
        self.inp_rot = NonTrainable(inp_rot)
        self.out_rot = NonTrainable(out_rot)
        self.complex_mode = complex_mode
        if callable(basis):
            basis = basis(size, self.inp_rot, torch.cat([self.out_rot]+[-self.out_rot][:not self.complex_mode]))
        self.basis = nn.ParameterDict(val_map(NonTrainable, basis))
        self.fmt = fmt
        self.unpad = unpad
        assert self.complex_mode or ((self.inp_rot >= 0).all() and (self.out_rot >= 0).all())

        # get frequencies after preprocessing step
        pre_inp_rot, pre_flt_rot = map(torch.flatten, torch.meshgrid(self.inp_rot, self.basis.frequency))
        pre_out_rot = pre_inp_rot + pre_flt_rot  # rotation orders of inputs and filters add up

        pre_mul = 1
        if not complex_mode:  # flip negative coefficients and remove duplicates
            pre_mul = torch.stack([torch.ones_like(pre_out_rot), (pre_out_rot + 0.5).sign()], -1)
            pre_mul[pre_inp_rot == 0] *= 0.5 * (pre_out_rot[pre_inp_rot == 0, None].sign() + 1)
            pre_mul[pre_out_rot == 0] *= 2
            pre_out_rot = pre_out_rot.abs()

        # define connectivity (which input+filter combinations map to which outputs)
        weight_connectivity = torch.eq(*torch.meshgrid(pre_out_rot, self.out_rot))

        # reduce number of inputs to weight application if possible (for speedup only)
        self.pre_cut = NonTrainable(pre_mul.any(1) & weight_connectivity.any(1))
        if self.pre_cut.all():
            self.pre_cut = None
        else:
            pre_mul = pre_mul[self.pre_cut]
            weight_connectivity = weight_connectivity[self.pre_cut]

        # store as parameters
        self.pre_mul = NonTrainable(pre_mul)
        self.weight_connectivity = NonTrainable(weight_connectivity)

        # create weights and initialize
        self.weights = nn.Parameter(torch.zeros(weight_connectivity.sum(), 2).float())
        nn.init.normal_(self.weights, std=(2 / self.weights.shape[0])**0.5)

    def forward(self, input):
        data, inp_grid = input
        out_grid = inp_grid if self.unpad == 0 else inp_grid[:, self.unpad:-self.unpad, self.unpad:-self.unpad]

        # make sure data has complex dtype
        if not torch.is_complex(data):
            assert not self.inp_rot.any()
            data = data + 0j

        # calculate relative coordinates / radii / angles for each pair of points
        pos = DotDict()
        pos.coords = (inp_grid.flatten(1)[:, :, None] - out_grid.flatten(1)[:, None])[..., None]
        pos.radius = pos.coords.norm(dim=0)
        pos.angles = torch.atan2(*pos.coords)

        # generate and apply preprocessing filterbank
        pre_filter = torch.exp(torch.complex(
            -((pos.radius - self.basis.radius) / (2**0.5 * self.basis.sigma)) ** 2,
            pos.angles * self.basis.frequency,
        )) * ((pos.radius != 0) | (self.basis.frequency == 0))  # fix singularities
        preprocess = torch.view_as_real(torch.einsum(f'{self.fmt},pqf->bqcf', data, pre_filter).flatten(2))
        if self.pre_cut is not None:
            preprocess = preprocess[..., self.pre_cut, :]
        preprocess = torch.view_as_complex(preprocess * self.pre_mul)

        # expand sparsely stored trainable weights to full size and apply
        weights = torch.zeros(*self.weight_connectivity.shape, 2, dtype=self.weights.dtype, device=self.weights.device)
        weights[self.weight_connectivity] = self.weights
        output = torch.einsum(f'bpf,fc->{self.fmt}', preprocess, torch.view_as_complex(weights))

        if not self.complex_mode:
            output.imag[..., self.out_rot == 0] = 0

        return output, out_grid


# variant of FlexChannelPConv2D using real-valued data and rotation orders up to a specified value per channel
class PConv2D(FlexChannelPConv2D):
    def __init__(self, ch_inp, ch_out, k_inp, k_out, **kwargs):
        self.out_shape = (ch_out, k_out + 1)
        super().__init__(
            inp_rot=torch.arange(k_inp + 1).repeat(ch_inp),
            out_rot=torch.arange(k_out + 1).repeat(ch_out),
            complex_mode=False, **kwargs,
        )

    def forward(self, input):
        data, *other = input
        data, *other = super().forward((data.flatten(2, 3), *other))
        data = data.reshape(*data.shape[:2], *self.out_shape)
        return data, *other
