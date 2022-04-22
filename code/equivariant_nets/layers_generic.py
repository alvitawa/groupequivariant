import torch
from torch import nn
from .pytorch_helpers import NonTrainable


# construct a layer for any function
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# skip the grid for layers that don't require it, remove it
class On0(nn.Sequential):
    def __init__(self, *args, cut=False):
        super().__init__(*args)
        self.cut = cut

    def forward(self, input):
        x, *others = input
        x = super().forward(x)
        if self.cut: return x
        return x, *others


# fourier conversion layer
class With_IRFFT(nn.Sequential):
    def __init__(self, *args, padding=None, norm='forward'):
        super().__init__(*args)
        self.norm = norm
        self.padding = padding  # use the 'forward' norm to make the transformation independent of padding
        assert padding is not None, 'padding must be set'

    def forward(self, input):
        x = input.flatten(0, -3)
        x = torch.fft.irfft(x, n=2 * x.shape[-1] - 1 + self.padding, norm=self.norm) if input.shape[-1] > 1 else x.real
        x = super().forward(x)  # (b*p, ch, values), compatible with e.g. nn.BatchNorm1d
        x = torch.fft.rfft(x, norm=self.norm)[..., :input.shape[-1]] if input.shape[-1] > 1 else x + 0j
        return x.reshape(*input.shape[:-2], *x.shape[1:])


# simple layer to grab results at various points in a network
class Store(nn.Module):
    STORE = False

    @classmethod
    def grab_intermediate_results(cls, call):
        cls.STORE = []
        func_ret = call()
        intermediate_results, cls.STORE = cls.STORE, False
        return func_ret, intermediate_results

    def forward(self, inputs):
        if isinstance(self.STORE, list):
            self.STORE.append((inputs[0] if isinstance(inputs, tuple) else inputs).detach())
        return inputs


# polynomial nonlinearities
class PolyReLU(nn.Module):
    def __init__(self, deg, c=0):
        super().__init__()
        coefficients = torch.tensor({
            # ReLU approximation from "POLYNOMIAL ACTIVATION FUNCTIONS" by Vikas Gottemukkula
            # https://openreview.net/forum?id=rkxsgkHKvH
            # suitable for inputs in range [-5, 5]
            2: [0.47, 0.50, 0.09],
            4: [0.29, 0.50, 0.16,  1.8e-10, -3.3e-03],
            6: [0.21, 0.50, 0.23,  7.6e-08, -1.1e-02, -3.5e-09, 2.3e-04],
        }[deg], dtype=torch.float32)
        if c == 0:  # non-trainable
            self.coefficients = NonTrainable(coefficients)
        else:  # trainable, use ch=1 for same weight in all channels
            self.coefficients = nn.Parameter(coefficients.repeat(c, 1, 1))

    def forward(self, x):
        return (self.coefficients * x[..., None] ** torch.arange(len(self.coefficients), device=x.device)).sum(-1)


# clamping in fourier space for use with polynomial activations
class FClamp(nn.Module):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def forward(self, input, eps=1e-8):
        supremum = input[..., :1].abs() + 2 * input[..., 1:].abs().sum(-1, keepdim=True) + eps
        return input * (self.limit / supremum).clamp(0, 1)


# norm-only nonlinearity
class NormAct(nn.Sequential):
    def __init__(self, *args, shape, eps=1e-8):
        super().__init__(*args)
        self.bias = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, input):
        c0, c1 = input[..., :1].real, input[..., 1:]
        c1_abs = c1.abs() + self.eps
        with torch.no_grad():
            self.bias.data.clamp_(0, None)
        result = super().forward(torch.cat([c0, c1_abs - self.bias], -1))
        r0, r1 = result[..., :1], result[..., 1:]
        return torch.cat([r0, c1 * (r1 / c1_abs)], -1)


# reduce to invariant output by taking the norm
class NormReduction(nn.Module):
    def forward(self, input):
        return torch.cat([input[..., :1].real, input[..., 1:].abs()], -1)


# batch norm for data in fourier basis
class FBN(nn.Module):
    def __init__(self, ch, eps=1e-05, momentum=None, affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch)) if affine else None
        self.bias = nn.Parameter(torch.zeros(ch)) if affine else None
        self.running_var = NonTrainable(torch.ones(ch))
        self.running_avg = NonTrainable(torch.zeros(ch))
        self.num_batches = NonTrainable(torch.zeros(1))
        self.reset_running_stats()
        self.momentum = momentum
        self.eps = eps

    def reset_running_stats(self):
        self.running_avg.fill_(0)
        self.running_var.fill_(1)
        self.num_batches.fill_(0)

    def forward(self, input):
        if self.training:
            red = input.flatten(0, -3)
            avg = red[..., 0].real.mean(0)
            avg_var = (red[..., 0].real - avg)**2
            var = ((red[..., 1:].abs() ** 2).sum(-1) * (red.shape[-1] - 1) / 2 + avg_var).mean(0)
            with torch.no_grad():
                if self.momentum:
                    self.running_avg *= 1 - self.momentum
                    self.running_var *= 1 - self.momentum
                self.running_avg += avg if self.momentum is None else avg * self.momentum
                self.running_var += var if self.momentum is None else var * self.momentum
                self.num_batches += 1
        else:
            avg = self.running_avg * ((1 if self.momentum is None else self.num_batches) / (self.num_batches - 0))
            var = self.running_var * ((1 if self.momentum is None else self.num_batches) / (self.num_batches - 1))

        res = input.clone()
        res[..., 0] -= avg
        res *= ((1 if self.weight is None else self.weight) / (var + self.eps) ** 0.5)[..., None]
        if self.bias is not None: res[..., 0] += self.bias
        return res
