import torch
from torch import nn
from equivariant_nets.train_runners import run_multi
from equivariant_nets.loaders_mnist import loaders_mnist
from equivariant_nets.models_2d import MnistRotModel
from equivariant_nets.layers_generic import PolyReLU

net, acc = run_multi(
    n_runs=10,
    model=MnistRotModel(k=4, clamp=5, act_pad=8, act=lambda: PolyReLU(2)),
    epochs=40,
    loss=nn.CrossEntropyLoss,
    opt=torch.optim.Adam,
    lr=lambda epoch: 0.015 * 0.8 ** max(0, (epoch - 15)),
    save_file='mnist_rot_9coeff_PolyReLU_deg2_fftpad8_norm.weights',
    **loaders_mnist(batch_size=64),
)
