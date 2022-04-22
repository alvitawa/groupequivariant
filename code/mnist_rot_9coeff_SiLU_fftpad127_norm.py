import torch
from torch import nn
from equivariant_nets.train_runners import run_multi
from equivariant_nets.loaders_mnist import loaders_mnist
from equivariant_nets.models_2d import MnistRotModel


net, acc = run_multi(
    n_runs=10,
    model=MnistRotModel(k=4, act_pad=127, act=nn.SiLU),
    epochs=40,
    loss=nn.CrossEntropyLoss,
    opt=torch.optim.Adam,
    lr=lambda epoch: 0.015 * 0.8 ** max(0, (epoch - 15)),
    save_file='mnist_rot_9coeff_SiLU_fftpad127_norm.weights',
    **loaders_mnist(batch_size=64),
)
