import torch
from torch import nn
from equivariant_nets.loaders_modelnet import loaders_modelnet40_presampled
from equivariant_nets.train_runners import run_multi
from equivariant_nets.layers_3d import RandRotZ, RandRotSO3
from equivariant_nets.models_3d import ModelNet40Preprocessing, ModelNet40Model


net, acc = run_multi(
    n_runs=5,
    model=ModelNet40Model(w=0.1, k=4, nr=1, nz=3, act_pad=127, act=nn.ReLU),
    train_pp=ModelNet40Preprocessing(augment=nn.Identity),
    test_pps=[
        ('N',   ModelNet40Preprocessing(augment=nn.Identity)),
        ('z',   ModelNet40Preprocessing(augment=RandRotZ)),
        ('SO3', ModelNet40Preprocessing(augment=RandRotSO3)),
    ],
    epochs=30,
    loss=nn.CrossEntropyLoss,
    opt=torch.optim.Adam,
    lr=lambda epoch: 0.001 * (1 - min(1, max(0, epoch - 10) / 20)),
    save_file='modelnet40_9coeff_ReLU_fftpad127_trainN.weights',
    **loaders_modelnet40_presampled(batch_size=32),
)
