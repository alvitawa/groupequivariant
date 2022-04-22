from torch import nn
from .layers_generic import On0, FBN, FClamp, NormAct, With_IRFFT, Store
from .layers_3d import BuildPointStack, PointInput, PConv3D, PPool3D


def ModelNet40Preprocessing(augment, dither_normals=True, flip_normals_out=True):
    return lambda: nn.Sequential(
        augment(),
        BuildPointStack(),
    )


def ModelNet40Model(w, k, act, clamp=None, act_pad=None, bn={'momentum': None}, **kwargs):
    C = lambda: [FClamp(clamp)] if clamp else []
    A = lambda k, c: NormAct(act(), shape=(c, k)) if act_pad == 'norm' else With_IRFFT(act(), padding=act_pad)
    return lambda: nn.Sequential(
        nn.Identity(),
        PointInput(p=0, val=1),
        PConv3D(p=(0, 1), k=(0, k), c=(1,  16), w=w*1, **kwargs), On0(FBN(16, **bn),   *C(), A(k, 16), Store()),
        PConv3D(p=(1, 2), k=(k, k), c=(16, 32), w=w*2, **kwargs), On0(FBN(32, **bn),   *C(), A(k, 32), Store()),
        PConv3D(p=(2, 2), k=(k, k), c=(32, 48), w=w*4, **kwargs), On0(FBN(48, **bn),   *C(), A(k, 48), Store()),
        PConv3D(p=(2, 2), k=(k, k), c=(48, 64), w=w*4, **kwargs), On0(FBN(64, **bn),   *C(), A(k, 64), Store()),
        PConv3D(p=(2, 2), k=(k, k), c=(64, 96), w=w*8, **kwargs), On0(FBN(96, **bn),   *C(), A(k, 96), Store()),
        PConv3D(p=(2, 2), k=(k, 0), c=(96, 40), w=w*8, **kwargs), On0(nn.BatchNorm1d(40, **bn), act(), Store()),
        PPool3D(p=2), nn.Flatten(1), Store(),
    )
