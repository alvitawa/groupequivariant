from torch import nn
from .layers_generic import On0, FBN, FClamp, NormAct, With_IRFFT, Store, NormReduction
from .layers_2d import Img2Pts, PConv2D, GPool


MnistRotPreprocessing = Img2Pts


def MnistRotModel(k, act, kf=None, clamp=None, act_pad=None, drop_p=0.3, bn={'momentum': None}, inc_pp=True):
    k0 = min(k, 6)  # avoid unnecessary channels if k>6 (filter basis has max. freq 6 in first layer)
    kf = k0 if kf is None else kf  # set kf=0 for conv2triv reduction
    C = lambda: [FClamp(clamp)] if clamp else []
    A = lambda k, c: NormAct(act(), shape=(c, k)) if act_pad == 'norm' else With_IRFFT(act(), padding=act_pad)
    return lambda: nn.Sequential(
        (MnistRotPreprocessing() if inc_pp else nn.Identity()),
        PConv2D(size=9, ch_inp=1,  ch_out=24, k_inp=0, k_out=k0, unpad=4), On0(FBN(24, **bn), *C(), A(k0, 24), Store()),
        PConv2D(size=7, ch_inp=24, ch_out=32, k_inp=k, k_out=k), GPool(2), On0(FBN(32, **bn), *C(), A(k,  32), Store()),
        PConv2D(size=7, ch_inp=32, ch_out=36, k_inp=k, k_out=k),           On0(FBN(36, **bn), *C(), A(k,  36), Store()),
        PConv2D(size=7, ch_inp=36, ch_out=36, k_inp=k, k_out=k), GPool(2), On0(FBN(36, **bn), *C(), A(k,  36), Store()),
        PConv2D(size=7, ch_inp=36, ch_out=64, k_inp=k, k_out=k),           On0(FBN(64, **bn), *C(), A(k,  64), Store()),
        PConv2D(size=5, ch_inp=64, ch_out=96, k_inp=k, k_out=kf, unpad=2), On0(FBN(96, **bn), *C(), A(kf, 96), Store()),
        On0(cut=True), NormReduction(), nn.Flatten(1),
        nn.Dropout(p=drop_p), nn.Linear(96*(1+kf), 96), nn.BatchNorm1d(96, **bn), act(),
        nn.Dropout(p=drop_p), nn.Linear(96,        96), nn.BatchNorm1d(96, **bn), act(),
        nn.Dropout(p=drop_p), nn.Linear(96,        10), Store(),
    )
