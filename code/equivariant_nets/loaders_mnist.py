# we reuse the loaders from the e2cnn experiment by Maurice Weiler and Gabriele Cesa
# https://github.com/QUVA-Lab/e2cnn_experiments
from e2cnn_experiments.experiments.datasets.mnist_rot.data_loader_mnist_rot import build_mnist_rot_loader


# avoid mnist-rot loader throwing lot of errors
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def loaders_mnist(batch_size):
    return {
        'train': build_mnist_rot_loader("trainval", batch_size=batch_size, rot_interpol_augmentation=True)[0],
        'test':  build_mnist_rot_loader("test", batch_size=batch_size)[0]
    }
