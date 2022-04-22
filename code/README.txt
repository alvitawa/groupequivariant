This is the code accompanying our paper "Nonlinearities in Steerable SO(2)-Equivariant CNNs".

To re-run our experiments, you need python 3.7 with the following packages installed:
numpy, pytorch, e2cnn

Our different experiments are located as single python files directly within the code folder.



For the 2D experiments on MNIST-rot:

The MNIST-rot dataset, as well as the e2cnn_experiments github repository by Maurice Weiler, Gabriele Cesa (we use the data loader from this repository) are required:
https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits
https://github.com/QUVA-Lab/e2cnn_experiments
These can be downloaded automatically under Linux by running the "download_mnist_rot_dataset_and_loaders.sh" script.

 

For the 3D surfel experiment on Modelnet40:

The pykeops python package is additionally required for sparse operations on point clouds.
Our 3D surfel architecture requires the input to be resampled to point clouds and stored as numpy arrays (".npy" files) in the shape (points, 2, 3), where the second axis contains points and normals, and last axis contains the (x,y,z)-coordinates.
