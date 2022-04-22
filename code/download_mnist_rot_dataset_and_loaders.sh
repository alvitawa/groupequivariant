# get e2cnn_experiments github repository
git clone https://github.com/QUVA-Lab/e2cnn_experiments.git

# download mnist-rot
mkdir -p datasets && cd datasets || exit
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
unzip -n mnist_rotation_new.zip -d mnist_rot

# convert mnist-rot
echo Converting mnist_rot...
cd mnist_rot && python ../../e2cnn_experiments/experiments/datasets/mnist_rot/convert.py
