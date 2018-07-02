#!/bin/bash
# install dependencies
# (this script must be run as root)


apt-get -y update
apt-get -y install build-essential cmake wget libjpeg-dev gcc g++

# download cuda and cudnn
CUDA_REPO_PKG=""
if [[ "$CUDA_VERSION" == "8.0" ]]; then
    CUDA_REPO_PKG=cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
elif [[ "$CUDA_VERSION" == "7.5" ]]; then
    CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb
fi

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
dpkg -i $CUDA_REPO_PKG
rm $CUDA_REPO_PKG

ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG
dpkg -i $ML_REPO_PKG

apt-get -y update
# install packages
apt-get install -y --no-install-recommends \
	cuda-core-$CUDA_VERSION \
	cuda-cudart-dev-$CUDA_VERSION \
   	cuda-cublas-dev-$CUDA_VERSION \
	cuda-curand-dev-$CUDA_VERSION \
    cuda-cufft-dev-$CUDA_VERSION

# manually create CUDA symlink
ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
apt-get install -y --no-install-recommends libcudnn${CUDNN_VERSION}-dev

# download glog and make install
wget https://github.com/google/glog/archive/v0.3.5.tar.gz
tar -xzf v0.3.5.tar.gz
cd glog-0.3.5
./configure --prefix=/usr/local/glog > /dev/null
make -j > /dev/null
make install
cd ..

# download OpenMPI and make install
# wget --no-check-certificate https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.0.tar.gz
# tar -xzf openmpi-2.1.0.tar.gz
# cd openmpi-2.1.0
# ./configure --prefix=/usr/local/mpi > /dev/null
# make -j > /dev/null
# make install
# cd ..

cp /usr/include/cudnn.h /usr/local/cuda/include/
cp /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/
cp scripts/config.linux .
