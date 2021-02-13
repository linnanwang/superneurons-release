# SuperNeurons

[![Build Status](https://travis-ci.com/linnanwang/superneurons.svg?token=NaYnnUzyHsfFSY6YVdAG&branch=master)](https://travis-ci.com/linnanwang/superneurons)

Superneurons is a brand new deep learning framework built for HPC. It is written in C++ and the codes are easy to modify to work for major HPC libraries. The first release is a mere demonstration of framework architecture. 

### one year after the initial release 
As a graduate student, I'm no longer able to maintain the code, and I decided to invest much of my time on Neural Architecture Search in hoping to build an AI that builds AI. However, <a href="https://github.com/microsoft/DeepSpeed">DeepSpeed</a> should provide a great alternative, and being compatiable to PyTorch and other frameworks.

### installation,

please configure config.osx or config.linux. Make a 'build' dir
```
mkdir build
cmake ..
make -j8
```
### Running the tests
please download cifar10 and mnist dataset, use the util convert_mnist and convert_cifar to prepare the dataset.

specify the path in texting\cifar10.cpp and change the path.

make again, and run the binaries at build/testing/cifar10

The testing folder has a variety of networks.

### Contributors
Chief Architect: Linnan Wang (Brown University)

Developers: Jinmian Ye (UESTC) and Yiyang Zhao (WPI)

For for information, please contact wangnan318@gmail.com. We're also looking for people to collaborate on this project, please feel free to email me if you're interested.

Please cite Superneurons in your publications if it helps your research:
<p>
Wang, Linnan, et al. "Superneurons: dynamic GPU memory management for training deep neural networks." Proceedings of the 23rd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. ACM, 2018.
</p>
<p>
Wang, Linnan, Wei Wu, Yiyang Zhao, Junyu Zhang, Hang Liu, George Bosilca, Jack Dongarra, Maurice Herlihy, and Rodrigo Fonseca. "SuperNeurons: FFT-based Gradient Sparsification in the Distributed Training of Deep Neural Networks." arXiv preprint arXiv:1811.08596 (2018).
</p>


