# SuperNeurons

[![Build Status](https://travis-ci.com/linnanwang/superneurons.svg?token=NaYnnUzyHsfFSY6YVdAG&branch=master)](https://travis-ci.com/linnanwang/superneurons)

Superneurons is a brand new deep learning framework built for HPC. It is written in C++ and the codes are easy to modify to work for major HPC libraries.

The first release is a mere demonstration of framework architecture, and future releases will incorporate other cool features.



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

Developers: Wei Wu (Los Alamos National Lab), Jinmian Ye (UESTC), and Yiyang Zhao (Northeastern University)

### Joining us?

For for information, please contact wangnan318@gmail.com. We're also looking for people to collaborate on this project, please feel free to email me if you're interested.

Please cite Superneurons in your publications if it helps your research:
<p>
Wang, Linnan, et al. "Superneurons: dynamic GPU memory management for training deep neural networks." Proceedings of the 23rd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. ACM, 2018.
</p>
