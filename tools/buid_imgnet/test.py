# coding: utf-8
# Created by ay27 at 27/01/2018
import numpy as np
from binary import ImageNet
import os
import cv2


def test1():
    print('test1')
    x = ImageNet('/home/ay27/hdd/imagenet/bin_file_256/val_label_0.bin', 'i')
    res = x.unpack(*x.read(50000))

    print(x.N, x.C, x.H, x.W)
    print(res.shape)
    print res.transpose()


def test2():
    print('test2')
    x = ImageNet('/home/ay27/hdd/imagenet/bin_file_256/val_data_0.bin', 'B')
    res = x.unpack(*x.read(100))
    print(x.N, x.C, x.H, x.W)
    print(res.shape)
    res = res.reshape(100, 3, 256, 256)
    print res[0][0][0]

    res = res[1].transpose([1, 2, 0])
    print res.shape
    cv2.imwrite('test2.jpg', res)


def test3():
    print('test3')
    x = ImageNet('/home/ay27/hdd/imagenet/py_bin/train_label0.bin', 'i')
    res = x.unpack(*x.read(50000))
    print(x.N, x.C, x.H, x.W)
    print(res.shape)
    print res.transpose()


def test4():
    print('test4')
    x = ImageNet('/home/ay27/hdd/imagenet/py_bin/train_data0.bin', 'B')
    res = x.unpack(*x.read(1000))
    print(x.N, x.C, x.H, x.W)
    print(res.shape)
    res = res.reshape(1000, 3, 256, 256)
    print res[0]

    res = res[789].transpose([1, 2, 0])
    print res.shape

    cv2.imwrite('test4.jpg', res)


def test5():
    print('test5')
    x = np.load('ilsvrc_2012_mean.npy')
    print(x.shape)
    y = np.load('/home/ay27/hdd/imagenet/py_bin/train_mean.binpkl')
    print(y.shape)

    print(np.sum(x, (1, 2)) / float(256 * 256))
    print(np.sum(y, (1, 2)) / float(256 * 256))

    print(np.abs(x - y))


def test6():
    files = ['n03018349/n03018349_4028.JPEG', 'n02105855/n02105855_2933.JPEG', 'n01644373/n01644373_15797.JPEG',
             'n01675722/n01675722_2656.JPEG', 'n02099429/n02099429_4619.JPEG', 'n03450230/n03450230_3540.JPEG']

    img = cv2.imread(os.path.join('/home/ay27/hdd/imagenet/raw/train', files[1]), 1)
    img = cv2.resize(img, (256, 256))
    print img.shape
    cv2.imwrite('test6.jpg', img)


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    print('test finish')
