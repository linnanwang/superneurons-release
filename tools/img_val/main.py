# Created by ay27 at 17/4/9
import os

import matplotlib.pyplot as plt
import struct
import numpy as np


def trans(row):
    return list(map(lambda x: np.uint8(x), row))


def read_image(filename):
    with open(filename, mode='rb') as file:
        n = file.read(8)
        n = struct.unpack("<Q", n)[0]
        c = file.read(8)
        c = struct.unpack("<Q", c)[0]
        h = file.read(8)
        h = struct.unpack("<Q", h)[0]
        w = file.read(8)
        w = struct.unpack("<Q", w)[0]
        print(n, c, h, w)

        for ii in range(n):
            r = trans(file.read(h*w))
            g = trans(file.read(h*w))
            b = trans(file.read(h*w))

            if ii == 100:
                break

        print(file.tell() == os.fstat(file.fileno()).st_size)

        img = np.array([r,g,b]).transpose(1,0).reshape(h,w,c)
        print(img.shape)
        plt.imshow(img)
        plt.show()


def read_label(path, ground_truth=None):
    with open(path, 'rb') as file:
        n = file.read(8)
        n = struct.unpack("<Q", n)[0]
        c = file.read(8)
        c = struct.unpack("<Q", c)[0]
        h = file.read(8)
        h = struct.unpack("<Q", h)[0]
        w = file.read(8)
        w = struct.unpack("<Q", w)[0]
        print(n, c, h, w)

        label = []
        sets = set()
        while not (file.tell() == os.fstat(file.fileno()).st_size):

            ch = file.read(4)
            num = struct.unpack("<l", ch)[0]
            label.append(num)
            sets.add(num)
        # print(file.tell() == os.fstat(file.fileno()).st_size)
        print(label)
        print(len(label))

        # print(label[900],label[901], label[902], label[903], label[904])

        return label

        # if ground_truth:
        #     g = []
        #     with open(ground_truth) as file:
        #         for line in file:
        #             g.append(int(line.split(' ')[1]))
        #     np.testing.assert_array_equal(g, label)

if __name__ == '__main__':
    # read_image('../../data/ilsvrc2012/img.bin')
    # read_label('../../data/ilsvrc2012/label.bin', '../../data/ilsvrc2012/val.txt')
    # read_image('../../build/cifar100_train_image.bin')
    # read_label('../../build/cifar100_train_label.bin')

    read_image('../../build/val_data_8.bin')
    for i in range(10):
        read_label('../../build/val_label_%d.bin' % i)

    # labels = []
    # for i in range(10):
    #     labels.append(read_label('../../build/val_label_%d.bin' % i))
    #
    # ground = []
    # with open('../../build/shuffled_list') as file:
    #     ground.append()