# coding: utf-8
# Created by ay27 at 27/01/2018
import struct
import threading
import numpy as np


class ImageNet(object):
    def __init__(self, file_path, data_type='B'):
        """
        :param file_path:
        :param data_type: 'B' for uint8 and 'i' for int32
        """
        assert (data_type == 'B') or (data_type == 'i')
        self.file_path = file_path
        self.WF = None
        self.RF = None
        self.data_type = data_type
        self.type_len = 1 if data_type == 'B' else 4
        self.dtype = np.uint8 if data_type == 'B' else np.int32
        self.N = 0
        self.C = 0
        self.H = 0
        self.W = 0
        self.RL = threading.Lock()
        self.WL = threading.Lock()
        self.r_cnt = 0

    def reset_NCHW(self):
        """
        when write finish, we should reset the meta data NCHW, because we can not get the real N before writing finish.
        :return:
        """
        if self.WF is not None:
            self.WF.close()
            self.WF = None
        with open(self.file_path, 'rb+') as f:
            f.seek(0, 0)
            f.write(struct.pack('<QQQQ', self.N, self.C, self.H, self.W))

    def read_meta(self):
        """
        read meta data NCHW
        :return:
        """
        if self.RF is None:
            self.RF = open(self.file_path, 'rb')
            self.N, self.C, self.H, self.W = struct.unpack('<QQQQ', self.RF.read(32))
            self.r_cnt = 0

    def pack(self, N, C, H, W, array):
        """
        pack data into binary
        :param array:
        :return:
        """
        array = np.asarray(array)
        array = array.reshape(N, C * H * W)

        fmt = '<' + self.data_type * (C * H * W)
        res = []
        for ii in range(N):
            res.append(struct.pack(fmt, *(array[ii])))
        return res

    def write(self, n, c, h, w, packed):
        """
        write packed binary data into file
        :param packed:
        :return:
        """
        with self.WL:
            self.C, self.H, self.W = c, h, w
            self.N += n

            if self.WF is None:
                self.WF = open(self.file_path, 'wb', buffering=1024*1024*100)
                self.WF.write(struct.pack('<QQQQ', n, c, h, w))

            for ii in range(n):
                self.WF.write(packed[ii])

    def read(self, n):
        """
        read n samples
        :param n:
        :return: packed binary data
        """
        with self.RL:
            if self.RF is None:
                self.read_meta()

            if self.r_cnt + n > self.N:
                read_n = self.N - self.r_cnt
            else:
                read_n = n

            res = []
            for ii in range(read_n):
                res.append(self.RF.read(self.C * self.H * self.W * self.type_len))
            self.r_cnt += read_n

            return read_n, res

    def unpack(self, n, encoded_array):
        """
        unpack binary data
        :param encoded_array:
        :return:
        """
        fmt = '<' + self.data_type * (self.C * self.H * self.W)
        res = np.zeros((n, self.C * self.H * self.W), self.dtype)
        for ii in range(n):
            res[ii] = np.asarray(struct.unpack(fmt, encoded_array[ii]), dtype=self.dtype)
        return res

    def close(self):
        if self.WF is not None:
            self.WF.close()
        if self.RF is not None:
            self.RF.close()

    def flush(self):
        if self.WF is not None:
            self.WF.flush()
        if self.RF is not None:
            self.RF.flush()


if __name__ == '__main__':
    label_file = '/home/ay27/hdd/imagenet/bin_file/val_label_0.bin'
    img_file = '/home/ay27/hdd/imagenet/bin_file/val_data_0.bin'

    # label = ImageNet(label_file, 'i')
    #
    # tmp = label.read(100)
    # print(tmp)
    # label.close()
    #
    # imgs = ImageNet(img_file, 'B')
    # tmp = imgs.read(1)
    # print(tmp[0][:100])
    # imgs.close()

    N = 100
    C = 3
    H = 227
    W = 227
    tmp = np.random.randint(1, 255, (N * C * H * W,), np.uint8)
    imgs = ImageNet('test.bin', 'B')
    imgs.write(N, C, H, W, tmp)
    imgs.reset_NCHW(N, C, H, W)
    imgs.close()

    imgs = ImageNet('test.bin', 'B')
    tmp1 = imgs.read(1)
    print(imgs.N, imgs.C, imgs.H, imgs.W)
    print(tmp[:100])
    print(tmp1)
    print(np.equal(tmp[:tmp1.shape[1]], tmp1))
    imgs.close()
