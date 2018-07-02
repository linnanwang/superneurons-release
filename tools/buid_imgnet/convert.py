# coding: utf-8
# Created by ay27 at 27/01/2018
import cv2
import numpy as np
import os
import time
from binary import ImageNet
from tqdm import tqdm
from multiprocessing import Queue, Process
import signal
import sys


def read_list(filename):
    files = []
    with open(filename, 'r') as f:
        for line in f:
            filepath, label = line.split()
            files.append([filepath, int(label)])
    return files


def save_list(filename, files):
    with open(filename, 'w') as f:
        for filepath, label in files:
            f.write('{} {}\n'.format(filepath, label))
    print('write list finish')


def transform_img(q, dataset, label, folder, fname, H, W):
    try:
        img = cv2.imread(os.path.join(folder, fname[0]), 1)
        img = cv2.resize(img, (H, W))
        # img = Image.open(os.path.join(folder, fname[0]))
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        # img = img.resize((H, W))
        img = np.asarray(img, np.uint8)

        img = img.transpose([2, 0, 1])
        assert img.shape == (3, H, W)

        batch, labels = dataset.pack(1, 3, H, W, img), label.pack(1, 1, 1, 1, [fname[1]])
    except:
        return fname
    else:
        q.put((batch, labels))
        return 1


def worker(q, index, total_workers, dataset, label, folder, files, H, W):
    start_idx = len(files) // total_workers * index
    end_idx = len(files) // total_workers * (index + 1)
    if index == total_workers - 1:
        end_idx = len(files)

    err_list = []
    for ii, fname in tqdm(enumerate(files[start_idx: end_idx]), total=end_idx - start_idx, desc='T-{}'.format(index),
                          position=index):
        res = transform_img(q, dataset, label, folder, fname, H, W)
        if not isinstance(res, int):
            err_list.append(res)

    for ii, fname in enumerate(err_list):
        res = transform_img(q, dataset, label, folder, fname, H, W)
        if not isinstance(res, int):
            print('err-file {} still error'.format(fname[0]))

    q.put([None, None])


def writer(dataset, label, H, W, length, q, total_worker):
    finish_cnt = 0
    while True:
        packed_img, packed_label = q.get(True)

        if (packed_img is None) or (label is None):
            finish_cnt += 1
            if finish_cnt == total_worker:
                break
            continue
        dataset.write(1, 3, H, W, packed_img)
        label.write(1, 1, 1, 1, packed_label)


if __name__ == '__main__':
    if len(sys.argv) != 10:
        print('Argument Error!\n'
              'python2.7 convert.py [src-folder] [list-file] [dst-folder] [prefix] '
              '[height] [width] [#split] [shuffle] [#workers]\n'
              '\tsrc-folder: \tImageNet train/val data directory, Example: ./data/train\n'
              '\tlist-file: \tImageNet train.txt or val.txt list file\n'
              '\tdst-folder: \toutput files directory, Example: ./data/output\n'
              '\tprefix: \toutput files prefix, Example: train\n'
              '\theight, width: \tresize images, Example: 256\n'
              '\t#splits: \tnumber of expected splits\n'
              '\tshuffle: \tshuffle or not, 1 for shuffle other for not\n'
              '\t#workers: \tcpu process to compute, Example: 4\n')
        exit(0)
    src_folder = sys.argv[1]
    list_file = sys.argv[2]
    dst_folder = sys.argv[3]
    prefix = sys.argv[4]
    height, width = int(sys.argv[5]), int(sys.argv[6])
    splits = int(sys.argv[7])
    shuffle = int(sys.argv[8]) == 1
    workers = int(sys.argv[9])

    files = read_list(list_file)
    print(len(files))

    if shuffle:
        np.random.seed(123456789)
        for ii in range(100):
            np.random.shuffle(files)
    save_list(os.path.join(dst_folder, '{}_list.txt'.format(prefix)), files)

    threads = []


    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        for t in threads:
            q.close()
            t.terminate()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()
    for ii in range(splits):
        dataset = ImageNet(os.path.join(dst_folder, '{}_data_{}.bin'.format(prefix, ii)), 'B')
        label = ImageNet(os.path.join(dst_folder, '{}_label_{}.bin'.format(prefix, ii)), 'i')

        start_idx = len(files) // splits * ii
        if ii == splits - 1:
            end_idx = len(files)
        else:
            end_idx = len(files) // splits * (ii + 1)

        q = Queue(1024)
        threads = []

        for k in range(workers):
            t = Process(target=worker, name='t%d' % k,
                        args=(q, k, workers, dataset, label,
                              src_folder, files[start_idx: end_idx],
                              height, width))
            t.start()
            threads.append(t)

        writer(dataset, label, height, width, end_idx - start_idx, q, workers)

        for t in threads:
            t.join()
            t.terminate()
        threads = []

        dataset.flush()
        label.flush()

        assert dataset.N == label.N
        print('success {}, failed {}'.format(dataset.N, end_idx - start_idx - dataset.N))

        dataset.reset_NCHW()
        label.reset_NCHW()
        dataset.close()
        label.close()

    print('finish, cost time: {} s, average speed {} img/s'.format(
        time.time() - start_time, len(files) / (time.time() - start_time)))
