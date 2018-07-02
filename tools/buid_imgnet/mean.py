# coding: utf-8
# Created by ay27 at 28/01/2018
import signal
import struct
from binary import ImageNet
import numpy as np
from tqdm import tqdm
import pickle
from multiprocessing import Queue, Process
import sys


def transform(p, q, dataset):
    sums = np.zeros((dataset.C * dataset.H * dataset.W), np.float64)
    cnt = 0
    while True:
        read_n, encoded = p.get(True)
        if (read_n is None) or (encoded is None):
            break
        batch = dataset.unpack(read_n, encoded)
        sums += np.sum(batch, axis=0, dtype=np.float64)
        # mean = mean * (float(cnt) / float(cnt + read_n)) + \
        #        (np.sum(batch, axis=0, dtype=np.float32) / float(cnt + read_n))
        cnt += batch.shape[0]

    q.put((cnt, sums))


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print('Argument Error!\n'
              'python2.7 mean.py [bin-format] [output] [#split] [#workers]\n'
              '\tbin-format: \ttrain binary file format, Example: train_data_{}.bin\n'
              '\toutput: \toutput file, Example: train_mean.bin\n'
              '\t#splits: \tnumber of original data split, Example: 1\n'
              '\t#workers: \tcpu process to compute, Example: 4\n')
        exit(0)

    bin_format = sys.argv[1]
    output_file = sys.argv[2]
    splits = int(sys.argv[3])
    workers = int(sys.argv[4])

    dataset_sum = None
    threads = []

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        q.close()
        p.close()
        for t in threads:
            t.terminate()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)

    C, H, W = 0, 0, 0
    cnt = 0.0

    for split in range(splits):
        print('processing split {}'.format(split))
        dataset = ImageNet(bin_format.format(split), 'B')
        dataset.read_meta()

        if dataset_sum is None:
            C = dataset.C
            H = dataset.H
            W = dataset.W
            dataset_sum = np.zeros([C * H * W], np.float64)

        p = Queue(1024)
        q = Queue(1024)

        for ii in range(workers):
            t = Process(target=transform, name='T-{}'.format(ii),
                        args=(p, q, dataset))
            threads.append(t)
            t.start()

        for ii in tqdm(range(dataset.N)):
            p.put(dataset.read(10))

        for t in threads:
            p.put((None, None))

        for ii in range(workers):
            processed_n, sums = q.get(True)
            dataset_sum += sums
            # mean = mean * (float(cnt) / float(cnt + processed_n)) + means * (
            #     float(processed_n) / float(cnt + processed_n))
            cnt += processed_n

        print('split #{} total processed n: {}'.format(split, cnt))

        for t in threads:
            t.join()
            t.terminate()
        threads = []

    mean = dataset_sum / float(cnt)
    mean = np.asarray(mean, np.float32)

    # write real mean file
    fmt = '<' + 'f' * (C * H * W)
    with open(output_file, 'wb') as f:
        f.write(struct.pack(fmt, *mean))

    mean = mean.reshape(C, H, W)
    pickle.dump(mean, open('{}pkl'.format(output_file), 'wb'), protocol=2)
    print(mean)
    print('channel mean:')
    print(np.sum(mean, (1, 2), dtype=np.float32) / float(H * W))
    print('finish')
