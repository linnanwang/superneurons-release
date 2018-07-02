# Created by ay27 at 05/12/2017
caffe_root = '/home/ay27/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys

sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf.json_format as jsf
import json
import sys
import struct
import numpy as np
import layers as LL
import utils


if __name__ == '__main__':
    # file_path = sys.argv[1]
    file_path = '/home/ay27/superneurons/build/lenet.dump'

    bin_file = open(file_path, 'rb')

    #############################################################################
    # read meta
    size_of_type, = struct.unpack('B', bin_file.read(1))
    layer_cnt, = struct.unpack('I', bin_file.read(4))
    print size_of_type, layer_cnt

    if size_of_type == 4:
        # float
        fmt = 'f'
    elif size_of_type == 8:
        # double
        fmt = 'd'
    else:
        raise ValueError('unrecognized data format %d' % size_of_type)

    #############################################################################
    # read solver
    utils.read_solver(bin_file, fmt, size_of_type)

    #############################################################################
    # read network
    layers = []
    for ii in range(layer_cnt):
        print '--------------------'
        layers.append(LL.Layer.read_layer(bin_file, fmt, size_of_type))

    print len(layers)

    j_layers = []
    for l in layers:
        j_layers.append(l.to_json('TRAIN'))

    net = dict()
    net['layer'] = j_layers
    print json.dumps(net)

    x = caffe_pb2.NetParameter()
    jsf.Parse(json.dumps(net), x)

    print x

    # with open('lenet.prototxt', 'w') as f:
    #     f.write(str(x))
