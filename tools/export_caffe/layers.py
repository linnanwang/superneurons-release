# Created by ay27 at 05/12/2017
import struct
import utils
import initializers


class Layer(object):
    #
    #      * value_type: layer_id
    #      * value_type: layer_type
    #      * value_type: previous layers count
    #      * value_type: next layers count
    #      * value_type: [pre layer id...], [next layer id...]
    #      */

    def __init__(self, layer_id, layer_type, pre_layers, next_layers):
        self.layer_id, self.layer_type, self.pre_layers, self.next_layers = layer_id, layer_type, pre_layers, next_layers

    @staticmethod
    def read_layer(bin_file, fmt, size_of_type):
        # read layer meta
        layer_id, layer_type, pre_cnt, next_cnt, = struct.unpack(fmt * 4, bin_file.read(size_of_type * 4))
        layer_id, layer_type, pre_cnt, next_cnt = int(layer_id), int(layer_type), int(pre_cnt), int(next_cnt)
        print layer_id, layer_type, pre_cnt, next_cnt
        pre_layers = struct.unpack(fmt * pre_cnt, bin_file.read(size_of_type * pre_cnt))
        pre_layers = list(pre_layers)
        next_layers = struct.unpack(fmt * next_cnt, bin_file.read(size_of_type * next_cnt))
        next_layers = list(next_layers)

        print pre_layers, next_layers

        print LayerType[layer_type].__name__
        args = LayerType[layer_type].read_from_binary(bin_file, fmt, size_of_type)
        return LayerType[layer_type](layer_id, layer_type, pre_layers, next_layers, *args)

    def gen_base_json(self, phase):
        res = dict()
        res['name'] = 'layer%d' % self.layer_id
        if len(self.pre_layers) > 0:
            res['bottom'] = ['layer%d' % _ for _ in self.pre_layers]
        res['top'] = ['layer%d' % self.layer_id]
        if phase is not None:
            res['phase'] = phase
        res['type'] = LayerCaffe[self.layer_type]
        return res

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        raise NotImplementedError()

    def to_json(self, phase):
        raise NotImplementedError()


class ConvLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers, num_out, kh, kw, sd, ph, pw, init, weight, bias):
        super(ConvLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)
        self.num_out, self.kh, self.kw, self.stride, self.ph, self.pw, self.init = num_out, kh, kw, sd, ph, pw, init
        self.weight = weight
        self.bias = bias

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        # num_output
        # kernel_h, kernel_w;
        # stride;
        # padding_h, padding_w;
        # weight_initializer
        # tensor_t: weight, bias
        num_out, kh, kw, sd, ph, pw, init, = struct.unpack(fmt * 7, bin_file.read(size_of_type * 7))
        num_out, kh, kw, sd, ph, pw, init = int(num_out), int(kh), int(kw), int(sd), int(ph), int(pw), int(init)
        print num_out, kh, kw, sd, ph, pw, init

        weight, layer_id1, tensor_type1 = utils.read_tensor(bin_file, fmt, size_of_type)
        bias, layer_id2, tensor_type2 = utils.read_tensor(bin_file, fmt, size_of_type)

        print weight.shape, bias.shape

        return [num_out, kh, kw, sd, ph, pw, init, weight, bias]

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        convolutionParam = dict()
        convolutionParam['weightFiller'] = dict()
        convolutionParam['weightFiller']['type'] = initializers.InitCaffe[self.init]
        if self.kh == self.kw:
            convolutionParam['kernelSize'] = [self.kh]
        else:
            convolutionParam['kernelSize'] = [self.kh, self.kw]
        convolutionParam['numOutput'] = self.num_out
        convolutionParam['stride'] = [self.stride]
        if self.ph == self.pw:
            convolutionParam['pad'] = [self.ph]
        else:
            convolutionParam['pad'] = [self.ph, self.pw]

        base['convolutionParam'] = convolutionParam

        return base


class PoolLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers, mode, vs, hs, kh, kw, vp, hp):
        super(PoolLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)
        self.mode, self.vs, self.hs, self.kh, self.kw, self.vp, self.hp = mode, vs, hs, kh, kw, vp, hp

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        # value_type: mode, vs, hs, kh, kw, vp, hp
        mode, vs, hs, kh, kw, vp, hp, = struct.unpack(fmt * 7, bin_file.read(size_of_type * 7))
        mode, vs, hs, kh, kw, vp, hp = int(mode), int(vs), int(hs), int(kh), int(kw), int(vp), int(hp)
        print mode, vs, hs, kh, kw, vp, hp

        return [mode, vs, hs, kh, kw, vp, hp]

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        poolingParam = dict()
        poolingParam['pool'] = utils.PoolCaffe[self.mode]
        if self.vs == self.hs:
            poolingParam['stride'] = self.vs
        else:
            poolingParam['stride'] = [self.vs, self.hs]
        if self.kh == self.kw:
            poolingParam['kernelSize'] = self.kh
        else:
            poolingParam['kernelSize'] = [self.kh, self.kw]
        if self.vp == self.hp:
            poolingParam['pad'] = self.vp
        else:
            poolingParam['pad'] = [self.vp, self.hp]

        base['poolingParam'] = poolingParam
        return base


class ActLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers, mode):
        super(ActLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)
        self.mode = mode

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        mode, = struct.unpack(fmt, bin_file.read(size_of_type))
        mode = int(mode)
        print utils.ActCaffe[mode]

        return [mode]

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        base['type'] = utils.ActCaffe[self.mode]
        return base


class BNLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers, eps, mode, weight, bias):
        super(BNLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)
        self.eps, self.mode, self.weight, self.bias = eps, mode, weight, bias

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        # / **
        # *value_type: epsilon
        # *value_type: BN mode
        # *tensor_t: weight gamma
        # *tensor_t: bias beta
        # * /
        eps, mode, = struct.unpack(fmt * 2, bin_file.read(size_of_type * 2))
        mode = int(mode)
        weight, layer_id1, tensor_type1 = utils.read_tensor(bin_file, fmt, size_of_type)
        bias, layer_id2, tensor_type2 = utils.read_tensor(bin_file, fmt, size_of_type)
        print eps, mode, weight.shape, bias.shape

        return [eps, mode, weight, bias]

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        return base


class FCLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers, out_dim, weight, bias):
        super(FCLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)
        self.out_dim, self.weight, self.bias = out_dim, weight, bias

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        # out, init, weight, bias
        out_dim, init, = struct.unpack(fmt * 2, bin_file.read(size_of_type * 2))
        out_dim, init = int(out_dim), int(init)
        weight, layer_id1, tensor_type1 = utils.read_tensor(bin_file, fmt, size_of_type)
        bias, layer_id2, tensor_type2 = utils.read_tensor(bin_file, fmt, size_of_type)
        print out_dim, init, weight.shape, bias.shape

        return [out_dim, weight, bias]

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        return base


class LRNLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers):
        super(LRNLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        return []

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        return base


class PadLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers, pc, ph, pw):
        super(PadLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)
        self.pc, self.ph, self.pw = pc, ph, pw

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        pc, ph, pw, = struct.unpack(fmt * 3, bin_file.read(size_of_type * 3))
        pc, ph, pw = int(pc), int(ph), int(pw)
        print pc, ph, pw

        return [pc, ph, pw]

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        return base


class DataLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers):
        super(DataLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        return []

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        return base


class DropoutLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers, drop_rate):
        super(DropoutLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)
        self.drop_rate = drop_rate

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        drop_rate, = struct.unpack(fmt, bin_file.read(size_of_type))
        print drop_rate

        return [drop_rate]

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        return base


class SoftmaxLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers):
        super(SoftmaxLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        return []

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        return base


class ConcatLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers):
        super(ConcatLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        return []

    def to_json(self, phase):
        base = self.gen_base_json(phase)
        return base


class ForkLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers):
        super(ForkLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        return []

    def to_json(self, phase):
        pass


class JoinLayer(Layer):
    def __init__(self, layer_id, layer_type, pre_layers, next_layers):
        super(JoinLayer, self).__init__(layer_id, layer_type, pre_layers, next_layers)

    @staticmethod
    def read_from_binary(bin_file, fmt, size_of_type):
        return []

    def to_json(self, phase):
        pass


# typedef enum LAYER {
#     /*---network layers---*/
#     CONV    = 0,
#     POOL    = 1,
#     ACT     = 2,
#     BN      = 3,
#     FC      = 4,
#     LRN     = 5,
#     PADDING = 6,
#     DATA_L  = 7,
#     DROPOUT = 8,
#     SOFTMAX = 9,
#     /*--structure layers--*/
#     CONCAT  = 10,
#     FORK_L  = 11,
#     JOIN_L  = 12
# } LAYER;
LayerType = [ConvLayer, PoolLayer, ActLayer,
             BNLayer, FCLayer, LRNLayer,
             PadLayer, DataLayer, DropoutLayer,
             SoftmaxLayer, ConcatLayer, ForkLayer, JoinLayer]
LayerCaffe = ['Convolution', 'Pooling', 'ReLU',
              'BatchNorm', 'InnerProduct', 'LRN',
              'Padding', 'Data', 'Dropout',
              'SoftmaxWithLoss', 'Concat', 'Fork', 'Join']
