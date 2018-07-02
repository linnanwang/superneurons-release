# Created by ay27 at 05/12/2017
import struct
import numpy as np

ActCaffe = ['Sigmoid', 'ReLU', 'Tanh', 'ReLU']
TensorType = ['DATA', 'GRAD', 'PARAM', 'AUX', 'BN_MEAN_VAR', 'CONV_BUFF', 'DATA_SOURCE']
SolverType = ['MOMENTUM', 'SGD', 'NESTEROV', 'ADAGRAD', 'RMSPROP']
PoolCaffe = ['MAX', 'AVE', 'AVE']

def read_tensor(bin_file, fmt, size_of_type):
    # NCHW, layer_id, data_type, data
    N, C, H, W, layer_id, tensor_type, = struct.unpack(fmt * 6, bin_file.read(size_of_type * 6))
    N, C, H, W, layer_id, tensor_type = int(N), int(C), int(H), int(W), int(layer_id), int(tensor_type)
    print N, C, H, W, layer_id, tensor_type

    data = np.zeros(N * C * H * W)
    for ii in range(N * C * H * W):
        data[ii], = struct.unpack(fmt, bin_file.read(size_of_type))
    data = data.reshape((N, C, H, W))
    return data, layer_id, tensor_type


# typedef enum lr_decay_type {
#     ITER = 0,
#     LOSS = 1
# } lr_decay_type;
#
# typedef enum solver_type {
#     MOMENTUM = 0,
#     SGD = 1,
#     NESTEROV = 2,
#     ADAGRAD = 3,
#     RMSPROP = 4
# } solver_type;

def read_solver(bin_file, fmt, size_of_type):
    # / **
    # *value_type: solver_type
    # *value_type: lr
    # *value_type: weight_decay
    # *value_type: decay_type
    # *value_type: policy_len
    # *value_type: {{point, lr}, {point, lr}, ...}
    # *value_type: param_cnt
    # *value_type: extra params
    # * /
    solver_type, lr, weight_decay, decay_type, policy_len, = struct.unpack(fmt * 5, bin_file.read(size_of_type * 5))
    solver_type = int(solver_type)
    decay_type = int(decay_type)
    policy_len = int(policy_len)
    print solver_type, lr, weight_decay, decay_type, policy_len

    # read policy
    policy = []
    for ii in range(policy_len):
        _point, _lr, = struct.unpack(fmt * 2, bin_file.read(size_of_type * 2))
        policy.append([_point, _lr])

    print policy

    extra_params_cnt, = struct.unpack(fmt, bin_file.read(size_of_type))

    if solver_type == 0:
        assert extra_params_cnt == 1
        m, = struct.unpack(fmt, bin_file.read(size_of_type))
        print 'momentum ', m
    elif solver_type == 1:
        print 'SGD'
        assert extra_params_cnt == 0
    elif solver_type == 2:
        assert extra_params_cnt == 1
        m, = struct.unpack(fmt, bin_file.read(size_of_type))
        print 'NESTEROV ', m
    elif solver_type == 3:
        assert extra_params_cnt == 1
        eps, = struct.unpack(fmt, bin_file.read(size_of_type))
        print 'ADAGRAD ', eps
    elif solver_type == 4:
        assert extra_params_cnt == 2
        eps, rms_decay, = struct.unpack(fmt * 2, bin_file.read(size_of_type * 2))
        print 'RMSPROP ', eps, rms_decay
    else:
        raise ValueError('unrecognized solver type %d' % solver_type)
