//
// Created by ay27 on 17/6/20.
//

#include <stdlib.h>
#include <superneurons.h>
using namespace SuperNeurons;


base_layer_t<float>* conv_bn_relu(base_layer_t<float>* bottom, size_t outnums,
                      size_t kernel_h, size_t kernel_w,
                      size_t stride,
                      size_t pad_h, size_t pad_w) {
    base_layer_t<float>* conv   = (base_layer_t<float> *) new conv_layer_t<float>(outnums, kernel_h, kernel_w, stride, pad_h, pad_w, new xavier_initializer_t<float>(), false);
//    base_layer_t<float>* bn     = (base_layer_t<float> *) new batch_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float>* relu   = (base_layer_t<float> *) new act_layer_t<float>();

    bottom->hook(conv);
    conv->hook(relu);
//    conv->hook(bn);
//    bn->hook(relu);

    return relu;
}

base_layer_t<float>* inception_a(base_layer_t<float>* bottom) {
    base_layer_t<float>* fork = (base_layer_t<float>*) new fork_layer_t<float>();
    base_layer_t<float>* concat = (base_layer_t<float>*) new concat_layer_t<float>();

    bottom->hook(fork);

    base_layer_t<float>* net;

    // branch 1
    // inception_a1_1x1_2
    net = conv_bn_relu(fork, 96, 1, 1, 1, 0, 0);
    net->hook(concat);

    // branch 2
    // inception_a1_3x3_reduce
    net = conv_bn_relu(fork, 64, 1, 1, 1, 0, 0);
    // inception_a1_3x3
    net = conv_bn_relu(net, 96, 3, 3, 1, 1, 1);
    net->hook(concat);

    // branch 3
    // inception_a1_3x3_2_reduce
    net = conv_bn_relu(fork, 64, 1, 1, 1, 0, 0);
    // inception_a1_3x3_2
    net = conv_bn_relu(net, 96, 3, 3, 1, 1, 1);
    // inception_a1_3x3_3
    net = conv_bn_relu(net, 96, 3, 3, 1, 1, 1);
    net->hook(concat);

    // branch 4
    base_layer_t<float>* pool = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 1, 1, 3, 3, 1, 1);
    fork->hook(pool);
    net = conv_bn_relu(pool, 96, 1, 1, 1, 0, 0);
    net->hook(concat);

    return concat;
}

base_layer_t<float>* inception_b(base_layer_t<float>* bottom) {
    base_layer_t<float>* fork = (base_layer_t<float>*) new fork_layer_t<float>();
    base_layer_t<float>* concat = (base_layer_t<float>*) new concat_layer_t<float>();

    bottom->hook(fork);

    base_layer_t<float> *net;

    // branch 1
    // inception_b1_1x1_2
    net = conv_bn_relu(fork, 384, 1, 1, 1, 0, 0);
    net->hook(concat);

    // branch 2
    // inception_b1_1x7_reduce
    net = conv_bn_relu(fork, 192, 1, 1, 1, 0, 0);
    // inception_b1_1x7
    net = conv_bn_relu(net, 224, 1, 7, 1, 0, 3);
    // inception_b1_7x1
    net = conv_bn_relu(net, 256, 7, 1, 1, 3, 0);
    net->hook(concat);

    // branch 3
    // inception_b1_7x1_2_reduce
    net = conv_bn_relu(fork, 192, 1, 1, 1, 0, 0);
    // inception_b1_7x1_2
    net = conv_bn_relu(net, 192, 7, 1, 1, 3, 0);
    // inception_b1_1x7_2
    net = conv_bn_relu(net, 224, 1, 7, 1, 0, 3);
    // inception_b1_7x1_3
    net = conv_bn_relu(net, 224, 7, 1, 1, 3, 0);
    // inception_b1_1x7_3
    net = conv_bn_relu(net, 256, 1, 7, 1, 0, 3);
    net->hook(concat);

    // branch 4
    // inception_b1_pool_ave
    base_layer_t<float>* pool = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 1, 1, 3, 3, 1, 1);
    // inception_b1_1x1
    fork->hook(pool);
    net = conv_bn_relu(pool, 128, 1, 1, 1, 0, 0);
    net->hook(concat);

    return concat;
}

base_layer_t<float>* inception_c(base_layer_t<float>* bottom) {
    base_layer_t<float>* fork = (base_layer_t<float>*) new fork_layer_t<float>();
    base_layer_t<float>* concat = (base_layer_t<float>*) new concat_layer_t<float>();

    bottom->hook(fork);

    base_layer_t<float> *net;

    // branch 1
    // inception_c1_1x1_2
    net = conv_bn_relu(fork, 256, 1, 1, 1, 0, 0);
    net->hook(concat);

    //branch 3
    // inception_c1_1x1_3
    net = conv_bn_relu(fork, 384, 1, 1, 1, 0, 0);
    base_layer_t<float>* fork_3 = (base_layer_t<float>*) new fork_layer_t<float>();
    net->hook(fork_3);
    // inception_c1_1x3
    net = conv_bn_relu(fork_3, 256, 1, 3, 1, 0, 1);
    net->hook(concat);
    // inception_c1_3x1
    net = conv_bn_relu(fork_3, 256, 3, 1, 1, 1, 0);
    net->hook(concat);

    // branch 4
    // inception_c1_1x1_4
    net = conv_bn_relu(fork, 384, 1, 1, 1, 0, 0);
    // inception_c1_3x1_2
    net = conv_bn_relu(net, 448, 3, 1, 1, 1, 0);
    // inception_c1_1x3_2
    net = conv_bn_relu(net, 512, 1, 3, 1, 0, 1);
    base_layer_t<float>* fork_4 = (base_layer_t<float>*) new fork_layer_t<float>();
    net->hook(fork_4);
    // inception_c1_1x3_3
    net = conv_bn_relu(fork_4, 256, 1, 3, 1, 0, 1);
    net->hook(concat);
    // inception_c1_3x1_3
    net = conv_bn_relu(fork_4, 256, 3, 1, 1, 1, 0);
    net->hook(concat);

    // branch 5
    // inception_c1_pool_ave
    base_layer_t<float>* inception_c1_pool_ave = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 1, 1, 3, 3, 1, 1);
    fork->hook(inception_c1_pool_ave);
    // inception_c1_1x1
    net = conv_bn_relu(inception_c1_pool_ave, 256, 1, 1, 1, 0, 0);
    net->hook(concat);

    return concat;

}

int main(int argc, char **argv) {

    char *train_label_bin;
    char *train_image_bin;
    char *test_label_bin;
    char *test_image_bin;
//    char *train_mean_file;
//    char *checkpoint;

//    train_mean_file = (char *) "/home/ay27/dataset/299/imgnet_train.mean";
    train_image_bin = (char *) "/home/ay27/dataset/299/val_data_0.bin";
    train_label_bin = (char *) "/home/ay27/dataset/299/val_label_0.bin";
    test_image_bin  = (char *) "/home/ay27/dataset/299/val_data_0.bin";
    test_label_bin  = (char *) "/home/ay27/dataset/299/val_label_0.bin";
//    checkpoint = (char*) "/storage/dataset/imagenet2012/299/checkpoint";
//    train_mean_file = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_image.mean";
//    train_image_bin = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_image_0.bin";
//    train_label_bin = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_label_0.bin";
//    test_image_bin  = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_test_image_0.bin";
//    test_label_bin  = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_test_label_0.bin";
    const size_t batch_size = static_cast<const size_t>(atoi(argv[1])); //train and test must be same

    const size_t C=3, H=299, W=299;

    float channel_mean[3] = {104, 117, 123};

    preprocessor<float>* p = new preprocessor<float>();
    p->add_preprocess(new mean_subtraction_t<float>(batch_size, C, H, W, channel_mean));
//     ->add_preprocess(new border_padding_t<float>(batch_size, C, H, W, 4, 4))
//     ->add_preprocess(new random_crop_t<float>(batch_size, C, H+8, W+8, batch_size, C, H, W))
//     ->add_preprocess(new random_flip_left_right_t<float>(batch_size, C, H, W));

#ifdef BLASX_MALLOC
    parallel_reader_t<float> reader2(test_image_bin, test_label_bin, 1, batch_size,C,H,W, p, 1, 1);
    base_layer_t<float> *data_test = (base_layer_t<float> *) new data_layer_t<float>(DATA_TEST, &reader2);
    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 4, batch_size,C,H,W, p, 4, 2);
    base_layer_t<float> *data_train = (base_layer_t<float> *) new data_layer_t<float>(DATA_TRAIN, &reader1);
#else
    parallel_reader_t<float> reader2(test_image_bin, test_label_bin, 2, batch_size, C, H, W, p, 1, 1);
    base_layer_t<float> *data_test = (base_layer_t<float> *) new data_layer_t<float>(DATA_TEST, &reader2);
    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 2, batch_size, C, H, W, p, 1, 1);
    base_layer_t<float> *data_train = (base_layer_t<float> *) new data_layer_t<float>(DATA_TRAIN, &reader1);
#endif


    /*--------------network configuration--------------*/

    base_solver_t<float> * solver = (base_solver_t<float> *) (new momentum_solver_t<float>(0.1, 0.0001, 0.9));
    solver->set_lr_decay_policy(ITER, {500000, 1000000}, {0.01, 0.001});
    network_t<float> n(solver);


    base_layer_t<float>* conv1_3x3_s2           = (base_layer_t<float> *) new conv_layer_t<float>(32, 3, 2, 0,0, new xavier_initializer_t<float>(), false);
    base_layer_t<float>* conv1_3x3_s2_bn        = (base_layer_t<float> *) new batch_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float>* conv1_3x3_s2_relu      = (base_layer_t<float> *) new act_layer_t<float>();

    data_test->hook_to(conv1_3x3_s2);
    data_train->hook(conv1_3x3_s2);
    conv1_3x3_s2->hook(conv1_3x3_s2_bn);
    conv1_3x3_s2_bn->hook(conv1_3x3_s2_relu);

    base_layer_t<float>* net;

    // conv2_3x3_s1
    net = conv_bn_relu(conv1_3x3_s2_relu, 32, 3, 3, 1, 0, 0);
    // conv3_3x3_s1
    net = conv_bn_relu(net, 64, 3, 3, 1, 1, 1);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // stem1
    // fork
    base_layer_t<float>* inception_stem0_fork = (base_layer_t<float> *) new fork_layer_t<float>();
    // concat
    base_layer_t<float>* inception_stem1      = (base_layer_t<float> *) new concat_layer_t<float>();

    net->hook(inception_stem0_fork);

    // left
    net = conv_bn_relu(inception_stem0_fork, 96, 3, 3, 2, 0, 0);
    net->hook(inception_stem1);

    // right
    base_layer_t<float>* inception_stem1_pool   = (base_layer_t<float> *) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);

    inception_stem0_fork->hook(inception_stem1_pool);
    inception_stem1_pool->hook(inception_stem1);

    net = inception_stem1;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // stem2

    // fork
    base_layer_t<float>* inception_stem1_fork   = (base_layer_t<float> *) new fork_layer_t<float>();
    // concat
    base_layer_t<float>* inception_stem2        = (base_layer_t<float> *) new concat_layer_t<float>();

    net->hook(inception_stem1_fork);

    // left
    net = conv_bn_relu(inception_stem1_fork, 64, 1, 1, 1, 0, 0);
    net = conv_bn_relu(net, 96, 3, 3, 1, 0, 0);
    net->hook(inception_stem2);

    // right
    net = conv_bn_relu(inception_stem1_fork, 64, 1, 1, 1, 0, 0);
    net = conv_bn_relu(net, 64, 1, 7, 1, 0, 3);
    net = conv_bn_relu(net, 64, 7, 1, 1, 3, 0);
    net = conv_bn_relu(net, 96, 3, 3, 1, 0, 0);
    net->hook(inception_stem2);

    net = inception_stem2;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // stem3
    // fork
    base_layer_t<float>* inception_stem2_fork   = (base_layer_t<float> *) new fork_layer_t<float>();
    // concat
    base_layer_t<float>* inception_stem3        = (base_layer_t<float> *) new concat_layer_t<float>();

    net->hook(inception_stem2_fork);

    // left
    net = conv_bn_relu(inception_stem2_fork, 192, 3, 3, 2, 0, 0);
    net->hook(inception_stem3);

    // right
    base_layer_t<float> *inception_stem3_pool = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    inception_stem2_fork->hook(inception_stem3_pool);
    inception_stem3_pool->hook(inception_stem3);

    net = inception_stem3;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // inception A

    for (int i = 0; i < 4; ++i) {
        net = inception_a(net);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reduction_a_concat

    base_layer_t<float>* inception_a4_fork  = (base_layer_t<float>*) new fork_layer_t<float>();
    base_layer_t<float>* reduction_a_concat = (base_layer_t<float>*) new concat_layer_t<float>();

    net->hook(inception_a4_fork);

    // branch 1
    // reduction_a_3x3
    net = conv_bn_relu(inception_a4_fork, 384, 3, 3, 2, 0, 0);
    net->hook(reduction_a_concat);

    // branch 2
    // reduction_a_3x3_2_reduce
    net = conv_bn_relu(inception_a4_fork, 192, 1, 1, 1, 0, 0);
    // reduction_a_3x3_2
    net = conv_bn_relu(net, 224, 3, 3, 1, 1, 1);
    // reduction_a_3x3_3
    net = conv_bn_relu(net, 256, 3, 3, 2, 0, 0);
    net->hook(reduction_a_concat);

    // branch 3
    // reduction_a_pool
    base_layer_t<float>* reduction_a_pool = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    inception_a4_fork->hook(reduction_a_pool);
    reduction_a_pool->hook(reduction_a_concat);

    net = reduction_a_concat;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // inception B

    for (int i = 0; i < 7; ++i) {
        net = inception_b(net);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // reduction_b_concat
    base_layer_t<float>* inception_b7_fork  = (base_layer_t<float>*) new fork_layer_t<float>();
    base_layer_t<float>* reduction_b_concat = (base_layer_t<float>*) new concat_layer_t<float>();

    net->hook(inception_b7_fork);

    // branch 1
    // reduction_b_3x3_reduce
    net = conv_bn_relu(inception_b7_fork, 192, 1, 1, 1, 0, 0);
    // reduction_b_3x3
    net = conv_bn_relu(net, 192, 3, 3, 2, 0, 0);
    net->hook(reduction_b_concat);

    // branch 2
    // reduction_b_1x7_reduce
    net = conv_bn_relu(inception_b7_fork, 256, 1, 1, 1, 0, 0);
    // reduction_b_1x7
    net = conv_bn_relu(net, 256, 1, 7, 1, 0, 3);
    // reduction_b_7x1
    net = conv_bn_relu(net, 320, 7, 1, 1, 3, 0);
    // reduction_b_3x3_2
    net = conv_bn_relu(net, 320, 3, 3, 2, 0, 0);
    net->hook(reduction_b_concat);

    // branch 3
    // reduction_b_pool
    base_layer_t<float>* reduction_b_pool = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    inception_b7_fork->hook(reduction_b_pool);
    reduction_b_pool->hook(reduction_b_concat);

    net = reduction_b_concat;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // inception C
    for (int i = 0; i < 3; ++i) {
        net = inception_c(net);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // global pooling
    // TODO : kernel size and stride
    base_layer_t<float>* pool_8x8_s1    = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 1, 1, 8, 8);
    base_layer_t<float>* drop           = (base_layer_t<float>*) new dropout_layer_t<float>(0.2);
    base_layer_t<float>* fc             = (base_layer_t<float>*) new fully_connected_layer_t<float>(1000, new xavier_initializer_t<float>(), true);
    base_layer_t<float>* softmax        = (base_layer_t<float>*) new softmax_layer_t<float>();

    net->hook(pool_8x8_s1);
    pool_8x8_s1->hook(drop);
    drop->hook(fc);
    fc->hook(softmax);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    n.fsetup(data_train);
    n.bsetup(softmax);

    n.setup_test(data_test, 50000 / batch_size);
    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs / batch_size;
    n.train(10, tracking_window, 4000);
}
