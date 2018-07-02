#include <stdlib.h>
#include <superneurons.h>

using namespace SuperNeurons;

base_layer_t<float> *conv2_x(base_layer_t<float> *bottom, bool increase_dim) {
    // fork
    base_layer_t<float> *fork = (base_layer_t<float> *) new fork_layer_t<float>();
    // join
    base_layer_t<float> *join = (base_layer_t<float> *) new join_layer_t<float>();

    bottom->hook(fork);

    // left part
    if (increase_dim) {
        base_layer_t<float> *conv_left = (base_layer_t<float> *) new conv_layer_t<float>(256, 1, 1, 0, 0,
                                                                                         new xavier_initializer_t<float>(),
                                                                                         false);
        base_layer_t<float> *bn_left = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
                CUDNN_BATCHNORM_SPATIAL, 0.001);

        fork->hook(conv_left);
        conv_left->hook(bn_left);
        bn_left->hook(join);
    } else {
        fork->hook(join);
    }


    // right part
    base_layer_t<float> *conv_right1 = (base_layer_t<float> *) new conv_layer_t<float>(64, 1, 1, 0, 0,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right1 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_right1 = (base_layer_t<float> *) new act_layer_t<float>();

    base_layer_t<float> *conv_right2 = (base_layer_t<float> *) new conv_layer_t<float>(64, 3, 1, 1, 1,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right2 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_right2 = (base_layer_t<float> *) new act_layer_t<float>();

    base_layer_t<float> *conv_right3 = (base_layer_t<float> *) new conv_layer_t<float>(256, 1, 1, 0, 0,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right3 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);


    //right part
    fork->hook(conv_right1);
    conv_right1->hook(bn_right1);
    bn_right1->hook(act_right1);
    act_right1->hook(conv_right2);
    conv_right2->hook(bn_right2);
    bn_right2->hook(act_right2);
    act_right2->hook(conv_right3);
    conv_right3->hook(bn_right3);
    bn_right3->hook(join);

    base_layer_t<float> *act = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                              CUDNN_NOT_PROPAGATE_NAN);
    join->hook(act);
    return act;

}

base_layer_t<float> *conv3_x(base_layer_t<float> *bottom, bool increase_dim) {

    // fork
    base_layer_t<float> *fork = (base_layer_t<float> *) new fork_layer_t<float>();
    // join
    base_layer_t<float> *join = (base_layer_t<float> *) new join_layer_t<float>();

    bottom->hook(fork);

    if (increase_dim) {

        // left part
        base_layer_t<float> *conv_left = (base_layer_t<float> *) new conv_layer_t<float>(512, 1, 2, 0, 0,
                                                                                         new xavier_initializer_t<float>(),
                                                                                         false);
        base_layer_t<float> *bn_left = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
                CUDNN_BATCHNORM_SPATIAL, 0.001);
        fork->hook(conv_left);
        conv_left->hook(bn_left);
        bn_left->hook(join);
    } else {
        fork->hook(join);
    }


    // right part

    base_layer_t<float> *conv_right1;

    if (increase_dim) {
        conv_right1 = (base_layer_t<float> *) new conv_layer_t<float>(128, 1, 2, 0, 0,
                                                                      new xavier_initializer_t<float>(),
                                                                      false);
    } else {
        conv_right1 = (base_layer_t<float> *) new conv_layer_t<float>(128, 1, 1, 0, 0,
                                                                      new xavier_initializer_t<float>(),
                                                                      false);
    }


    base_layer_t<float> *bn_right1 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_right1 = (base_layer_t<float> *) new act_layer_t<float>();

    base_layer_t<float> *conv_right2 = (base_layer_t<float> *) new conv_layer_t<float>(128, 3, 1, 1, 1,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right2 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_right2 = (base_layer_t<float> *) new act_layer_t<float>();

    base_layer_t<float> *conv_right3 = (base_layer_t<float> *) new conv_layer_t<float>(512, 1, 1, 0, 0,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right3 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);


    //right part
    fork->hook(conv_right1);
    conv_right1->hook(bn_right1);
    bn_right1->hook(act_right1);
    act_right1->hook(conv_right2);
    conv_right2->hook(bn_right2);
    bn_right2->hook(act_right2);
    act_right2->hook(conv_right3);
    conv_right3->hook(bn_right3);
    bn_right3->hook(join);
    base_layer_t<float> *act = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                              CUDNN_NOT_PROPAGATE_NAN);
    join->hook(act);
    return act;

}

base_layer_t<float> *conv4_x(base_layer_t<float> *bottom, bool increase_dim) {
    // fork
    base_layer_t<float> *fork = (base_layer_t<float> *) new fork_layer_t<float>();
    // join
    base_layer_t<float> *join = (base_layer_t<float> *) new join_layer_t<float>();

    bottom->hook(fork);


    if (increase_dim) {
        // left part
        base_layer_t<float> *conv_left = (base_layer_t<float> *) new conv_layer_t<float>(1024, 1, 2, 0, 0,
                                                                                         new xavier_initializer_t<float>(),
                                                                                         false);
        base_layer_t<float> *bn_left = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
                CUDNN_BATCHNORM_SPATIAL, 0.001);
        fork->hook(conv_left);
        conv_left->hook(bn_left);
        bn_left->hook(join);
    } else {
        fork->hook(join);
    }


    // right part

    base_layer_t<float> *conv_right1;

    if (increase_dim) {
        conv_right1 = (base_layer_t<float> *) new conv_layer_t<float>(256, 1, 2, 0, 0,
                                                                      new xavier_initializer_t<float>(),
                                                                      false);
    } else {
        conv_right1 = (base_layer_t<float> *) new conv_layer_t<float>(256, 1, 1, 0, 0,
                                                                      new xavier_initializer_t<float>(),
                                                                      false);
    }

    base_layer_t<float> *bn_right1 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_right1 = (base_layer_t<float> *) new act_layer_t<float>();

    base_layer_t<float> *conv_right2 = (base_layer_t<float> *) new conv_layer_t<float>(256, 3, 1, 1, 1,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right2 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_right2 = (base_layer_t<float> *) new act_layer_t<float>();

    base_layer_t<float> *conv_right3 = (base_layer_t<float> *) new conv_layer_t<float>(1024, 1, 1, 0, 0,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right3 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);

    //right part
    fork->hook(conv_right1);
    conv_right1->hook(bn_right1);
    bn_right1->hook(act_right1);
    act_right1->hook(conv_right2);
    conv_right2->hook(bn_right2);
    bn_right2->hook(act_right2);
    act_right2->hook(conv_right3);
    conv_right3->hook(bn_right3);
    bn_right3->hook(join);
    base_layer_t<float> *act = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                              CUDNN_NOT_PROPAGATE_NAN);
    join->hook(act);
    return act;

}

base_layer_t<float> *conv5_x(base_layer_t<float> *bottom, bool increase_dim) {
    // fork
    base_layer_t<float> *fork = (base_layer_t<float> *) new fork_layer_t<float>();
    // join
    base_layer_t<float> *join = (base_layer_t<float> *) new join_layer_t<float>();

    bottom->hook(fork);

    // left part
    if (increase_dim) {
        base_layer_t<float> *conv_left = (base_layer_t<float> *) new conv_layer_t<float>(2048, 1, 2, 0, 0,
                                                                                         new xavier_initializer_t<float>(),
                                                                                         false);
        base_layer_t<float> *bn_left = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
                CUDNN_BATCHNORM_SPATIAL, 0.001);
        fork->hook(conv_left);
        conv_left->hook(bn_left);
        bn_left->hook(join);
    } else {
        fork->hook(join);
    }


    // right part
    base_layer_t<float> *conv_right1;

    if (increase_dim) {
        conv_right1 = (base_layer_t<float> *) new conv_layer_t<float>(512, 1, 2, 0, 0,
                                                                      new xavier_initializer_t<float>(),
                                                                      false);
    } else {
        conv_right1 = (base_layer_t<float> *) new conv_layer_t<float>(512, 1, 1, 0, 0,
                                                                      new xavier_initializer_t<float>(),
                                                                      false);
    }

    base_layer_t<float> *bn_right1 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_right1 = (base_layer_t<float> *) new act_layer_t<float>();

    base_layer_t<float> *conv_right2 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right2 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_right2 = (base_layer_t<float> *) new act_layer_t<float>();

    base_layer_t<float> *conv_right3 = (base_layer_t<float> *) new conv_layer_t<float>(2048, 1, 1, 0, 0,
                                                                                       new xavier_initializer_t<float>(),
                                                                                       false);
    base_layer_t<float> *bn_right3 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(
            CUDNN_BATCHNORM_SPATIAL, 0.001);



    //right part
    fork->hook(conv_right1);
    conv_right1->hook(bn_right1);
    bn_right1->hook(act_right1);
    act_right1->hook(conv_right2);
    conv_right2->hook(bn_right2);
    bn_right2->hook(act_right2);
    act_right2->hook(conv_right3);
    conv_right3->hook(bn_right3);
    bn_right3->hook(join);
    base_layer_t<float> *act = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                              CUDNN_NOT_PROPAGATE_NAN);
    join->hook(act);

    return act;

}


int main(int argc, char **argv) {
    char *train_label_bin;
    char *train_image_bin;
    char *test_label_bin;
    char *test_image_bin;
    char *train_mean_file;
    char *checkpoint;
    int lay_num = 50;

    if (lay_num < 50) {
        fprintf(stderr, "not implement for the layernum=%d\n", lay_num);
        exit(-1);
    }

    base_solver_t<float> *solver = (base_solver_t<float> *) new nesterov_solver_t<float>(0.1, 0.0004, 0.9);
    solver->set_lr_decay_policy(ITER, {500000, 1000000}, {0.01, 0.001});
    network_t<float> n(solver);

    train_mean_file = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/train_mean.bin";
    train_image_bin = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/train_data_0.bin";
    train_label_bin = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/train_label_0.bin";
    test_image_bin  = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/val_data_0.bin";
    test_label_bin  = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/val_label_0.bin";
    checkpoint      = (char *) "/data/lwang53/checkpoints/residual_50";

    network_saver* saver = new network_saver_impl<float>(checkpoint, n.get_registry());
    install_signal_processor(saver);

    const size_t batch_size = 36; //train and test must be same
    const size_t C = 3, H = 227, W = 227;
    
    preprocessor<float>* processor_train = new preprocessor<float>();
    processor_train->add_preprocess(new mean_subtraction_t<float>(batch_size, 3, 256, 256, train_mean_file));
    processor_train->add_preprocess(new random_crop_t<float>(batch_size, 3, 256, 256, batch_size, 3, H, W));
    processor_train->add_preprocess(new random_flip_left_right_t<float>(batch_size, 3, 227, 227));
    processor_train->add_preprocess(new per_image_standardization_t<float>(batch_size, C, H, W) );

    
    preprocessor<float>* processor_test = new preprocessor<float>();
    processor_test->add_preprocess(new mean_subtraction_t<float>(batch_size, 3, 256, 256, train_mean_file));
    processor_test->add_preprocess(new central_crop_t<float>(batch_size, 3, 256, 256, batch_size, 3, H, W));
    processor_train->add_preprocess(new per_image_standardization_t<float>(batch_size, C, H, W) );


    parallel_reader_t<float> reader2(test_image_bin, test_label_bin, 1, batch_size, C, H, W, processor_test, 1, 1);
    base_layer_t<float> *data_2 = (base_layer_t<float> *) new data_layer_t<float>(DATA_TEST, &reader2);
    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 4, batch_size, C, H, W, processor_train, 4, 2);
    base_layer_t<float> *data_1 = (base_layer_t<float> *) new data_layer_t<float>(DATA_TRAIN, &reader1);

    //if the dims of H,W after conv is not reduced, pad with half the filter sizes (round down). 3/2 = 1.5 = 1;
    base_layer_t<float> *conv_1 = (base_layer_t<float> *) new conv_layer_t<float>(64, 7, 2, 3, 3,
                                                                                  new xavier_initializer_t<float>(),
                                                                                  false);
    base_layer_t<float> *bn_1 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL,
                                                                                               0.001);
    base_layer_t<float> *act_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float> *pool_1 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);

    //setup test
    data_2->hook_to(conv_1);
    //setup network
    data_1->hook(conv_1);
    conv_1->hook(bn_1);
    bn_1->hook(act_1);
    act_1->hook(pool_1);

    base_layer_t<float> *net = pool_1;
    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            net = conv2_x(net, true);
        } else {
            net = conv2_x(net, false);
        }
    }

    for (int i = 0; i < 4; i++) {
        if (i == 0) {
            net = conv3_x(net, true);
        } else {
            net = conv3_x(net, false);
        }
    }

    for (int i = 0; i < 6; i++) {
        if (i == 0) {
            net = conv4_x(net, true);
        } else {
            net = conv4_x(net, false);
        }
    }

    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            net = conv5_x(net, true);
        } else {
            net = conv5_x(net, false);
        }
    }

    base_layer_t<float> *pool_2 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 1, 1, 7, 7);

    // 2048 x 1024
    base_layer_t<float> *full_conn_1 = (base_layer_t<float> *) new fully_connected_layer_t<float>(1000,
                                                                                                  new xavier_initializer_t<float>(),
                                                                                                  true);
    base_layer_t<float> *softmax = (base_layer_t<float> *) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE,
                                                                                      CUDNN_SOFTMAX_MODE_INSTANCE);


    net->hook(pool_2);
    pool_2->hook(full_conn_1);
    full_conn_1->hook(softmax);

    n.fsetup(data_1);
    n.bsetup(softmax);

    // the validation set has 50000 images.
    n.setup_test(data_2, 50000 / batch_size);
    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs / batch_size;

    //saver->load();

    n.train(600000, tracking_window, 1000);

    //saver->save();
}

