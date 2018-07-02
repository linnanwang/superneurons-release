#include <stdlib.h>
#include<math.h>
#include <superneurons.h>

using namespace SuperNeurons;


int main(int argc, char **argv) {
    char *train_label_bin;
    char *train_image_bin;
    char *test_label_bin;
    char *test_image_bin;
//    char *checkpoint;

//    train_mean_file = (char *) "/home/ay27/dataset/bin_file/imgnet_train.mean";
    train_image_bin = (char *) "/home/ay27/dataset/bin_file/train_data_0.bin";
    train_label_bin = (char *) "/home/ay27/dataset/bin_file/train_label_0.bin";
    test_image_bin  = (char *) "/home/ay27/dataset/bin_file/val_data_0.bin";
    test_label_bin  = (char *) "/home/ay27/dataset/bin_file/val_label_0.bin";
//    checkpoint_file = (char *) "/home/ay27/dataset/bin_file/alexnet_checkpoint";

//    train_image_bin = (char *) "/home/wwu/DeepLearning/data/imgnet/imgnet_val_data_0.bin";
//    train_label_bin = (char *) "/home/wwu/DeepLearning/data/imgnet/imgnet_val_label_0.bin";
//    test_image_bin  = (char *) "/home/wwu/DeepLearning/data/imgnet/imgnet_val_data_0.bin";
//    test_label_bin  = (char *) "/home/wwu/DeepLearning/data/imgnet/imgnet_val_label_0.bin";
//    checkpoint      = (char *) "/uac/ascstd/jmye/storage/superneurons/val100/checkpoint";


    base_solver_t<float>* solver = (base_solver_t<float>*) new nesterov_solver_t<float>(0.01, 0.0005, 0.9);
    solver->set_lr_decay_policy(ITER, {100000, 200000, 300000}, {0.001, 0.0001, 0.00001});
    network_t<float> n(solver);

//    network_saver* saver = new network_saver_impl<float>(checkpoint, n.get_registry());
//    install_signal_processor(saver);


    const size_t batch_size = static_cast<const size_t>(atoi(argv[1])); //train and test must be same
    const size_t C = 3, H = 227, W = 227;


    base_preprocess_t<float> *flip = (base_preprocess_t<float> *) new random_flip_left_right_t<float>(
            batch_size, C, H, W);
    base_preprocess_t<float> *standardization =
            (base_preprocess_t<float> *) new per_image_standardization_t<float>(
                    batch_size, C, H, W);

    float channel_mean[3] = {103.939, 116.779, 123.68};

    base_preprocess_t<float> *mean_sub =
            (base_preprocess_t<float> *) new mean_subtraction_t<float>(batch_size, C, H, W, channel_mean);

    preprocessor<float> *processor = new preprocessor<float>();
    processor->add_preprocess(mean_sub)
            ->add_preprocess(flip)
            ->add_preprocess(standardization);


    parallel_reader_t<float> reader2(test_image_bin, test_label_bin, 1, batch_size, C,H,W, processor, 1, 1);
    base_layer_t<float> *data_2 = (base_layer_t<float> *) new data_layer_t<float>(DATA_TEST, &reader2);

    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 4, batch_size,C,H,W, processor, 4, 2);
    base_layer_t<float> *data_1 = (base_layer_t<float> *) new data_layer_t<float>(DATA_TRAIN, &reader1);

    //if the dims of H,W after conv is not reduced, pad with half the filter sizes (round down). 3/2 = 1.5 = 1;
    base_layer_t<float> *conv1_1 = (base_layer_t<float> *) new conv_layer_t<float>(64, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01),
                                                                                   true);
    base_layer_t<float> *act1_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv1_2 = (base_layer_t<float> *) new conv_layer_t<float>(64, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01),
                                                                                   true);
    base_layer_t<float> *act1_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *pool_1 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);


    base_layer_t<float> *conv2_1 = (base_layer_t<float> *) new conv_layer_t<float>(128, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01),
                                                                                   true);
    base_layer_t<float> *act2_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv2_2 = (base_layer_t<float> *) new conv_layer_t<float>(128, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01),
                                                                                   true);
    base_layer_t<float> *act2_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *pool_2 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);



    base_layer_t<float> *conv3_1 = (base_layer_t<float> *) new conv_layer_t<float>(256, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act3_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv3_2 = (base_layer_t<float> *) new conv_layer_t<float>(256, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act3_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv3_3 = (base_layer_t<float> *) new conv_layer_t<float>(256, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act3_3 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *pool_3 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);



    base_layer_t<float> *conv4_1 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act4_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv4_2 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act4_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv4_3 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act4_3 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float> *pool_4 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);



    base_layer_t<float> *conv5_1 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act5_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv5_2 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act5_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv5_3 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act5_3 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *pool_5 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);




    base_layer_t<float> *full_conn_1 = (base_layer_t<float> *) new fully_connected_layer_t<float>(4096, new gaussian_initializer_t<float>(0, 0.01), true);
    base_layer_t<float> *relu6       = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *drop6       = (base_layer_t<float> *) new dropout_layer_t<float>(0.5);

    base_layer_t<float> *full_conn_2 = (base_layer_t<float> *) new fully_connected_layer_t<float>(4096, new gaussian_initializer_t<float>(0, 0.01),
                                                                                                  true);
    base_layer_t<float> *relu7       = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *drop7       = (base_layer_t<float> *) new dropout_layer_t<float>(0.5);

    base_layer_t<float> *full_conn_3 = (base_layer_t<float> *) new fully_connected_layer_t<float>(1000, new gaussian_initializer_t<float>(0, 0.01),
                                                                                                  true);
    base_layer_t<float> *softmax = (base_layer_t<float> *) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE,
                                                                                      CUDNN_SOFTMAX_MODE_INSTANCE);

    //setup test
    data_2->hook_to(conv1_1);
    //setup network
    data_1->hook(conv1_1);


    conv1_1->hook(act1_1);
    act1_1->hook(conv1_2);
    conv1_2->hook(act1_2);
    act1_2->hook(pool_1);
    pool_1->hook(conv2_1);





    conv2_1->hook(act2_1);
    act2_1->hook(conv2_2);
    conv2_2->hook(act2_2);
    act2_2->hook(pool_2);
    pool_2->hook(conv3_1);





    conv3_1->hook(act3_1);
    act3_1->hook(conv3_2);
    conv3_2->hook(act3_2);
    act3_2->hook(conv3_3);
    conv3_3->hook(act3_3);
    act3_3->hook(pool_3);
    pool_3->hook(conv4_1);




    conv4_1->hook(act4_1);
    act4_1->hook(conv4_2);
    conv4_2->hook(act4_2);
    act4_2->hook(conv4_3);
    conv4_3->hook(act4_3);
    act4_3->hook(pool_4);
    pool_4->hook(conv5_1);



    conv5_1->hook(act5_1);
    act5_1->hook(conv5_2);
    conv5_2->hook(act5_2);
    act5_2->hook(conv5_3);
    conv5_3->hook(act5_3);
    act5_3->hook(pool_5);
    pool_5->hook(full_conn_1);

    full_conn_1->hook(relu6);
    relu6->hook(drop6);

    drop6->hook(full_conn_2);
    full_conn_2->hook(relu7);
    relu7->hook(drop7);
    drop7->hook(full_conn_3);

    full_conn_3->hook(softmax);

    n.fsetup(data_1);
    n.bsetup(softmax);

    // test set #50000 imgs
    n.setup_test(data_2, 50000 / batch_size);

    const size_t train_imgs = 1281166;
    const size_t tracking_window = train_imgs / batch_size;

//    saver->load();

    // 100 epoch
    // testing every epoch
    printf("total iteration: %zu, test interval : %zu\n", (train_imgs/batch_size)*100, train_imgs/batch_size);
//    n.train((train_imgs/batch_size)*100, tracking_window, tracking_window);
    n.train(10, 100, 100);

//    saver->save();
}

