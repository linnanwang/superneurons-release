#include <stdlib.h>
#include <superneurons.h>

using namespace SuperNeurons;

int main(int argc, char **argv) {

    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    char* train_mean_file;

    train_mean_file = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_image.mean";
    train_image_bin = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_image_0.bin";
    train_label_bin = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_label_0.bin";
    test_image_bin  = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_test_image_0.bin";
    test_label_bin  = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_test_label_0.bin";
    const size_t batch_size = 100; //train and test must be same
    const size_t C = 3, H = 32, W = 32;


    base_preprocess_t<float>* mean_sub =
    (base_preprocess_t<float>*) new mean_subtraction_t<float>(batch_size, C, H, W, train_mean_file);

    base_preprocess_t<float> *pad = (base_preprocess_t<float> *) new border_padding_t<float>(
                                                                                             batch_size, C, H, W, 4, 4);
    base_preprocess_t<float> *crop = (base_preprocess_t<float> *) new random_crop_t<float>(
                                                                                           batch_size, C, H + 8, W + 8, batch_size, C, H, W);
    base_preprocess_t<float> *flip = (base_preprocess_t<float> *) new random_flip_left_right_t<float>(
                                                                                                      batch_size, C, H, W);
    base_preprocess_t<float> *bright = (base_preprocess_t<float> *) new random_brightness_t<float>(
                                                                                                   batch_size, C, H, W, 63);
    base_preprocess_t<float> *contrast = (base_preprocess_t<float> *) new random_contrast_t<float>(
                                                                                                   batch_size, C, H, W, 0.2, 1.8);
    base_preprocess_t<float> *standardization =
    (base_preprocess_t<float> *) new per_image_standardization_t<float>(
                                                                        batch_size, C, H, W);

    preprocessor<float>* processor = new preprocessor<float>();
    processor->add_preprocess(mean_sub)
    ->add_preprocess(pad)
    ->add_preprocess(crop)
    ->add_preprocess(flip)
    ->add_preprocess(bright)
    ->add_preprocess(contrast)
    ->add_preprocess(standardization);

    //test
    parallel_reader_t<float > reader2(test_image_bin, test_label_bin, 2, batch_size,C, H, W, processor);
    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, &reader2);
    //train
    parallel_reader_t<float > reader1(train_image_bin, train_label_bin, 2, batch_size,C,H,W, processor);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader1);


    /*--------------network configuration--------------*/

    base_solver_t<float>* solver = (base_solver_t<float> *) new momentum_solver_t<float>(0.002, 0.0, 0.9);
    network_t<float> n(solver);

    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float>(64, 5, 1, 2, 2, new gaussian_initializer_t<float>(0, 0.02), true);
    base_layer_t<float>* relu_1 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_2 = (base_layer_t<float>*) new conv_layer_t<float>(64, 5, 1, 2, 2, new gaussian_initializer_t<float>(0, 0.02), true);
    base_layer_t<float>* relu_2 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (384, new gaussian_initializer_t<float>(0, 0.02), true);
    base_layer_t<float>* relu_3 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* full_conn_2 = (base_layer_t<float>*) new fully_connected_layer_t<float> (192, new gaussian_initializer_t<float>(0, 0.02), true);
    base_layer_t<float>* relu_4 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* full_conn_3 = (base_layer_t<float>*) new fully_connected_layer_t<float> (10, new gaussian_initializer_t<float>(0, 0.02), true);
    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

    //setup test
    data_2->hook_to( conv_1 );
    //setup network
    data_1->hook( conv_1 );

    conv_1->hook( relu_1 );
    relu_1->hook( conv_2 );

    conv_2->hook( relu_2 );
    relu_2->hook(full_conn_1);

    full_conn_1->hook(relu_3);
    relu_3->hook(full_conn_2);

    full_conn_2->hook(relu_4);
    relu_4->hook(full_conn_3);

    full_conn_3->hook(softmax);

    n.fsetup(data_1);
    n.bsetup(softmax);

    n.setup_test( data_2, 100 );
    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs/batch_size;
    n.train(2, tracking_window, 500);
}
