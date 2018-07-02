//
// Created by ay27 on 17/6/20.
//

#include <superneurons.h>

using namespace std;
using namespace SuperNeurons;


int main() {
    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    network_t<float> n(0.1, 0.0004, 0.9);


    train_image_bin = (char *) "/home/ay27/data/mnist/mnist_train_image.bin";
    train_label_bin = (char *) "/home/ay27/data/mnist/mnist_train_label.bin";
    test_image_bin  = (char *) "/home/ay27/data/mnist/mnist_test_image.bin";
    test_label_bin  = (char *) "/home/ay27/data/mnist/mnist_test_label.bin";
    const size_t batch_size = 1; //train and test must be same
//    parallel_reader_t<float > reader2(test_image_bin, test_label_bin, 2, batch_size);
//    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, &reader2);
    parallel_reader_t<float > reader1(train_image_bin, train_label_bin, 2, batch_size);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader1);


    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float> (1, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.02), true);
    base_layer_t<float>* act_1  = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* fork = (base_layer_t<float>*) new fork_layer_t<float>();

    // left
    base_layer_t<float>* conv_21 = (base_layer_t<float>*) new conv_layer_t<float>(1, 3,1,1,1,new gaussian_initializer_t<float>(0, 0.02), true);
    //right
    base_layer_t<float>* conv_22 = (base_layer_t<float>*) new conv_layer_t<float>(1, 3,1,1,1,new gaussian_initializer_t<float>(0, 0.02), true);

    base_layer_t<float>* join = (base_layer_t<float>*) new concat_layer_t<float>();

    base_layer_t<float>* fc = (base_layer_t<float>*) new fully_connected_layer_t<float>(10, new gaussian_initializer_t<float>(0, 0.02), true);
    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

    data_1->hook(conv_1);
    conv_1->hook(act_1);
    act_1->hook(fork);

    fork->hook(conv_21);
    fork->hook(conv_22);

    conv_21->hook(join);
    conv_22->hook(join);

    join->hook(fc);
    fc->hook(softmax);

    n.fsetup(data_1);
    n.bsetup(softmax);

    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs/batch_size;

    n.train(1, tracking_window, 1000);

}