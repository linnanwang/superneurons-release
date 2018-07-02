//
// Created by ay27 on 7/4/17.
//

#include <superneurons.h>
using namespace SuperNeurons;

int main() {
    char* train_image_bin;
    char* train_label_bin;
    char* test_image_bin;
    char* test_label_bin;
    char* train_mean_file;

    train_image_bin = (char *) "/home/ay27/data/mnist/mnist_train_image.bin";
    train_label_bin = (char *) "/home/ay27/data/mnist/mnist_train_label.bin";
    test_image_bin  = (char *) "/home/ay27/data/mnist/mnist_test_image.bin";
    test_label_bin  = (char *) "/home/ay27/data/mnist/mnist_test_label.bin";
    train_mean_file = (char *) "/home/ay27/data/mnist/train_mean.bin";


    momentum_solver_t<float> solver(0.01, 0.0002, 0.9);
    network_t<float> n((base_solver_t<float> *) &solver);
//    network_t<float> n(0.01, 0.002, 0.1);
    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 1, 1);
    base_layer_t<float>* data_layer = (base_layer_t<float> *) new data_layer_t<float>(DATA_TRAIN, &reader1);

    base_layer_t<float>* pad = (base_layer_t<float> *) new padding_layer_t<float>(1, 1, 0);
    base_layer_t<float>* fc = (base_layer_t<float> *) new fully_connected_layer_t<float>(10, new xavier_initializer_t<float>(), false);
    base_layer_t<float>* softmax = (base_layer_t<float> *) new softmax_layer_t<float>();

    data_layer->hook(pad);
    pad->hook(fc);
    fc->hook(softmax);


    n.fsetup(data_layer);
    n.bsetup(softmax);

    n.train(1, 10, 10);

}