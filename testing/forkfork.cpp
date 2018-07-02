//
// Created by ay27 on 7/31/17.
//

#include <superneurons.h>

using namespace SuperNeurons;
using namespace std;

int main() {
    char *train_label_bin;
    char *train_image_bin;
    char *test_label_bin;
    char *test_image_bin;
    char *train_mean_file;

    train_mean_file = (char *) "/home/ay27/data/cifar10/cifar10_train_image.mean";
    train_image_bin = (char *) "/home/ay27/data/cifar10/cifar10_train_image_0.bin";
    train_label_bin = (char *) "/home/ay27/data/cifar10/cifar10_train_label_0.bin";
    test_image_bin = (char *) "/home/ay27/data/cifar10/cifar10_test_image_0.bin";
    test_label_bin = (char *) "/home/ay27/data/cifar10/cifar10_test_label_0.bin";

    size_t batch_size = 2;

    //test
    parallel_reader_t<float > reader2(test_image_bin, test_label_bin, 1, batch_size, 3, 32, 32);
    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, &reader2);
    //train
    parallel_reader_t<float > reader1(train_image_bin, train_label_bin, 1, batch_size, 3, 32, 32);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader1);


    /*--------------network configuration--------------*/

    base_solver_t<float>* solver = (base_solver_t<float> *) new momentum_solver_t<float>(0.002, 0.0, 0.9);
    network_t<float> n(solver);

    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float>(2, 5, 2, 0, 0, new gaussian_initializer_t<float>(0, 0.02),
                                                                                 false);
    base_layer_t<float>* relu_1 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* fork =(base_layer_t<float>*) new fork_layer_t<float>();
    // left
    base_layer_t<float>* conv_l1 = (base_layer_t<float>*)new conv_layer_t<float>(2, 2, 1, 0, 0, new xavier_initializer_t<float>(), false);
    base_layer_t<float>* fork_2 = (base_layer_t<float>*) new fork_layer_t<float>();

    base_layer_t<float>* convll = (base_layer_t<float>*) new conv_layer_t<float>(2, 2, 1, 0, 0,new xavier_initializer_t<float>(), false);
    base_layer_t<float>* convlr = (base_layer_t<float>*) new conv_layer_t<float>(2, 2, 1, 0, 0, new xavier_initializer_t<float>(), false);

    // right
    base_layer_t<float>* conv_r1 = (base_layer_t<float>*)new conv_layer_t<float>(3, 2, 1, 0, 0, new xavier_initializer_t<float>(), false);
    base_layer_t<float>* conv_r2 = (base_layer_t<float>*)new conv_layer_t<float>(3, 2, 1, 0, 0, new xavier_initializer_t<float>(), false);

    // join
    base_layer_t<float>* join = (base_layer_t<float>*) new concat_layer_t<float>();


    base_layer_t<float>* full_conn_3 = (base_layer_t<float>*) new fully_connected_layer_t<float> (10, new gaussian_initializer_t<float>(0, 0.02), true);
    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

    //setup test
    data_2->hook_to( conv_1 );
    //setup network
    data_1->hook( conv_1 );


    conv_1->hook( relu_1 );
    relu_1->hook(fork);

    fork->hook(conv_l1);
    fork->hook(conv_r1);

    conv_l1->hook(fork_2);
    fork_2->hook(convll);
    fork_2->hook(convlr);

    convll->hook(join);
    convlr->hook(join);
    conv_r1->hook(conv_r2);
    conv_r2->hook(join);

    join->hook(full_conn_3);
    full_conn_3->hook(softmax);

    n.fsetup(data_1);
    n.bsetup(softmax);

    n.setup_test( data_2, 100 );
    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs/batch_size;
    n.train(1, tracking_window, 500);

}