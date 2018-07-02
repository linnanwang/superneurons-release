//
// Created by ay27 on 7/18/17.
//

#include <solver.h>
#include <superneurons.h>
using namespace SuperNeurons;
using namespace std;


int main() {
    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    char* train_mean_file;

    const size_t batch_size = 2; //train and test must be same

    train_mean_file = (char *) "/uac/ascstd/jmye/storage/superneurons/cifar10/cifar10_train_image.mean";
    train_image_bin = (char *) "/uac/ascstd/jmye/storage/superneurons/cifar10/cifar10_train_image_0.bin";
    train_label_bin = (char *) "/uac/ascstd/jmye/storage/superneurons/cifar10/cifar10_train_label_0.bin";
    test_image_bin  = (char *) "/uac/ascstd/jmye/storage/superneurons/cifar10/cifar10_test_image_0.bin";
    test_label_bin  = (char *) "/uac/ascstd/jmye/storage/superneurons/cifar10/cifar10_test_label_0.bin";

//    parallel_reader_t<float > reader2(test_image_bin, test_label_bin, 1, batch_size);
//    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, &reader2);
    //train
    parallel_reader_t<float > reader1(train_image_bin, train_label_bin, 1, batch_size);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader1);


//    base_solver_t<float>* solver = (base_solver_t<float> *) new momentum_solver_t<float>(0.001, 0.0001, 0.9);
//    base_solver_t<float>* solver = (base_solver_t<float>*)new sgd_solver_t<float>(0.001, 0.001);
//    base_solver_t<float>* solver = (base_solver_t<float>*)new nesterov_solver_t<float>(0.001, 0.001, 0.9);
//    base_solver_t<float>* solver = (base_solver_t<float>*) new adagrad_solver_t<float>(0.01, 0.0004, 0.9);
    base_solver_t<float>* solver = (base_solver_t<float>*) new rmsprop_solver_t<float>(0.01, 0.0004, 0.9, 0.001);
    network_t<float> n(solver);

    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float>(1, 3, 2, 0, 0, new xavier_initializer_t<float>(), true);

    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (10, new xavier_initializer_t<float>(), true);
    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

    //network train
    data_1->hook( conv_1 );
    conv_1->hook( full_conn_1 );
    full_conn_1->hook( softmax );

    n.fsetup(data_1);
    n.bsetup(softmax);
//    n.setup_test( data_2, 100 );

    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs/batch_size;

    n.train(2, tracking_window, 500);
}