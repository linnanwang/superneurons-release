//
// Created by xluo/DeepLearning on 17/4/19.
//

#include <util/saver.h>
#include <superneurons.h>

using namespace SuperNeurons;
using namespace std;

//void test_lenet() {
//    char *train_label_bin;
//    char *train_image_bin;
//    char *test_label_bin;
//    char *test_image_bin;
//    char *checkpoint_file;
//
//    train_image_bin = (char *) "/home/xluo/DeepLearning/data/mnist/mnist_train_image.bin";
//    train_label_bin = (char *) "/home/xluo/DeepLearning/data/mnist/mnist_train_label.bin";
//    test_image_bin  = (char *) "/home/xluo/DeepLearning/data/mnist/mnist_test_image.bin";
//    test_label_bin  = (char *) "/home/xluo/DeepLearning/data/mnist/mnist_test_label.bin";
//    checkpoint_file = (char *) "/home/xluo/DeepLearning/build/checkpoint";
//
//    network_t<float> n(0.01, 0.0005, 0.9, 500);
//
//    network_saver* saver = new network_saver_impl<float>(checkpoint_file, n.get_registry());
//    // capture the ctrl+c interrupt signal
//    install_signal_processor(saver);
//
//    //test
//    parallel_reader_t<float> reader2(test_image_bin, test_label_bin, 1, 100);
//    base_layer_t<float> *data_2 = (base_layer_t<float> *) new data_layer_t<float>(data_test, &reader2);
//    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 1, 100);
//    base_layer_t<float> *data_1 = (base_layer_t<float> *) new data_layer_t<float>(data_train, &reader1);
//
//    base_layer_t<float> *conv_1 = (base_layer_t<float> *) new conv_layer_t<float>(20, 5, 1, 0, 0, true);
//    base_layer_t<float> *pool_1 = (base_layer_t<float> *) new pool_layer_t<float>(CUDNN_POOLING_MAX,
//                                                                                  CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);
//    base_layer_t<float> *conv_2 = (base_layer_t<float> *) new conv_layer_t<float>(50, 5, 1, 0, 0, true);
//    base_layer_t<float> *pool_2 = (base_layer_t<float> *) new pool_layer_t<float>(CUDNN_POOLING_MAX,
//                                                                                  CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);
//    base_layer_t<float> *full_conn_1 = (base_layer_t<float> *) new fully_connected_layer_t<float>(500, true);
//    base_layer_t<float> *act_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_SIGMOID,
//                                                                                CUDNN_NOT_PROPAGATE_NAN);
//    base_layer_t<float> *full_conn_2 = (base_layer_t<float> *) new fully_connected_layer_t<float>(10, true);
//    base_layer_t<float> *softmax = (base_layer_t<float> *) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE,
//                                                                                      CUDNN_SOFTMAX_MODE_INSTANCE);
//    //network test
//    data_2->hook_to(conv_1);
//    //network train
//    data_1->hook(conv_1);
//    conv_1->hook(pool_1);
//    pool_1->hook(conv_2);
//    conv_2->hook(pool_2);
//    pool_2->hook(full_conn_1);
//    full_conn_1->hook(act_1);
//    act_1->hook(full_conn_2);
//    full_conn_2->hook(softmax);
//
//    n.fsetup(data_1);
//    n.bsetup(softmax);
//
//    n.setup_test(data_2, 100);
//
//    saver->load();
//
//    n.train(1000);
//
//    saver->save();
//
//    delete saver;
//}

void test_cifar() {
    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    char* train_mean_file;
    char *checkpoint_file;

    train_mean_file = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_image.mean";
    train_image_bin = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_image.bin";
    train_label_bin = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_label.bin";
    test_image_bin  = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_test_image.bin";
    test_label_bin  = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_test_label.bin";
    checkpoint_file = (char *) "/home/xluo/DeepLearning/build/checkpoint";

    const size_t batch_size = 100; //train and test must be same
    //test
    parallel_reader_t<float > reader2(test_image_bin, test_label_bin, 2, batch_size, train_mean_file);
    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(data_test, &reader2);
    //train
    parallel_reader_t<float > reader1(train_image_bin, train_label_bin, 2, batch_size, train_mean_file);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(data_train, &reader1);


    /*--------------network configuration--------------*/
    network_t<float> n(0.001, 0.0004, 0.9);

    network_saver* saver = new network_saver_impl<float>(checkpoint_file, n.get_registry());
    install_signal_processor(saver);

    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float>(32, 5, 1, 2, 2, XAVIER, 0, 0, true);

    base_layer_t<float>* pool_1 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);

    base_layer_t<float>* relu_1 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* conv_2 = (base_layer_t<float>*) new conv_layer_t<float>(32, 5, 1, 2, 2, XAVIER, 0, 0, true);

    base_layer_t<float>* relu_2 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* lrn_2  = (base_layer_t<float>*) new LRN_layer_t<float>();

    base_layer_t<float>* pool_2 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);

    base_layer_t<float>* conv_3 = (base_layer_t<float>*) new conv_layer_t<float>(64, 5, 1, 2, 2, XAVIER, 0, 0, true);

    base_layer_t<float>* relu_3 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float>* lrn_3  = (base_layer_t<float>*) new LRN_layer_t<float>();

    base_layer_t<float>* pool_3 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);

    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (64, XAVIER, 0, 0, true);

    base_layer_t<float>* drop_1      = (base_layer_t<float>*) new dropout_layer_t<float>(0.1);

    base_layer_t<float>* full_conn_2 = (base_layer_t<float>*) new fully_connected_layer_t<float> (10, XAVIER, 0, 0, true);

    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

    //setup test
    data_2->hook_to( conv_1 );
    //setup network
    data_1->hook( conv_1 );
    conv_1->hook( pool_1 );
    pool_1->hook( relu_1 );
    relu_1->hook( conv_2 );
    conv_2->hook( relu_2 );
    relu_2->hook( lrn_2  );
    lrn_2->hook(  pool_2 );
    pool_2->hook( conv_3 );
    conv_3->hook( relu_3 );
    relu_3->hook( lrn_3  );
    lrn_3->hook(  pool_3 );
    pool_3->hook( full_conn_1 );
    full_conn_1->hook( drop_1 );
    drop_1->hook( full_conn_2 );
    full_conn_2->hook( softmax );

    n.fsetup(data_1);
    n.bsetup(softmax);

    n.setup_test( data_2, 50 );

//    n.train(40000);

    saver->load();

    n.train(2000);

    saver->save();

    delete saver;
}

int main() {
//    test_lenet();
    test_cifar();
}
