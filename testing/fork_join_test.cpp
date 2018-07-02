#include <stdlib.h>
#include <superneurons.h>
using namespace SuperNeurons;

int main(int argc, char **argv) {
    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    char* train_mean_file;
    network_t<float> n(0.01, 0.0005, 0.9);
    const size_t batch_size = 100; //train and test must be same
    
    train_mean_file = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_image.mean";
    train_image_bin = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_image_0.bin";
    train_label_bin = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_train_label_0.bin";
    test_image_bin  = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_test_image_0.bin";
    test_label_bin  = (char *) "/home/wwu/DeepLearning/data/cifar/cifar10_test_label_0.bin";
    parallel_reader_t<float > reader2(test_image_bin, test_label_bin, 2, batch_size, train_mean_file);
    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, &reader2);
    //train
    parallel_reader_t<float > reader1(train_image_bin, train_label_bin, 2, batch_size, train_mean_file);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader1);
    
    base_layer_t<float>* fork   = (base_layer_t<float>*) new fork_layer_t<float>();
    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float>(1, 3, 2, 0, 0, XAVIER, 0, 0.0001, true);
    base_layer_t<float>* conv_2 = (base_layer_t<float>*) new conv_layer_t<float>(1, 3, 2, 0, 0, XAVIER, 0, 0.0001, true);
    base_layer_t<float>* conv_3 = (base_layer_t<float>*) new conv_layer_t<float>(1, 3, 2, 0, 0, XAVIER, 0, 0.0001, true);
    base_layer_t<float>* join   = (base_layer_t<float>*) new join_layer_t<float>();
    
    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (10, XAVIER, 0, 0.1, true);
    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);
    
    //setup test
    data_2->hook_to( fork );
    //network train
    data_1->hook( fork );
    fork->hook( conv_1 );
    fork->hook( conv_2 );
    fork->hook( conv_3 );
    conv_1->hook( join );
    conv_2->hook( join );
    conv_3->hook( join );
    join->hook( full_conn_1 );
    full_conn_1->hook( softmax );
    
    n.fsetup(data_1);
    n.bsetup(softmax);
    n.setup_test( data_2, 100 );

    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs/batch_size;

    n.train(1000, tracking_window, 500);
}
