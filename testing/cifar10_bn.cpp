#include <stdlib.h>
#include <superneurons.h>
using namespace SuperNeurons;

int main(int argc, char **argv) {
    
    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    char* train_mean_file;
    
    train_mean_file = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_image.mean";
    train_image_bin = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_image.bin";
    train_label_bin = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_train_label.bin";
    test_image_bin  = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_test_image.bin";
    test_label_bin  = (char *) "/home/xluo/DeepLearning/data/cifar10/cifar10_test_label.bin";
    
    data_transformer<float> transformer(3, 32, 32, 1.0, NULL, train_mean_file);
    
    const size_t batch_size = 100; //train and test must be same
    //test
    DataReader<float> reader2(test_image_bin, test_label_bin, &transformer);
    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(data_test, &reader2, 100);
    //train
    DataReader<float> reader1(train_image_bin, train_label_bin, &transformer);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(data_train, &reader1, 100);
    
    
    /*--------------network configuration--------------*/
    network_t<float> n(0.001, 0.004, 0.9, 1000);
    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float>(32, 5, 1, 2, 2, true);
    
    base_layer_t<float>* bn_1   = (base_layer_t<float>*) new batch_normalization_layer_t<float> (CUDNN_BATCHNORM_SPATIAL, 0.0001);

    base_layer_t<float>* pool_1 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    
    base_layer_t<float>* relu_1 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    
    base_layer_t<float>* conv_2 = (base_layer_t<float>*) new conv_layer_t<float>(32, 5, 1, 2, 2, true);
    
    base_layer_t<float>* bn_2   = (base_layer_t<float>*) new batch_normalization_layer_t<float> (CUDNN_BATCHNORM_SPATIAL, 0.0001);
    
    base_layer_t<float>* relu_2 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    
    base_layer_t<float>* pool_2 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    
    base_layer_t<float>* conv_3 = (base_layer_t<float>*) new conv_layer_t<float>(64, 5, 1, 2, 2, true);
    
    base_layer_t<float>* bn_3   = (base_layer_t<float>*) new batch_normalization_layer_t<float> (CUDNN_BATCHNORM_SPATIAL, 0.0001);
    
    base_layer_t<float>* relu_3 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    
    base_layer_t<float>* pool_3 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    
    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (64, true);
    
    base_layer_t<float>* full_conn_2 = (base_layer_t<float>*) new fully_connected_layer_t<float> (10, true);
    
    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);
    
    //setup test
    data_2->hook_to( conv_1 );
    //setup network
    data_1->hook( conv_1 );
    conv_1->hook( pool_1 );
    pool_1->hook( relu_1 );
    relu_1->hook( bn_1 );
    bn_1->hook( conv_2 );
    conv_2->hook( relu_2 );
    relu_2->hook( pool_2 );
    pool_2->hook( bn_2 );
    bn_2->hook( conv_3 );
    conv_3->hook( relu_3 );
    relu_3->hook( pool_3 );
    pool_3->hook( bn_3 );
    bn_3->hook( full_conn_1 );
    full_conn_1->hook( full_conn_2 );
    full_conn_2->hook( softmax );
    
    n.fsetup(data_1);
    n.bsetup(softmax);
    
    n.setup_test( data_2, 100 );
    
    n.train(40000);
}
