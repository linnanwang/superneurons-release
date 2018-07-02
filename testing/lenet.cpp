#include <stdlib.h>
#include <superneurons.h>
#include <util/saver.h>

using namespace SuperNeurons;

int main(int argc, char **argv) {
    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;

    train_image_bin = (char *) "/home/ay27/data/mnist/mnist_train_image.bin";
    train_label_bin = (char *) "/home/ay27/data/mnist/mnist_train_label.bin";
    test_image_bin  = (char *) "/home/ay27/data/mnist/mnist_test_image.bin";
    test_label_bin  = (char *) "/home/ay27/data/mnist/mnist_test_label.bin";

    const size_t batch_size = 100; //train and test must be same
    //train
    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 2, batch_size, 1, 28, 28);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader1);

    //test
    parallel_reader_t<float> reader2(test_image_bin, test_label_bin, 2, batch_size, 1, 28, 28);
    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, &reader2);
    
    /*--------------network configuration--------------*/
    base_solver_t<float>* solver = (base_solver_t<float> *) new momentum_solver_t<float>(0.001, 0.0, 0.9);
    network_t<float> n(solver);

    base_layer_t<float>* conv_1      = (base_layer_t<float>*) new conv_layer_t<float>(20, 5, 1, 0, 0, new xavier_initializer_t<float>(), true);
    base_layer_t<float>* pool_1      = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);
    base_layer_t<float>* conv_2      = (base_layer_t<float>*) new conv_layer_t<float>(50, 5, 1, 0, 0,new xavier_initializer_t<float>(), true);
    base_layer_t<float>* pool_2      = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);
    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (500,new xavier_initializer_t<float>(), true);
    base_layer_t<float>* act_1       = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float>* full_conn_2 = (base_layer_t<float>*) new fully_connected_layer_t<float> (10,new xavier_initializer_t<float>(), true);
    base_layer_t<float>* softmax     = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);
    //network test
    data_2->hook_to( conv_1 );
    //network train
    data_1->hook( conv_1 );


    conv_1->hook( pool_1 );
    pool_1->hook( conv_2 );
    conv_2->hook( pool_2 );
    pool_2->hook( full_conn_1 );
    full_conn_1->hook( act_1 );
    act_1->hook( full_conn_2 );
    full_conn_2->hook( softmax );
    
    n.fsetup(data_1);
    n.bsetup(softmax);
    
    n.setup_test( data_2, 100 );
    
    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs/batch_size;

    n.train(500, tracking_window, 500);

    saver_impl<float>* s = new saver_impl<float>("/home/ay27/superneurons/build/lenet.dump", n.get_registry(), solver);
    s->save();

}
