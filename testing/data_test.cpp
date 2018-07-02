#include <stdlib.h>
#include <superneurons.h>
using namespace SuperNeurons;

int main(int argc, char **argv) {
    
    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    char* train_mean_file;
    char* checkpoint_file;
    
    if( argc != 3 ) {
        printf("please run as ./alexnet batch_size training_iters\n");
    }
    
    
    train_mean_file = (char *) "/data/lwang53/imgnet/bin/imgnet_train.mean";
    train_image_bin = (char *) "/data/lwang53/imgnet/bin/train_data_0.bin";
    train_label_bin = (char *) "/data/lwang53/imgnet/bin/train_label_0.bin";
    test_image_bin  = (char *) "/data/lwang53/imgnet/bin/val_data_0.bin";
    test_label_bin  = (char *) "/data/lwang53/imgnet/bin/val_label_0.bin";
    checkpoint_file = (char *) "/data/lwang53/imgnet/checkpoint/alexnet_checkpoint";
    
    const size_t batch_size = atoi(argv[1]); //train and test must be same
    
    preprocessor<float>* processor_train = new preprocessor<float>();
    processor_train->add_preprocess(new central_crop_t<float>(batch_size, 3, 256, 256, batch_size, 3, 227, 227));
    
    preprocessor<float>* processor_test = new preprocessor<float>();
    processor_test->add_preprocess(new central_crop_t<float>(batch_size, 3, 256, 256, batch_size, 3, 227, 227));
    
    //test
    parallel_reader_t<float > reader2 (test_image_bin, test_label_bin, 1, batch_size, 3, 227, 227, processor_train, 1, 1);
    base_layer_t<float>*   data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, &reader2);
    //train
    parallel_reader_t<float > reader1 (train_image_bin, train_label_bin, 4, batch_size, 3, 227, 227, processor_test, 4, 2);
    base_layer_t<float>*   data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader1);
    
    /*--------------network configuration--------------*/
    base_solver_t<float>* solver = (base_solver_t<float> *) new momentum_solver_t<float>(0.01, 0.0005, 0.9);
    solver->set_lr_decay_policy(ITER, {100000, 200000, 300000, 400000}, {0.001, 0.0001, 0.00001, 0.000001});
    network_t<float> n(solver);
    
    network_saver* saver = new network_saver_impl<float>(checkpoint_file, n.get_registry());
    install_signal_processor(saver);
    
    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float>(1, 11, 4, 0, 0, new gaussian_initializer_t<float>(0, 0.01), true);
    
    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (1000, new gaussian_initializer_t<float>(0, 0.01), true);
    
    base_layer_t<float>* softmax = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);
    
    //setup test
    data_2->hook_to( conv_1 );
    //setup network
    data_1->hook( conv_1 );
    conv_1->hook( full_conn_1 );
    full_conn_1->hook( softmax  );
    
    n.fsetup(data_1);
    n.bsetup(softmax);
    
    n.setup_test( data_2, 50000 / batch_size );
    
    const size_t train_imgs = 1281166;
    const size_t tracking_window = train_imgs / batch_size;
    
    // 200 epoch
    // testing every epoch
    fprintf(stderr, "total iteration: %zu, test interval : %zu\n", (train_imgs/batch_size)*200, train_imgs/batch_size);
    n.train(atoi(argv[2]), tracking_window, 1000);
}
