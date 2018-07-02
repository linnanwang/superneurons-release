#include <stdlib.h>
#include <superneurons.h>
#include <thread>
#include <util/image_reader.h>

using namespace SuperNeurons;

int main(int argc, char **argv) {
    char* train_image_bin;
    char* train_label_bin;
    char* test_image_bin;
    char* test_label_bin;
//    char* train_mean_file;

    train_image_bin = (char *) "/uac/ascstd/jmye/storage/superneurons/mnist/mnist_train_image.bin";
    train_label_bin = (char *) "/uac/ascstd/jmye/storage/superneurons/mnist/mnist_train_label.bin";
    test_image_bin  = (char *) "/uac/ascstd/jmye/storage/superneurons/mnist/mnist_test_image.bin";
    test_label_bin  = (char *) "/uac/ascstd/jmye/storage/superneurons/mnist/mnist_test_label.bin";
//    train_mean_file = (char *) "/home/ay27/data/mnist/train_mean.bin";

//    train_image_bin  = (char *) "/home/ay27/superneurons/build/val_data_1.bin";
//    train_label_bin  = (char *) "/home/ay27/superneurons/build/val_label_1.bin";
//     train_mean_file = (char *) "/home/ay27/data/ilsvrc2012/train_mean.bin";

//    image_reader_t<float> reader(train_image_bin, train_label_bin, 10, false);

//    network_t<float> n((base_solver_t<float> *) new momentum_solver_t<float>(0.01, 0.001, 0.9));;
//    std::vector<tensor_t<float >* >* reg = new std::vector<tensor_t<float>* >();
//    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 1, 2);
//    base_layer_t<float>* data_1 = (base_layer_t<float> *) new data_layer_t<float>(DATA_TRAIN, &reader1);
//    base_layer_t<float> *conv = (base_layer_t<float> *) new conv_layer_t<float>(2, 5, 3, 0, 0, new xavier_initializer_t<float>(), false);
//    base_layer_t<float> *softmax = (base_layer_t<float> *) new softmax_layer_t<float>();
//
//    data_1->hook(conv);
//    conv->hook(softmax);
//
//    n.fsetup(data_1);
//    n.bsetup(softmax);
//
//    n.train(2, 10, 10);

//    std::thread threads[5];
//    pthread_mutex_t mutex;
//    pthread_mutex_init(&mutex, NULL);
//    for (int i = 0; i < 5; ++i) {
//        threads[i] = std::thread([&]() {
//            for(int j = 0; j < 1000; j ++) {
//                pthread_mutex_lock(&mutex);
//                reader1.get_batch(data_1.get_f_out(), n.get_registry()->get_train_label());
//                pthread_mutex_unlock(&mutex);
//            }
//        });
//    }
//    for (int i = 0; i < 5; ++i) {
//        threads[i].join();
//    }
//    pthread_mutex_destroy(&mutex);
}

