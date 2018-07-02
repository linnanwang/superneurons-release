//
// Created by ay27 on 17/6/10.
//


#include <superneurons.h>
#include <tensor.h>
#include "testing.h"
#include <util/preprocess.h>
#include <util/print_util.h>

using namespace std;
using namespace SuperNeurons;

class test_preprocess : public TestSuite {
protected:
    float *src, *dst;
    size_t N, C, H, W;
public:
    void setup() {
        N = 2;
        C = 3;
        H = 3;
        W = 4;

        src = new float[N * C * H * W];
        for (size_t i = 0; i < N * C * H * W; ++i) {
            src[i] = i;
        }
    }

    void teardown() {
        delete[] src;
        delete[] dst;
    }
};


ADDTEST(test_preprocess, test_all) {
    char* train_label_bin;
    char* train_image_bin;
    char* train_mean_file;

    train_mean_file = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/train_mean.bin";
    train_image_bin = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/val_data_0.bin";
    train_label_bin = (char *) "/data/lwang53/dataset/imgnet/bin_256x256_imgnet/val_label_0.bin";

    size_t batch_size = 1;
    size_t H = 100, W = 100;
    preprocessor<float>* processor_train = new preprocessor<float>();
    //processor_train->add_preprocess(new mean_subtraction_t<float>(batch_size, 3, 256, 256, train_mean_file));
    processor_train->add_preprocess(new central_crop_t<float>(batch_size, 3, 256, 256, batch_size, 3, H, W));
    processor_train->add_preprocess(new random_flip_left_right_t<float>(batch_size, 3, H, W));
    
    parallel_reader_t<float > reader2 (train_image_bin, train_label_bin, 1, batch_size, 3, H, W, processor_train, 1, 1);
    base_layer_t<float>*   data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader2);
    
    std::vector<tensor_t<float>* > *reg = new std::vector<tensor_t<float>* >();
    
    tensor_t<float>* output = new tensor_t<float>(batch_size, 3, H, W, reg, DATA_SOURCE, 1);
    tensor_t<float> *label  = new tensor_t<float>(batch_size, 1, 1, 1, reg, DATA_SOURCE, 1);
    for(int i = 0; i <1000 ; i++) {
        reader2.get_batch(output, label);
    }
    reader2.get_batch(output, label);
    output->printTensorNoDebug("image");
    label->printTensorNoDebug("label");
}


/*
ADDTEST(test_preprocess, test_tensor) {
    std::vector<tensor_t<float>* > *reg = new std::vector<tensor_t<float>* >();
    tensor_t<float> *label  = new tensor_t<float>(10, 1, 1, 1, reg, DATA_SOURCE, 1);
    label->init(new constant_initializer_t<float>(1));
    cublasHandle_t  cublas_handle;
    checkCublasErrors( cublasCreate(&cublas_handle) );
    float result = label->squared_sum(&cublas_handle);
    checkCublasErrors( cublasDestroy(cublas_handle) );
    printf("tensor squared sum:%f\n", result);
}
*/

//ADDTEST(test_preprocess, test_padding) {
//    size_t padH = 4, padW = 2;
//    dst = new float[N * C * (H + padH + padH) * (W + padW + padW)];
//    base_preprocess_t<float> *padding = (base_preprocess_t<float> *) new border_padding_t<float>(N, C, H, W, padH,
//                                                                                                 padW);
//    padding->transfer(src, dst);
//    print_array<float>(dst, N, C, H + padH + padH, W + padW + padW);
//}
//
//ADDTEST(test_preprocess, test_random_crop) {
//    size_t padH = 4, padW = 2;
//    dst = new float[N * C * (H + padH + padH) * (W + padW + padW)];
//    base_preprocess_t<float> *padding = (base_preprocess_t<float> *) new border_padding_t<float>(N, C, H, W, padH,
//                                                                                                 padW);
//    padding->transfer(src, dst);
//
//    base_preprocess_t<float> *random_crop = (base_preprocess_t<float> *) new random_crop_t<float>(N, C, H + padH + padH,
//                                                                                                  W + padW + padW,
//                                                                                                  N, C, H, W);
//    
//    base_preprocess_t<float> *central_crop = (base_preprocess_t<float> *) new central_crop_t<float>(N, C, H + padH + padH,
//                                                                                                  W + padW + padW,
//                                                                                                  N, C, H, W);
//    random_crop->transfer(dst, src);
//    print_array(src, N, C, H, W);
//    printf("second random crop-------\n");
//    random_crop->transfer(dst, src);
//    print_array(src, N, C, H, W);
//    
//    printf("central crop-------------\n");
//    central_crop->transfer(dst, src);
//    print_array(src, N, C, H, W);
//    
//    
//}
//
//ADDTEST(test_preprocess, test_random_flip_left_right) {
//    dst = new float[N * C * H * W];
//    base_preprocess_t<float> *flip = (base_preprocess_t<float> *) new random_flip_left_right_t<float>(N, C, H, W);
//    flip->transfer(src, dst);
//    print_array(dst, N, C, H, W);
//}
//
//ADDTEST(test_preprocess, test_random_brightness) {
//    dst = new float[N * C * H * W];
//    base_preprocess_t<float> *bright = (base_preprocess_t<float> *) new random_brightness_t<float>(N, C, H, W, 100);
//    bright->transfer(src, dst);
//    print_array(dst, N, C, H, W);
//}
//
//ADDTEST(test_preprocess, test_random_contrast_t) {
//    dst = new float[N * C * H * W];
//    base_preprocess_t<float> *contrast = (base_preprocess_t<float> *) new random_contrast_t<float>(N, C, H, W, 0.2,
//                                                                                                   1.8);
//    contrast->transfer(src, dst);
//    print_array(dst, N, C, H, W);
//}
//
//ADDTEST(test_preprocess, test_per_image_standardization) {
//    dst = new float[N * C * H * W];
//    base_preprocess_t<float> *per = (base_preprocess_t<float> *) new per_image_standardization_t<float>(N, C, H, W);
//    per->transfer(src, dst);
//    print_array(dst, N, C, H, W);
//}
//
//ADDTEST(test_preprocess, test_preprocessor) {
//    dst = new float[N*C*(H-2)*(W-2)];
//    preprocessor<float>* p = new preprocessor<float>();
////    p->add_preprocess(new border_padding_t<float>(N,C,H,W, 2,2));
//    p->add_preprocess(new random_crop_t<float>(N,C,H,W, N,C,H-2,W-2));
//    p->add_preprocess(new random_flip_left_right_t<float>(N,C,H-2,W-2));
//    p->process(src, dst);
//    print_array(src, N,C,H,W);
//    print_array(dst, N, C, H-2, W-2);
//}
