//
// Created by ay27 on 17/3/30.
//

#include <ios>
#include <fstream>
#include <superneurons.h>
#include <util/image_reader.h>
#include <util/binary_dumper.h>

using namespace std;
using namespace SuperNeurons;

typedef unsigned char value_type;

const int CIFAR_C = 3;
const int CIFAR_H = 32;
const int CIFAR_W = 32;
const int LABEL_SIZE = 1;
const int IMAGE_SIZE = 3072;
const int CIFAR_100_TEST_N = 10000;
const int CIFAR_100_TRAIN_N = 50000;

void read_one_bin_file(const char *bin_path, char *data, char *label, int N) {
    ifstream bin_file(bin_path, ios::in | ios::binary);
    assert(bin_file);

    char drop_coarse_labe;
    for (int i = 0; i < N; ++i) {
        bin_file.read(&drop_coarse_labe, 1);
        bin_file.read(label + i, 1);
        bin_file.read(data + i * IMAGE_SIZE, IMAGE_SIZE);
        if (bin_file.gcount() != IMAGE_SIZE) {
            printf("i = %d\n", i);
            printf("error when reading file %s\n", bin_path);
            break;
        }
    }
}

void convert_cifar(const char *cifar_dir, const char *output_dir) {

    char *data = new char[CIFAR_100_TRAIN_N * IMAGE_SIZE];
    char *label = new char[CIFAR_100_TRAIN_N * LABEL_SIZE];

    char *bin_file_path = new char[200];
    sprintf(bin_file_path, "%s/train.bin", cifar_dir);
    read_one_bin_file(bin_file_path, data, label, CIFAR_100_TRAIN_N);

    char *dst_image_path = new char[200];
    char *dst_label_path = new char[200];
    sprintf(dst_image_path, "%s/cifar100_train_image.bin", output_dir);
    sprintf(dst_label_path, "%s/cifar100_train_label.bin", output_dir);

    Dumper train_data_dumper(CIFAR_100_TRAIN_N, CIFAR_C, CIFAR_H, CIFAR_W, dst_image_path);
    Dumper train_label_dumper(CIFAR_100_TRAIN_N, CIFAR_C, 1, 1, dst_label_path);
    train_data_dumper.dump_image(data, CIFAR_100_TRAIN_N * IMAGE_SIZE);
    train_label_dumper.dump_label(label, CIFAR_100_TRAIN_N * LABEL_SIZE);

    /////////////////////////////////////////////////////////////////////////////////////////////
    // dump test batch
    sprintf(bin_file_path, "%s/test.bin", cifar_dir);
    sprintf(dst_image_path, "%s/cifar100_test_image.bin", output_dir);
    sprintf(dst_label_path, "%s/cifar100_test_label.bin", output_dir);

    read_one_bin_file(bin_file_path, data, label, CIFAR_100_TEST_N);

    Dumper test_data_dumper(CIFAR_100_TEST_N, CIFAR_C, CIFAR_H, CIFAR_W, dst_image_path);
    Dumper test_label_dumper(CIFAR_100_TEST_N, CIFAR_C, 1, 1, dst_label_path);
    test_data_dumper.dump_image(data, CIFAR_100_TEST_N * IMAGE_SIZE);
    test_label_dumper.dump_label(label, CIFAR_100_TEST_N * LABEL_SIZE);

    delete[] data;
    delete[] label;
    delete[] bin_file_path;
    delete[] dst_image_path;
    delete[] dst_label_path;
}

//void load_test(const char *cifar_dir, const char *output_dir) {
//    char *data = new char[CIFAR_100_TRAIN_N * IMAGE_SIZE];
//    char *label = new char[CIFAR_100_TRAIN_N * LABEL_SIZE];
//
//    char *bin_file_path = new char[200];
//    sprintf(bin_file_path, "%s/train.bin", cifar_dir);
//    read_one_bin_file(bin_file_path, data, label, CIFAR_100_TRAIN_N);
//
//    /////////////////////////////////////////////////////////////////////////////////////////////
//    char *dst_image_path = new char[200];
//    char *dst_label_path = new char[200];
//    sprintf(dst_image_path, "%s/cifar100_train_image.bin", output_dir);
//    sprintf(dst_label_path, "%s/cifar100_train_label.bin", output_dir);
//
//    DataReader<float> reader(dst_image_path, dst_label_path, CIFAR_100_TRAIN_N);
//    assert(reader.get_n() == CIFAR_100_TRAIN_N);
//    assert(reader.get_c() == CIFAR_C);
//    assert(reader.get_h() == CIFAR_H);
//    assert(reader.get_w() == CIFAR_W);
//
//    float *batch_data = new float[CIFAR_100_TRAIN_N * IMAGE_SIZE * sizeof(float)];
//    float *batch_label = new float[CIFAR_100_TRAIN_N * LABEL_SIZE * sizeof(float)];
//    reader.get_batch(batch_data, batch_label);
//
//    for (int j = 0; j < IMAGE_SIZE * CIFAR_100_TRAIN_N; ++j) {
//        assert(batch_data[j] == (float) (unsigned char) data[j]);
//    }
//    for (int j = 0; j < CIFAR_100_TRAIN_N; ++j) {
//        printf("%d ", label[j]);
//        assert(batch_label[j] == (float)label[j]);
//    }
//
//    // test seek to first
//    reader.get_batch(batch_data, batch_label);
//
//    /////////////////////////////////////////////////////////////////////////////////////////////
//    sprintf(bin_file_path, "%s/test.bin", cifar_dir);
//    sprintf(dst_image_path, "%s/cifar100_test_image.bin", output_dir);
//    sprintf(dst_label_path, "%s/cifar100_test_label.bin", output_dir);
//
//    read_one_bin_file(bin_file_path, data, label, CIFAR_100_TEST_N);
//
//    DataReader<float> reader1(dst_image_path, dst_label_path, CIFAR_100_TRAIN_N);
//    reader1.get_batch(batch_data, batch_label);
//    for (int i = 0; i < IMAGE_SIZE * CIFAR_100_TEST_N; ++i) {
//        assert(batch_data[i] == (float) (unsigned char) data[i]);
//    }
//    for (int i = 0; i < LABEL_SIZE * CIFAR_100_TEST_N; ++i) {
//        printf("%d ", label[i]);
//        assert(batch_label[i] == (float) label[i]);
//    }
//    // test seek to first
//    reader1.get_batch(batch_data, batch_label);
//
//    printf("test for cifar100 success\n");
//
//    delete[] batch_data;
//    delete[] batch_label;
//    delete[] bin_file_path;
//    delete[] data;
//    delete[] label;
//}

int main(int argc, char **argv) {
#ifdef DEBUG
    printf("debug mode\n");
#endif
    if (argc != 3) {
        fprintf(stderr,
                "Usage:\nconvert_cifar100 [cifar_folder] [binary_output_folder] \n");
        exit(-1);
    }
    char *cifar_dir = argv[1];
    char *output_dir = argv[2];

    convert_cifar(cifar_dir, output_dir);
//#ifdef DEBUG
//    load_test(cifar_dir, output_dir);
//#endif
    printf("convert finish, output dir is : %s\n", output_dir);

}
