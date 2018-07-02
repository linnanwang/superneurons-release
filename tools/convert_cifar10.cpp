//
// Created by ay27 on 17/3/29.
//
#include <ios>
#include <fstream>
#include <superneurons.h>
#include <util/binary_dumper.h>

using namespace std;
using namespace SuperNeurons;

typedef unsigned char value_type;

const int CIFAR_C = 3;
const int CIFAR_H = 32;
const int CIFAR_W = 32;
const int LABEL_SIZE = 1;
const int IMAGE_SIZE = 3072;
const int CIFAR_10_N_EVERY_BIN = 10000;
const int CIFAR_TRAIN_BATCH = 5;

void read_one_bin_file(const char *bin_path, char *data, char *label) {
    ifstream bin_file(bin_path, ios::in | ios::binary);
    assert(bin_file);

    for (int i = 0; i < CIFAR_10_N_EVERY_BIN; ++i) {
        bin_file.read(label + i, 1);
        bin_file.read(data + i * IMAGE_SIZE, IMAGE_SIZE);
    }
}

void convert_cifar(const char *cifar_dir, const char *output_dir,
                   const int train_total_blocks,
                   const int test_total_blocks) {

    char *data = new char[CIFAR_10_N_EVERY_BIN * IMAGE_SIZE * CIFAR_TRAIN_BATCH];
    char *label = new char[CIFAR_10_N_EVERY_BIN * LABEL_SIZE * CIFAR_TRAIN_BATCH];

    // read all data
    char *bin_file_path = new char[200];
    for (int i = 1; i <= CIFAR_TRAIN_BATCH; ++i) {
        sprintf(bin_file_path, "%s/data_batch_%d.bin", cifar_dir, i);
        read_one_bin_file(bin_file_path,
                          data + (i - 1) * CIFAR_10_N_EVERY_BIN * IMAGE_SIZE,
                          label + (i - 1) * CIFAR_10_N_EVERY_BIN * LABEL_SIZE);
    }

    char *dst_image_path = new char[200];
    char *dst_label_path = new char[200];

    for (int i = 0; i < train_total_blocks; ++i) {
        sprintf(dst_image_path, "%s/cifar10_train_image_%d.bin", output_dir, i);
        sprintf(dst_label_path, "%s/cifar10_train_label_%d.bin", output_dir, i);

        int block_size = (int) floor(CIFAR_TRAIN_BATCH * CIFAR_10_N_EVERY_BIN / train_total_blocks);
        int start = block_size * i;
        int end = block_size * (i + 1);
        if (i == train_total_blocks - 1) {
            end = CIFAR_10_N_EVERY_BIN * CIFAR_TRAIN_BATCH;
        }
        size_t n = (size_t) (end - start);

        printf("start %d, end %d, n %zu\n", start, end, n);

        Dumper train_data_dumper(n, CIFAR_C, CIFAR_H, CIFAR_W, dst_image_path);
        Dumper train_label_dumper(n, 1, 1, 1, dst_label_path);

        train_data_dumper.dump_image(data + (start * IMAGE_SIZE), n * IMAGE_SIZE);
        train_label_dumper.dump_label(label + (start * LABEL_SIZE), n * LABEL_SIZE);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // dump test batch

    sprintf(bin_file_path, "%s/test_batch.bin", cifar_dir);
    read_one_bin_file(bin_file_path, data, label);

    for (int i = 0; i < test_total_blocks; ++i) {
        sprintf(dst_image_path, "%s/cifar10_test_image_%d.bin", output_dir, i);
        sprintf(dst_label_path, "%s/cifar10_test_label_%d.bin", output_dir, i);

        int block_size = (int) floor(CIFAR_10_N_EVERY_BIN / test_total_blocks);
        int start = block_size * i;
        int end = block_size * (i + 1);
        if (i == test_total_blocks - 1) {
            end = CIFAR_10_N_EVERY_BIN;
        }
        size_t n = (size_t) (end - start);

        printf("start %d, end %d, n %zu\n", start, end, n);

        Dumper test_data_dumper(n, CIFAR_C, CIFAR_H, CIFAR_W, dst_image_path);
        Dumper test_label_dumper(n, 1, 1, 1, dst_label_path);

        read_one_bin_file(bin_file_path, data, label);
        test_data_dumper.dump_image(data + (start * IMAGE_SIZE), n * IMAGE_SIZE);
        test_label_dumper.dump_label(label + (start * LABEL_SIZE), n * LABEL_SIZE);
    }


    delete[] data;
    delete[] label;
    delete[] bin_file_path;
    delete[] dst_image_path;
    delete[] dst_label_path;
}

//void load_test(const char *cifar_dir, const char *output_dir,
//               const int train_total_blocks,
//               const int test_total_blocks) {
//    char *data = new char[CIFAR_10_N_EVERY_BIN * CIFAR_TRAIN_BATCH * IMAGE_SIZE];
//    char *label = new char[CIFAR_10_N_EVERY_BIN * CIFAR_TRAIN_BATCH * LABEL_SIZE];
//
//    char *bin_file_path = new char[200];
//    for (int i = 1; i <= CIFAR_TRAIN_BATCH; ++i) {
//        sprintf(bin_file_path, "%s/data_batch_%d.bin", cifar_dir, i);
//        int image_offset = (i - 1) * CIFAR_10_N_EVERY_BIN * IMAGE_SIZE;
//        int label_offset = (i - 1) * CIFAR_10_N_EVERY_BIN * LABEL_SIZE;
//        read_one_bin_file(bin_file_path, data + image_offset, label + label_offset);
//    }
//
//    /////////////////////////////////////////////////////////////////////////////////////////////
//    char *dst_image_path = new char[200];
//    char *dst_label_path = new char[200];
//    float *batch_data = new float[IMAGE_SIZE * CIFAR_10_N_EVERY_BIN *CIFAR_TRAIN_BATCH];
//    float *batch_label = new float[LABEL_SIZE * CIFAR_10_N_EVERY_BIN *CIFAR_TRAIN_BATCH];
//
//    int n=0;
//
//    for (int i = 0; i < train_total_blocks; ++i) {
//        sprintf(dst_image_path, "%s/cifar10_train_image_%d.bin", output_dir, i);
//        sprintf(dst_label_path, "%s/cifar10_train_label_%d.bin", output_dir, i);
//
//        parallel_reader_t<float> reader(dst_image_path, dst_label_path, 1, (size_t) floor(CIFAR_10_N_EVERY_BIN * CIFAR_TRAIN_BATCH / train_total_blocks));
//        reader.get_batch(batch_data+(n*IMAGE_SIZE), batch_label+(n*LABEL_SIZE));
//
//        n+= reader.get_n();
//
////        assert(reader.get_n() == CIFAR_10_N_EVERY_BIN * CIFAR_TRAIN_BATCH);
//        assert(reader.get_c() == CIFAR_C);
//        assert(reader.get_h() == CIFAR_H);
//        assert(reader.get_w() == CIFAR_W);
//
//    }
//
//    printf("n == %d\n", n);
//    assert(n == CIFAR_10_N_EVERY_BIN*CIFAR_TRAIN_BATCH);
//
////    for (int i = 0; i < CIFAR_TRAIN_BATCH; ++i) {
//
//        for (int j = 0; j < IMAGE_SIZE * CIFAR_10_N_EVERY_BIN*CIFAR_TRAIN_BATCH; ++j) {
//            assert(batch_data[j] == (float) (unsigned char) data[j]);
//        }
//        for (int j = 0; j < LABEL_SIZE * CIFAR_10_N_EVERY_BIN*CIFAR_TRAIN_BATCH; ++j) {
//            assert(batch_label[j] == (float) (unsigned char) label[j]);
//        }
////    }
//    // test seek to first
////    for (int i = 0; i < IMAGE_SIZE * CIFAR_10_N_EVERY_BIN; ++i) {
////        assert(batch_data[i] == (float) (unsigned char) data[i]);
////    }
//
//
//    /////////////////////////////////////////////////////////////////////////////////////////////
//    sprintf(bin_file_path, "%s/test_batch.bin", cifar_dir);
//
//    read_one_bin_file(bin_file_path, data, label);
//
//    int test_n = 0;
//    for (int i = 0; i < test_total_blocks; ++i) {
//        sprintf(dst_image_path, "%s/cifar10_test_image_%d.bin", output_dir, i);
//        sprintf(dst_label_path, "%s/cifar10_test_label_%d.bin", output_dir, i);
//        DataReader<float> reader1(dst_image_path, dst_label_path, (size_t) (CIFAR_10_N_EVERY_BIN / test_total_blocks));
//        assert(reader1.get_c() == CIFAR_C);
//        assert(reader1.get_h() == CIFAR_H);
//        assert(reader1.get_w() == CIFAR_W);
//
//        reader1.get_batch(batch_data+(test_n*IMAGE_SIZE), batch_label+(test_n*LABEL_SIZE));
//        test_n+=reader1.get_n();
//    }
//
//    printf("test n %d\n", test_n);
//    assert(test_n == CIFAR_10_N_EVERY_BIN);
//
//    // test seek to first
////    for (int k = 0; k < 2; ++k) {
////        reader1.get_batch(batch_data, batch_label, CIFAR_10_N_EVERY_BIN);
//        for (int i = 0; i < IMAGE_SIZE * CIFAR_10_N_EVERY_BIN; ++i) {
//            assert(batch_data[i] == (float) (unsigned char) data[i]);
//        }
//        for (int i = 0; i < LABEL_SIZE * CIFAR_10_N_EVERY_BIN; ++i) {
//            assert(batch_label[i] == (float) (unsigned char) label[i]);
//        }
////    }
//
//    printf("test for cifar10 success\n");
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
    if (argc != 5) {
        fprintf(stderr,
                "Usage:\nconvert_cifar10 [cifar_folder] [binary_output_folder] [train_num_to_split]"
                        " [test_num_to_split]\n");
        exit(-1);
    }
    char *cifar_dir = argv[1];
    char *output_dir = argv[2];
    int train_total_blocks = atoi(argv[3]);
    int test_total_blocks = atoi(argv[4]);

    convert_cifar(cifar_dir, output_dir, train_total_blocks, test_total_blocks);
//#ifdef DEBUG
//    load_test(cifar_dir, output_dir, train_total_blocks, test_total_blocks);
//#endif
    printf("convert finish, output dir is : %s\n", output_dir);
}
