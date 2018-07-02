//
// Created by ay27 on 17/3/7.
//
#include <ios>
#include <fstream>
#include <superneurons.h>
#include <util/image_reader.h>
#include <util/binary_dumper.h>

using namespace std;
using namespace SuperNeurons;

typedef unsigned char value_type;

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_meta_data(ifstream *image_file, ifstream *label_file, uint32_t *num_items,
                    uint32_t *rows, uint32_t *cols) {
    uint32_t magic, num_labels;

    image_file->read(reinterpret_cast<char *>(&magic), 4);
    magic = swap_endian(magic);
    assert(magic == 2051);

    label_file->read(reinterpret_cast<char *>(&magic), 4);
    magic = swap_endian(magic);
    assert(magic == 2049);

    image_file->read(reinterpret_cast<char *>(num_items), 4);
    *num_items = swap_endian(*num_items);
    label_file->read(reinterpret_cast<char *>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    assert((*num_items) == (num_labels));

    image_file->read(reinterpret_cast<char *>(rows), 4);
    *rows = swap_endian(*rows);
    image_file->read(reinterpret_cast<char *>(cols), 4);
    *cols = swap_endian(*cols);
}

void convert_file(const char *src_image_path, const char *src_label_path,
                  const char *dst_image_path, const char *dst_label_path) {
    ifstream image_file(src_image_path, ios::in | ios::binary);
    ifstream label_file(src_label_path, ios::in | ios::binary);
    if (!image_file) {
        printf("not found file \"%s\", skip\n", src_image_path);
        return;
    }
    if (!label_file) {
        printf("not found file \"%s\", skip\n", src_label_path);
        return;
    }

    uint32_t num_items, rows, cols;
    read_meta_data(&image_file, &label_file, &num_items, &rows, &cols);
    size_t total = num_items * rows * cols;

    image_t *images = (image_t *) malloc(total);
    image_file.read((char *) images, total);
    Dumper data_dumper(num_items, 1, rows, cols, dst_image_path);
    data_dumper.dump_image((const char *) images, num_items * rows * cols);

    char *labels = (char *) malloc(num_items);
    label_file.read(labels, num_items);
    Dumper label_dumper(num_items, 1, 1, 1, dst_label_path);
    label_dumper.dump_label(labels, num_items);

    delete[] images;
    delete[] labels;
}

void convert_mnist(const char *src_dir, const char *dst_dir) {
    char src_image[200];
    char src_label[200];
    char dst_image[200];
    char dst_label[200];
    sprintf(src_image, "%s/train-images-idx3-ubyte", src_dir);
    sprintf(src_label, "%s/train-labels-idx1-ubyte", src_dir);
    sprintf(dst_image, "%s/mnist_train_image.bin", dst_dir);
    sprintf(dst_label, "%s/mnist_train_label.bin", dst_dir);

    convert_file(src_image, src_label, dst_image, dst_label);

    sprintf(src_image, "%s/t10k-images-idx3-ubyte", src_dir);
    sprintf(src_label, "%s/t10k-labels-idx1-ubyte", src_dir);
    sprintf(dst_image, "%s/mnist_test_image.bin", dst_dir);
    sprintf(dst_label, "%s/mnist_test_label.bin", dst_dir);
    convert_file(src_image, src_label, dst_image, dst_label);
}

template<typename value_type>
void print_image(value_type *data, value_type *label) {
    printf("%d\n", (unsigned char) label[0]);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            printf("%d ", (bool)((unsigned char) data[i * 28 + j]));
        }
        printf("\n");
    }
}

void load_file_test(const char *src_image_path, const char *src_label_path, const char *dst_image_path,
                    const char *dst_label_path) {
    ifstream image_file(src_image_path, ios::in | ios::binary);
    ifstream label_file(src_label_path, ios::in | ios::binary);
    if (!image_file) {
        printf("not found file \"%s\", skip\n", src_image_path);
        return;
    }
    if (!label_file) {
        printf("not found file \"%s\", skip\n", src_label_path);
        return;
    }

    uint32_t num_items, rows, cols;
    read_meta_data(&image_file, &label_file, &num_items, &rows, &cols);
    uint32_t total = num_items * rows * cols;

    char *images = (char *) malloc(total);
    char *labels = (char *) malloc(num_items);

    image_file.read(images, total);
    label_file.read(labels, num_items);

//    printf("source no.1 and no.100 image:\n");
//    print_image(images, labels);
//    print_image(images + 100 * (28 * 28), labels + 100);

    /////////////////////////////////////////////////////////////////////////

    parallel_reader_t<float> reader(dst_image_path, dst_label_path, 1, num_items, 1, 28, 28);
    assert(reader.getN() == num_items);
    assert(reader.getC() == 1);
    assert(reader.getH() == rows);
    assert(reader.getW() == cols);

    float *data = (float *) malloc(total * sizeof(float));
    float *ll = (float *) malloc(num_items * sizeof(float));
//    reader.get_batch(data, ll);

    print_image(data, ll);
    print_image(data + 100 * (28 * 28), ll + 100);

//    reader.get_batch(data, ll);

    print_image(data, ll);
    print_image(data + 100 * (28 * 28), ll + 100);

    for (uint32_t i = 0; i < total; ++i) {
        assert(data[i] == (float) (unsigned char) images[i]);
    }
    for (uint32_t i = 0; i < num_items; ++i) {
        assert(ll[i] == (float) (unsigned char) labels[i]);
    }

    printf("test success\n");

    delete[] images;
    delete[] labels;
    delete[] data;
    delete[] ll;
}

void load_test(const char *src_dir, const char *dst_dir) {
    char src_image[200];
    char src_label[200];
    char dst_image[200];
    char dst_label[200];
    sprintf(src_image, "%s/train-images-idx3-ubyte", src_dir);
    sprintf(src_label, "%s/train-labels-idx1-ubyte", src_dir);
    sprintf(dst_image, "%s/mnist_train_image.bin", dst_dir);
    sprintf(dst_label, "%s/mnist_train_label.bin", dst_dir);

    load_file_test(src_image, src_label, dst_image, dst_label);

    sprintf(src_image, "%s/t10k-images-idx3-ubyte", src_dir);
    sprintf(src_label, "%s/t10k-labels-idx1-ubyte", src_dir);
    sprintf(dst_image, "%s/mnist_test_image.bin", dst_dir);
    sprintf(dst_label, "%s/mnist_test_label.bin", dst_dir);
    load_file_test(src_image, src_label, dst_image, dst_label);
}

int main(int argc, char **argv) {
#ifdef DEBUG
    printf("debug mode\n");
#endif
    if (argc != 3) {
        fprintf(stderr,
                "Usage:\nconvert_mnist [mnist_folder] [binary_output_folder]\n");
        exit(-1);
    }
    char *src_dir = argv[1];
    char *dst_dir = argv[2];

    convert_mnist(src_dir, dst_dir);
#ifdef DEBUG
    load_test(src_dir, dst_dir);
#endif
    printf("convert finish, output dir is : %s\n", dst_dir);

}
