//
// Created by ay27 on 17/4/8.
//

#include <util/image_reader.h>

using namespace std;
using namespace SuperNeurons;

void read_meta(ifstream *file, size_t *N, size_t *C, size_t *H, size_t *W) {
    char *meta = new char[MAGIC_LEN];

    file->read(meta, MAGIC_LEN);
    *N = (uint64_t) *((uint64_t *) (meta));
    *C = (uint64_t) *((uint64_t *) (meta + META_ATOM));
    *H = (uint64_t) *((uint64_t *) (meta + META_ATOM * 2));
    *W = (uint64_t) *((uint64_t *) (meta + META_ATOM * 3));

    delete[] meta;
}

void compute_mean(const char *input_binary_path, const char *output_mean_path) {
    ifstream input_binary(input_binary_path, ios::in | ios::binary);
    ofstream output_mean(output_mean_path, ios::out | ios::binary);
    assert(input_binary);
    assert(output_mean);

    size_t N, C, H, W;

    read_meta(&input_binary, &N, &C, &H, &W);
    printf("read meta finish, N=%zu, C=%zu, H=%zu, W=%zu\n", N, C, H, W);

    size_t image_size = C * H * W;
    printf("image size = %zu\n", image_size);
    uint8_t *data = new uint8_t[image_size];
    float *sum_matrix = new float[image_size];
    for (size_t i = 0; i < image_size; ++i) {
        sum_matrix[i] = 0;
    }

    for (size_t n = 0; n < N; ++n) {
        input_binary.read((char *) data, image_size);
        assert((size_t)input_binary.gcount() == image_size);
        for (size_t i = 0; i < image_size; ++i) {
            sum_matrix[i] = sum_matrix[i] * ((float)(n) / (float)(n + 1)) + ((float) data[i] / (float)(n + 1));
        }
    }
    // read finish
    input_binary.close();

    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                printf("%1.1f ", sum_matrix[((c * H) + h) * W + w]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // write mean to file
    output_mean.write((char *) sum_matrix, image_size * sizeof(float));
    output_mean.close();

    printf("mean file write finish\n");

    // compute channel mean
    for (size_t c = 0; c < C; ++c) {
        float channel_mean = 0;
        for (size_t i = 0; i < H * W; ++i) {
            channel_mean += sum_matrix[c * H * W + i];
        }
        channel_mean /= (float) (H * W);
        printf("mean_value channel %zu : %f\n", c, channel_mean);
    }

    delete[] data;
    delete[] sum_matrix;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage:\n"
                "compute_image_mean [input_binary_file] [output_mean_file]\n");
        exit(-1);
    }

    char *input_binary_path = argv[1];
    char *output_mean_path = argv[2];

    compute_mean(input_binary_path, output_mean_path);
}
