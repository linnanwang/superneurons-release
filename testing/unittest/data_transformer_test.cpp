//
// Created by ay27 on 17/4/8.
//

#include <util/data_transformer.h>
#include <util/preprocess.h>
#include "testing.h"

using namespace SuperNeurons;

class data_transformer_test : public TestSuite {
protected:
    void setup() {
        printf("\n");
    }

    void teardown() {
    }
};

ADDTEST(data_transformer_test, test_normal) {
    data_transformer<float> transformer(1, 10, 10);
    uint8_t *src = new uint8_t[10 * 10];
    float *dst = new float[10 * 10];
    for (int i = 0; i < 10 * 10; ++i) {
        src[i] = (uint8_t) i;
        printf("%d ", src[i]);
    }
    printf("\n");

    transformer.transform_one_image(src, dst);
    for (int i = 0; i < 10 * 10; ++i) {
        printf("%1.1f ", dst[i]);
    }
    printf("\n");
}

ADDTEST(data_transformer_test, test_mean_value) {
    const size_t c = 3;
    std::vector<double> mean_value(c);
    for (int i = 0; i < c; ++i) {
        mean_value[i] = (double) (i + 10);
    }
    data_transformer<double> transformer(c, 3, 3, 1.0, &mean_value);

    uint8_t src[c * 3 * 3];
    for (int i = 0; i < c * 3 * 3; ++i) {
        src[i] = (uint8_t) i;
        printf("%d ", src[i]);
    }
    printf("\n");

    double dst[c * 3 * 3];
    transformer.transform_one_image(src, dst);
    for (int i = 0; i < c * 3 * 3; ++i) {
        printf("%1.1f ", dst[i]);
    }
    printf("\n");
}

ADDTEST(data_transformer_test, test_mean_file) {
    const size_t c = 3, h = 10, w = 10;
    const char *path = "/tmp/mean_file";
    float mean_value[c * h * w];
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < w; ++k) {
                mean_value[(i * h + j) * w + k] = (float) (i + 10);
            }
        }
    }

    std::ofstream mean_file(path, std::ios::out | std::ios::binary);
    mean_file.write((char *) mean_value, c * h * w * sizeof(float));
    mean_file.close();

    data_transformer<double> transformer(c, h, w, 1, NULL, path);

    float src[c * h * w];
    for (int i = 0; i < c * h * w; ++i) {
        src[i] = i;
        printf("%1.1f ", src[i]);
    }
    printf("\n");

    float dst[c * h * w];
//    transformer.transform_one_image(src, dst);
//    for (int i = 0; i < c * h * w; ++i) {
//        printf("%1.1f ", dst[i]);
//    }
//    printf("\n");

    mean_subtraction_t<float>* p = new mean_subtraction_t<float>(1,c,h,w, path, 2.0);
    p->transfer(src, dst);
    for (int i = 0; i < c * h * w; ++i) {
        printf("%1.1f ", dst[i]);
    }
    printf("\n");
}
