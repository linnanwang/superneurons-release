//
// Created by ay27 on 17/6/10.
//

#include <util/preprocess.h>
#include <util/common.h>
#include <util/print_util.h>

namespace SuperNeurons {

template<class value_type>
void border_padding_t<value_type>::transfer(value_type *src, value_type *dst) {
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            // padding top
            size_t pH = H + padH + padH, pW = W + padW + padW;
            for (size_t h = 0; h < padH; ++h) {
                for (size_t w = 0; w < W + padW + padW; ++w) {
                    dst[((n * C + c) * pH + h) * pW + w] = 0.0;
                }
            }
            for (size_t h = 0; h < H; ++h) {
                // padding left
                for (size_t w = 0; w < padW; ++w) {
                    dst[((n * C + c) * pH + padH + h) * pW + w] = 0.0;
                }
                for (size_t w = 0; w < W; ++w) {
                    dst[((n * C + c) * pH + padH + h) * pW + padW + w] =
                            src[((n * C + c) * H + h) * W + w];
                }
                // padding right
                for (size_t w = 0; w < padW; ++w) {
                    dst[((n * C + c) * pH + padH + h) * pW + padW + W + w] = 0.0;
                }
            }
            // padding bottom
            for (size_t h = 0; h < padH; ++h) {
                for (size_t w = 0; w < W + padW + padW; ++w) {
                    dst[((n * C + c) * pH + padH + H + h) * pW + w] = 0.0;
                }
            }
        }
    }

#ifdef DEBUG_PRE
    print_array(src, N, C, H, W, "board padding src:");
    print_array(dst, N, C, H + 2 * padH, W + 2 * padW, "board padding dst:");
#endif
}

template<class value_type>
size_t border_padding_t<value_type>::output_size() {
    return N * C * (H + 2 * padH) * (W + 2 * padW);
}

template<class value_type>
size_t border_padding_t<value_type>::input_size() {
    return N * C * H * W;
}

template<class value_type>
void random_crop_t<value_type>::transfer(value_type *src, value_type *dst) {
    size_t limitH = src_H - dst_H + 1;
    size_t limitW = src_W - dst_W + 1;
    unsigned seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    for (size_t n = 0; n < dst_N; ++n) {

        // image independent
        size_t offsetH = rand() % limitH;
        size_t offsetW = rand() % limitW;


        for (size_t c = 0; c < dst_C; ++c) {
            for (size_t h = 0; h < dst_H; ++h) {
                for (size_t w = 0; w < dst_W; ++w) {
                    dst[((((n * dst_C) + c) * dst_H) + h) * dst_W + w] =
                            src[((((n * src_C) + c) * src_H) + h + offsetH) * src_W + w + offsetW];
                }
            }
        }
    }

#ifdef DEBUG_PRE
    print_array(src, src_N, src_C, src_H, src_W, "random crop src:");
    print_array(dst, dst_N, dst_C, dst_H, dst_W, "random crop dst:");
#endif
}

template<class value_type>
size_t random_crop_t<value_type>::output_size() {
    return dst_N * dst_C * dst_H * dst_W;
}

template<class value_type>
size_t random_crop_t<value_type>::input_size() {
    return src_N * src_C * src_H * src_W;
}

template<class value_type>
void central_crop_t<value_type>::transfer(value_type *src, value_type *dst) {
    size_t offsetH = (src_H-dst_H) >> 1;
    size_t offsetW = (src_W-dst_W) >> 1;

    for (size_t n = 0; n < dst_N; ++n) {

        for (size_t c = 0; c < dst_C; ++c) {
            for (size_t h = 0; h < dst_H; ++h) {
                for (size_t w = 0; w < dst_W; ++w) {
                    dst[((((n * dst_C) + c) * dst_H) + h) * dst_W + w] =
                        src[((((n * src_C) + c) * src_H) + h + offsetH) * src_W + w + offsetW];
                }
            }
        }
    }

#ifdef DEBUG_PRE
    print_array(src, src_N, src_C, src_H, src_W, "central crop src:");
    print_array(dst, dst_N, dst_C, dst_H, dst_W, "central crop dst:");
    printf("offsetH %zu, offsetW %zu\n", offsetH, offsetW);
#endif
}

template<class value_type>
size_t central_crop_t<value_type>::output_size() {
    return dst_N * dst_C * dst_H * dst_W;
}

template<class value_type>
size_t central_crop_t<value_type>::input_size() {
    return src_N * src_C * src_H * src_W;
}


template<class value_type>
void random_flip_left_right_t<value_type>::transfer(value_type *src, value_type *dst) {
    unsigned seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    // flip according to the W
    for (size_t n = 0; n < N; ++n) {

        // image independent
        float probability = (float) (rand() / double(RAND_MAX));
        bool do_not_flip = probability < 0.5;
#ifdef DEBUG_PRE
        printf("random_flip:n=%d, p=%f, do_not_flip=%d\n", n, probability, do_not_flip);
#endif
        if (do_not_flip) {
            for (size_t i = 0; i < C * H * W; ++i) {
                dst[n * C * H * W + i] = src[n * C * H * W + i];
            }
            continue;
        }

        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    dst[((((n * C) + c) * H) + h) * W + w] =
                            src[((((n * C) + c) * H) + h) * W + (W - w - 1)];
                }
            }
        }
    }
#ifdef DEBUG_PRE
    print_array(src, N, C, H, W, "random flip src:");
    print_array(dst, N, C, H, W, "random flip dst:");
#endif
}

template<class value_type>
size_t random_flip_left_right_t<value_type>::output_size() {
    return N * C * H * W;
}

template<class value_type>
size_t random_flip_left_right_t<value_type>::input_size() {
    return N * C * H * W;
}

template<class value_type>
void random_brightness_t<value_type>::transfer(value_type *src, value_type *dst) {
    unsigned seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    for (size_t n = 0; n < N; ++n) {

        // image independent
        // rand for [-max_delta, +max_delta]
        value_type delta = (rand() / (double) RAND_MAX) * (max_delta - (-max_delta)) + (-max_delta);
#ifdef DEBUG_PRE
        printf("random_brightness: n=%d, delta=%f, max_delta=%f\n", n, delta, max_delta);
#endif
        for (size_t i = 0; i < C * H * W; ++i) {
            dst[n * C * H * W + i] = src[n * C * H * W + i] + delta;
        }
    }

#ifdef DEBUG_PRE
    print_array(src, N, C, H, W, "random brightness src:");
    print_array(dst, N, C, H, W, "random brightness dst:");
#endif
}

template<class value_type>
size_t random_brightness_t<value_type>::output_size() {
    return N * C * H * W;
}

template<class value_type>
size_t random_brightness_t<value_type>::input_size() {
    return N * C * H * W;
}

template<class value_type>
void random_contrast_t<value_type>::transfer(value_type *src, value_type *dst) {
    unsigned seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    value_type *means = new value_type[C];

    for (size_t n = 0; n < N; ++n) {

        // image independent
        // rand for [lower, upper]
        value_type factor = (rand() / (double) RAND_MAX) * (upper - lower) + lower;

        // compute every channel mean
        for (size_t c = 0; c < C; ++c) {
            means[c] = 0.0;
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    means[c] += src[(((n * C + c) * H) + h) * W + w];
                }
            }
            means[c] /= (value_type) (H * W);
        }


#ifdef DEBUG_PRE
        printf("random_contrast: n=%d, factor=%f\n", n, factor);
        for (int c = 0; c < C; ++c) {
            printf("%6.2f ", means[c]);
        }
        printf("\n");
#endif

        // for every pixel, (x - mean) * contrast_factor + mean
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    dst[(((n * C + c) * H) + h) * W + w] =
                            (src[(((n * C + c) * H) + h) * W + w] - means[c]) * factor + means[c];
                }
            }
        }
    }
    delete[] means;

#ifdef DEBUG_PRE
    print_array(src, N, C, H, W, "random contrast src:");
    print_array(dst, N, C, H, W, "random contrast dst:");
#endif
}

template<class value_type>
size_t random_contrast_t<value_type>::output_size() {
    return N * C * H * W;
}

template<class value_type>
size_t random_contrast_t<value_type>::input_size() {
    return N * C * H * W;
}


template<class value_type>
void per_image_standardization_t<value_type>::transfer(value_type *src, value_type *dst) {
    unsigned seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    for (size_t n = 0; n < N; ++n) {

        // image independent

        // This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
        // of all values in image, and
        // `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.
        // `stddev` is the standard deviation of all values in `image`.

        value_type mean = 0.0;
        double square_mean = 0.0;
        for (size_t i = 0; i < C * H * W; ++i) {
            mean += src[n * C * H * W + i];
            square_mean += src[n * C * H * W + i] * src[n * C * H * W + i];
        }
        mean /= (value_type) (C * H * W);
        square_mean /= (value_type) (C * H * W);

        value_type variance = square_mean - (mean * mean);
        variance = (variance - 0.0 > 0.0001) ? variance : 0.0;     // max(0.0, variance)
        value_type stddev = sqrt(variance);

        value_type min_stddev = 1.0 / sqrt((value_type) (C * H * W));
        value_type pixel_value_scale = (stddev - min_stddev > 0.0001) ? stddev : min_stddev;   // maximum
        value_type pixel_value_offset = mean;

#ifdef DEBUG_PRE
        printf("--------per_image_standardization--------\n");
        printf("mean=%f, square_mean=%f, variance=%f\n", mean, square_mean, variance);
        printf("stddev=%f, min_stddev=%f, pixel_value_scale=%f\n", stddev, min_stddev, pixel_value_scale);
        printf("------------------end--------------------\n");
#endif

        for (size_t i = 0; i < C * H * W; ++i) {
            dst[n * C * H * W + i] = (src[n * C * H * W + i] - pixel_value_offset) / pixel_value_scale;
        }
    }

#ifdef DEBUG_PRE
    print_array(src, N,C,H,W, "standardization src:");
    print_array(dst, N,C,H,W, "standardization dst:");
#endif
}

template<class value_type>
size_t per_image_standardization_t<value_type>::output_size() {
    return N * C * H * W;
}

template<class value_type>
size_t per_image_standardization_t<value_type>::input_size() {
    return N * C * H * W;
}


template<class value_type>
mean_subtraction_t<value_type>::mean_subtraction_t(size_t N_, size_t C_, size_t H_, size_t W_,
                                                   const char *mean_file_path, value_type scale)
        : N(N_), C(C_), H(H_), W(W_), scale(scale), is_channel_mean(false) {
    // read mean from file
    // assume that the mean file store in float format
    mean_value = (float *) malloc(C * H * W * sizeof(float));
    std::ifstream mean_file(mean_file_path, std::ios::in | std::ios::binary);
    if (mean_file.fail()) {
        fprintf(stderr, "failed to open file %s\n", mean_file_path);
        exit(-1);
    }
    mean_file.read((char *) mean_value, C * H * W * sizeof(float));
    mean_file.close();

#ifdef DEBUG_PRE
    print_array(mean_value, 1, C, H, W, "mean matrix:");
#endif
}

template<class value_type>
mean_subtraction_t<value_type>::mean_subtraction_t(size_t N_, size_t C_, size_t H_, size_t W_,
                                                   value_type* channel_mean_, value_type scale)
        : N(N_), C(C_), H(H_), W(W_), scale(scale), is_channel_mean(true), channel_mean(channel_mean_) {

}

template<class value_type>
void mean_subtraction_t<value_type>::transfer(value_type *src, value_type *dst) {
    if (is_channel_mean) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t hw = 0; hw < H * W; ++hw) {
                    dst[(i * C + c) * H * W + hw] = (src[(i * C + c) * H * W + hw] - channel_mean[c]) * scale;
                }
            }
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            for (size_t chw = 0; chw < C * H * W; ++chw) {
                dst[i * C * H * W + chw] = (src[i * C * H * W + chw] - mean_value[chw]) * scale;
            }
        }
    }
#ifdef DEBUG_PRE
    print_array(src, N, C, H, W, "mean subtraction src:");
    print_array(dst, N, C, H, W, "mean subtraction dst:");
#endif
}

template<class value_type>
size_t mean_subtraction_t<value_type>::output_size() {
    return N * C * H * W;
}

template<class value_type>
size_t mean_subtraction_t<value_type>::input_size() {
    return N * C * H * W;
}


template<class value_type>
preprocessor<value_type> *preprocessor<value_type>::add_preprocess(base_preprocess_t<value_type> *processor) {
    processors.push_back(processor);
    value_type *tmp;
    cudaMallocHost(&tmp, processor->output_size() * sizeof(value_type));
    tmps.push_back(tmp);
    return this;
}

template<class value_type>
void preprocessor<value_type>::process(value_type *src, value_type* dst) {
    if (processors.size() == 1) {
        processors[0]->transfer(src, dst);
        return;
    }

    processors[0]->transfer(src, tmps[0]);

    for (size_t i = 1; i < processors.size() - 1; ++i) {
        processors[i]->transfer(tmps[i - 1], tmps[i]);
    }

    processors.back()->transfer(tmps[processors.size() - 2], dst);
}


INSTANTIATE_CLASS(border_padding_t);

INSTANTIATE_CLASS(random_crop_t);

INSTANTIATE_CLASS(central_crop_t);

INSTANTIATE_CLASS(random_flip_left_right_t);

INSTANTIATE_CLASS(random_brightness_t);

INSTANTIATE_CLASS(random_contrast_t);

INSTANTIATE_CLASS(per_image_standardization_t);

INSTANTIATE_CLASS(mean_subtraction_t);

INSTANTIATE_CLASS(preprocessor);

} // namespace SuperNeurons
