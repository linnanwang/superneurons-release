//
// Created by ay27 on 17/6/9.
//

#ifndef SUPERNEURONS_PREPROCESS_H
#define SUPERNEURONS_PREPROCESS_H

#include <tensor.h>
#include <vector>
#include <fstream>

namespace SuperNeurons {

template<class value_type>
class base_preprocess_t {
public:
    virtual void transfer(value_type *src, value_type *dst) = 0;

    virtual size_t output_size() = 0;

    virtual size_t input_size() = 0;
};


template<class value_type>
class border_padding_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W, padH, padW;
public:
    border_padding_t(size_t N, size_t C, size_t H, size_t W, size_t padH, size_t padW) :
            N(N), C(C), H(H), W(W), padH(padH), padW(padW) {
        assert(padH > 0);
        assert(padW > 0);
    }

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class random_crop_t : public base_preprocess_t<value_type> {
private:
    size_t src_N, src_C, src_H, src_W, dst_N, dst_C, dst_H, dst_W;
public:
    random_crop_t(size_t src_N, size_t src_C, size_t src_H, size_t src_W,
                  size_t dst_N, size_t dst_C, size_t dst_H, size_t dst_W) :
            src_N(src_N), src_C(src_C), src_H(src_H), src_W(src_W),
            dst_N(dst_N), dst_C(dst_C), dst_H(dst_H), dst_W(dst_W) {
        assert(src_N == dst_N);
        assert(src_C == dst_C);
#ifdef DEBUG
        printf("random_crop: (%zu,%zu,%zu,%zu) -> (%zu,%zu,%zu,%zu)\n",
               src_N, src_C, src_H, src_W, dst_N, dst_C, dst_H, dst_W);
#endif
    }

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};

template<class value_type>
class central_crop_t : public base_preprocess_t<value_type> {
private:
    size_t src_N, src_C, src_H, src_W, dst_N, dst_C, dst_H, dst_W;
public:
    central_crop_t(size_t src_N, size_t src_C, size_t src_H, size_t src_W,
                  size_t dst_N, size_t dst_C, size_t dst_H, size_t dst_W) :
        src_N(src_N), src_C(src_C), src_H(src_H), src_W(src_W),
        dst_N(dst_N), dst_C(dst_C), dst_H(dst_H), dst_W(dst_W) {
        assert(src_N == dst_N);
        assert(src_C == dst_C);
#ifdef DEBUG
        printf("central_crop: (%zu,%zu,%zu,%zu) -> (%zu,%zu,%zu,%zu)\n",
               src_N, src_C, src_H, src_W, dst_N, dst_C, dst_H, dst_W);
#endif
    }

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class random_flip_left_right_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;
public:
    random_flip_left_right_t(size_t N, size_t C, size_t H, size_t W) : N(N), C(C), H(H), W(W) {}

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class random_brightness_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;
    value_type max_delta;
    bool is_src_regularized;
public:
    random_brightness_t(size_t N, size_t C, size_t H, size_t W, value_type max_delta, bool is_src_regularized = false) :
            N(N), C(C), H(H), W(W), max_delta(max_delta), is_src_regularized(is_src_regularized) {
        if (is_src_regularized) {
            assert((max_delta >= -1.0) && (max_delta <= 1.0));
        }
    }

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;

};


template<class value_type>
class random_contrast_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;
    value_type lower, upper;
public:
    random_contrast_t(size_t N, size_t C, size_t H, size_t W, value_type lower, value_type upper) :
            N(N), C(C), H(H), W(W), lower(lower), upper(upper) {
        assert(lower <= upper);
    }

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class per_image_standardization_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;
public:
    per_image_standardization_t(size_t N, size_t C, size_t H, size_t W) :
            N(N), C(C), H(H), W(W) {
    }

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class mean_subtraction_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;

    value_type scale;
    float *mean_value = NULL;
    value_type *channel_mean = NULL;
    const bool is_channel_mean;

public:
    mean_subtraction_t(size_t N_, size_t C_, size_t H_, size_t W_,
                       const char *mean_file_path, value_type scale = 1.0);

    mean_subtraction_t(size_t N_, size_t C_, size_t H_, size_t W_, value_type* channel_mean, value_type scale = 1.0);

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class preprocessor {
private:
    std::vector<base_preprocess_t<value_type> *> processors;
    std::vector<value_type *> tmps;
public:
    preprocessor() {}

    ~preprocessor() {
        for (size_t i = 0; i < tmps.size(); ++i) {
            cudaFree(tmps[i]);
        }
    }

    preprocessor<value_type> *add_preprocess(base_preprocess_t<value_type> *processor);

    size_t input_size() {
        CHECK_NOTNULL(processors[0]);
        return processors[0]->input_size();
    }

    size_t output_size() {
        return processors.back()->output_size();
    }

    // pass through all preprocessors inplace
    void process(value_type *src, value_type* dst);
};


}

#endif //SUPERNEURONS_PREPROCESS_H
