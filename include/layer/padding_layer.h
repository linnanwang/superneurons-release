//
// Created by ay27 on 7/3/17.
//

#ifndef SUPERNEURONS_PADDING_LAYER_H
#define SUPERNEURONS_PADDING_LAYER_H


#include <switch.h>
#include <assert.h>
#include <tensor.h>
#include <util/common.h>
#include <layer/base_network_layer.h>
#include <layer/base_structure_layer.h>


#define index(n, c, h, w, C, H, W) ((((n)*(C)+(c))*(H)+(h))*(W)+(w))

namespace SuperNeurons {

template<class value_type>
class padding_layer_t : base_network_layer_t<value_type> {
private:
    const size_t padC, padH, padW;
public:
    padding_layer_t(const size_t padC, const size_t padH, const size_t padW) : base_network_layer_t<value_type>(PADDING), padC(padC), padH(padH), padW(padW)   {}


    void forward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) override;

    void backward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) override;

    std::vector<value_type> forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
                                    registry_t<value_type> *reg) override;

    void backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
                  registry_t<value_type> *reg) override;

    void gen_description(char* buff, size_t* len_in_byte) {
        size_t meta_len_in_byte;
        size_t SIZE = sizeof(value_type);

        this->gen_meta_description(buff, &meta_len_in_byte);

        value_type pc = padC, ph = padH, pw = padW;

        memcpy(buff + meta_len_in_byte, &pc, SIZE);
        memcpy(buff + meta_len_in_byte + 1 * SIZE, &ph, SIZE);
        memcpy(buff + meta_len_in_byte + 2 * SIZE, &pw, SIZE);

        *len_in_byte = meta_len_in_byte + 3 * SIZE;
    }
};

}

#endif //SUPERNEURONS_PADDING_LAYER_H
