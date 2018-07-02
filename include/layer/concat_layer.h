//
// Created by ay27 on 17/6/20.
//

#ifndef SUPERNEURONS_CONCAT_LAYER_H
#define SUPERNEURONS_CONCAT_LAYER_H

#include <switch.h>
#include <assert.h>
#include <tensor.h>
#include <util/common.h>
#include <layer/base_structure_layer.h>

namespace SuperNeurons {

template<class value_type>
class concat_layer_t : base_structure_t<value_type> {
private:
    //cudnn setup
    const value_type zero;
    const value_type one;

public:
    concat_layer_t() : one(1), zero(0), base_structure_t<value_type>(CONCAT) {
        this->set_structure_type(JOIN);
    }

    ~concat_layer_t() {

    }

    void forward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h);

    void backward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h);

    std::vector<value_type>
    forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg);

    void backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg);

    void gen_description(char* buff, size_t* len_in_byte) {
        this->gen_meta_description(buff, len_in_byte);
    }
};

}
#endif //SUPERNEURONS_CONCAT_LAYER_H
