#if !defined(_CUDNN_BATCH_NORMALIZATION_H_)
#define _CUDNN_BATCH_NORMALIZATION_H_
#include <switch.h>
#include <tensor.h>
#include <assert.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons{

template <class value_type>
class batch_normalization_layer_t:base_network_layer_t<value_type>
{
private:
    const value_type one;
    const value_type zero;
    const value_type negative_one;

    double iter;
    const double epsilon;
    const cudnnBatchNormMode_t mode;

    //forward tensor_ts
    tensor_t<value_type>* resultRunningMean;
    tensor_t<value_type>* resultRunningVariance;
    tensor_t<value_type>* resultSaveMean;
    tensor_t<value_type>* resultSaveInvVariance;


public:
    batch_normalization_layer_t(cudnnBatchNormMode_t m, double eps)
    :mode(m), one(1.0f), zero(0.0f), iter(1), epsilon(eps), negative_one(-1.0f),
     base_network_layer_t<value_type>(BN)
    {

    }

    ~batch_normalization_layer_t()
    {

    }

    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );

    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );

    std::vector<value_type> forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);

    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);

    /**
     * Meta: meta description
     * value_type: epsilon
     * value_type: BN mode
     * tensor_t: weight gamma
     * tensor_t: bias beta
     */
    void gen_description(char* buff, size_t* len_in_byte) {
        size_t meta_len_in_byte, t1, t2;

        this->gen_meta_description(buff, &meta_len_in_byte);

        size_t SIZE = sizeof(value_type);
        value_type eps = epsilon;
        value_type bn_mode = mode;

        memcpy(buff + meta_len_in_byte, &eps, SIZE);
        memcpy(buff + meta_len_in_byte + SIZE, &mode, SIZE);

        this->get_weight()->gen_description(buff + meta_len_in_byte + 2 * SIZE, &t1);
        this->get_bias()->gen_description(buff + meta_len_in_byte + 2 * SIZE + t1, &t2);

        *len_in_byte = meta_len_in_byte + 2 * SIZE + t1 + t2;

    }
};

} //SuperNeurons namespace

#endif

