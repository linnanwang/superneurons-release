#if !defined(_LRN_H_)
#define _LRN_H_
#include <switch.h>
#include <tensor.h>
#include <assert.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons{

template <class value_type>
class LRN_layer_t:base_network_layer_t<value_type>
{
private:
    cudnnLRNDescriptor_t norm_desc;
    cudnnLRNMode_t LRN_mode;
    const value_type one;
    const value_type zero;
    const unsigned lrnN;
    const double   lrnAlpha;
    const double   lrnBeta;
    const double   lrnK;
    /*
     * Create an instance of LRN (Local Response Normalization) descriptor
     * Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
     */

public:
    LRN_layer_t():one(1), zero(0), LRN_mode(CUDNN_LRN_CROSS_CHANNEL_DIM1), lrnN(5.0f), lrnAlpha(0.0001f), lrnBeta(0.75f), lrnK(1.0f), base_network_layer_t<value_type>(LRN)
    {
        //ensure network is set
        checkCUDNN( cudnnCreateLRNDescriptor( &(this->norm_desc) ) );
    }
    ~LRN_layer_t()
    {
        checkCUDNN( cudnnDestroyLRNDescriptor( this->norm_desc ) );
    }
    
    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    std::vector<value_type> forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);
    
    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);

    void gen_description(char* buff, size_t* len_in_byte) {
        this->gen_meta_description(buff, len_in_byte);
    }
};
    
} //SuperNeurons namespace

#endif

