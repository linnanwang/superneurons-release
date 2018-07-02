#if !defined(_CUDNN_SOFTMAX_H_)
#define _CUDNN_SOFTMAX_H_
#include <math.h>       /* log */
#include <switch.h>
#include <tensor.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons{
    
template <class value_type>
class softmax_layer_t:base_network_layer_t<value_type>
{
private:
    const value_type beta;
    const value_type alpha;
    const cudnnSoftmaxMode_t mode;
    const cudnnSoftmaxAlgorithm_t softmax_alg;
    base_network_layer_t<value_type>* label;

public:
    softmax_layer_t(cudnnSoftmaxAlgorithm_t alg = CUDNN_SOFTMAX_ACCURATE,
                    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE)
    :softmax_alg(alg), mode(CUDNN_SOFTMAX_MODE_INSTANCE), alpha(1), beta(0), base_network_layer_t<value_type>(SOFTMAX)
    {
    }
    
    ~softmax_layer_t()
    {
        
    }
    
    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    std::vector<value_type> forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg );
    
    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg );

    void gen_description(char* buff, size_t* len_in_byte) {
        this->gen_meta_description(buff, len_in_byte);
    }
};
    
} //SuperNeurons namespace

#endif



