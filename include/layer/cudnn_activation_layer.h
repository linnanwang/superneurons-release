#if !defined(_CUDNN_ACT_H_)
#define _CUDNN_ACT_H_
#include <switch.h>
#include <assert.h>
#include <tensor.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons{
    
template <class value_type>
class act_layer_t:base_network_layer_t<value_type>
{
private:
    //cudnn setup
    const value_type zero;
    const value_type one;
    const cudnnActivationMode_t mode;
    const cudnnNanPropagation_t p_nan;
    cudnnActivationDescriptor_t act_desc;
    
public:
    act_layer_t(cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU,
                cudnnNanPropagation_t p_nan = CUDNN_NOT_PROPAGATE_NAN)
    :mode(mode), one(1), zero(0), p_nan(p_nan), base_network_layer_t<value_type>(ACT)
    {
    }
    
    ~act_layer_t() {
        checkCUDNN( cudnnDestroyActivationDescriptor( this->act_desc ) );
    }
    
    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
        
    std::vector<value_type> forward (network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);
    
    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);

    void gen_description(char* buff, size_t* len_in_byte) {
        size_t meta_len_in_byte;
        this->gen_meta_description(buff, &meta_len_in_byte);

//        typedef enum
//        {
//            CUDNN_ACTIVATION_SIGMOID      = 0,
//            CUDNN_ACTIVATION_RELU         = 1,
//            CUDNN_ACTIVATION_TANH         = 2,
//            CUDNN_ACTIVATION_CLIPPED_RELU = 3
//        } cudnnActivationMode_t;

        value_type _m = mode;
        memcpy(buff+meta_len_in_byte, &_m, sizeof(value_type));

        *len_in_byte = meta_len_in_byte + sizeof(value_type);
    }
};

} // superneuron namespace
#endif // _CUDNN_ACT_H_



