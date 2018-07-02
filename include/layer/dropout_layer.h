#if !defined(_DROPOUT_H_)
#define _DROPOUT_H_
#include <switch.h>
#include <tensor.h>
#include <assert.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons{

template <class value_type>
class dropout_layer_t:base_network_layer_t<value_type>
{
private:
    const value_type one;
    const value_type zero;
    const double dropout_rate;
    size_t  state_size_bytes;
    size_t  buff_size_bytes;
    tensor_t<value_type>* dropout_state;
    tensor_t<value_type>* dropout_buff;
    cudnnDropoutDescriptor_t dropout_desc;
    uint64_t seed = 1337ull;
    

    
public:
    dropout_layer_t(double dr):one(1), zero(0), buff_size_bytes(0), state_size_bytes(0), dropout_rate(dr), base_network_layer_t<value_type>(DROPOUT)
    {
        checkCUDNN( cudnnCreateDropoutDescriptor( &(this->dropout_desc) ) );
    }
    ~dropout_layer_t()
    {
        checkCUDNN( cudnnDestroyDropoutDescriptor( this->dropout_desc ) );
    }
    
    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    std::vector<value_type> forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);
    
    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);

    void gen_description(char* buff, size_t* len_in_byte) {
        size_t meta_len_in_byte;
        this->gen_meta_description(buff, &meta_len_in_byte);

        value_type dr = dropout_rate;
        memcpy(buff+meta_len_in_byte, &dr, sizeof(value_type));

        *len_in_byte = meta_len_in_byte + sizeof(value_type);
    }
};
    
} //SuperNeurons namespace

#endif

