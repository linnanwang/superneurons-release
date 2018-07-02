#if !defined(_CUDNN_FULLY_CONNECTED_H_)
#define _CUDNN_FULLY_CONNECTED_H_
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons{
    
template <class value_type>
class fully_connected_layer_t:base_network_layer_t<value_type>
{
private:
    //cudnn setup
    const size_t output_dim;
    const value_type one;
    const value_type zero;
    
//    double std;
//    double mean;
//    weight_filler_t filler_t;
    initializer_t<value_type> *weight_initializer;
    initializer_t<value_type> *bias_initializer;
    
    tensor_t<value_type>* bias_multiplier = NULL;
    
    void mat_multiply(cublasHandle_t* cublas_h,
                     int m, int n, int k,
                     cublasOperation_t TransA, cublasOperation_t TransB,
                     value_type alpha, value_type beta,
                     value_type* A, int lda,
                     value_type* B, int ldb,
                     value_type* C, int ldc);
    
    void backward_data(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h,   registry_t<value_type> *reg);
    void backward_bias(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h,   registry_t<value_type> *reg);
    void backward_weight(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg);
    
public:
    fully_connected_layer_t(size_t output_dim,
                            initializer_t<value_type> *weight_initializer,
                            bool enable_bias,
                            initializer_t<value_type> *bias_initializer = new constant_initializer_t<float>(0.0) )
    : output_dim(output_dim), one(1), zero(0), weight_initializer(weight_initializer), bias_initializer(bias_initializer),
      base_network_layer_t<value_type>(FC)
    {
        this->enable_bias(enable_bias);
        if(enable_bias) assert(this->bias_initializer != NULL);
    }
    
    ~fully_connected_layer_t() {
    }
    
    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    //CUDNN supports in-place operations on this layer
    void hook( base_network_layer_t<value_type>* prev  = NULL, base_network_layer_t<value_type>* next  = NULL );
    
    std::vector<value_type> forward (network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg);
    
    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg);

    /**
     * Meta Data, out, init, weight, bias
     */
    void gen_description(char* buff, size_t* len_in_byte) {
        size_t meta_len_in_byte, t1, t2;
        size_t SIZE = sizeof(value_type);

        this->gen_meta_description(buff, &meta_len_in_byte);

        value_type _out = output_dim,
            _init = weight_initializer->get_type();

        memcpy(buff + meta_len_in_byte, &_out, SIZE);
        memcpy(buff + meta_len_in_byte + 1 * SIZE, &_init, SIZE);

        this->get_weight()->gen_description(buff + meta_len_in_byte + 2 * SIZE, &t1);
        this->get_bias()->gen_description(buff + meta_len_in_byte + 2 * SIZE + t1, &t2);

        *len_in_byte = meta_len_in_byte + 2 * SIZE + t1 + t2;
    }
};
    
} // SuperNeurons namespace
#endif // _CUDNN_FULLY_CONNECTED_H_



