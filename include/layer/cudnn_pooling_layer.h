#if !defined(_CUDNN_POOL_H_)
#define _CUDNN_POOL_H_
#include <switch.h>
#include <tensor.h>
#include <assert.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons{

template <class value_type>
class pool_layer_t:base_network_layer_t<value_type>
{
private:
    const cudnnPoolingMode_t mode;
    const cudnnNanPropagation_t p_nan;
    cudnnPoolingDescriptor_t pool_desc;
    const int vertical_stride;
    const int horizontal_stride;
    const int kernel_height;
    const int kernel_width;
    const int vertical_padding;
    const int horizontal_padding;
    const value_type one;
    const value_type zero;
    
public:
    //hs: horizontal stride
    //vs: vertical   stride
    //kh: kernel     height
    //hw: kernel     width
    pool_layer_t(cudnnPoolingMode_t mode,
                 cudnnNanPropagation_t p_nan,
                 int hs, int vs, int kh, int kw, int hp=0, int vp=0)
    :mode(mode), p_nan(p_nan), horizontal_stride(hs), vertical_stride(vs),
    kernel_height(kh), kernel_width(kw), one(1), zero(0), vertical_padding(vp),
    horizontal_padding(hp), base_network_layer_t<value_type>(POOL)
    {
        //ensure network is set
        checkCUDNN( cudnnCreatePoolingDescriptor( &(this->pool_desc) ) );
    }
    ~pool_layer_t()
    {
        checkCUDNN( cudnnDestroyPoolingDescriptor( this->pool_desc ) );
    }
    
    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    
    std::vector<value_type> forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);
    
    void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg);

    /**
     * value_type: mode, vs, hs, kh, kw, vp, hp
     */
    void gen_description(char *buff, size_t *len_in_byte) {
        size_t meta_len_in_byte;
        this->gen_meta_description(buff, &meta_len_in_byte);

        size_t SIZE = sizeof(value_type);
        value_type _m = mode,
            vs = vertical_stride, hs = horizontal_stride,
            kh = kernel_height, kw = kernel_width,
            vp = vertical_padding, hp = horizontal_padding;

        memcpy(buff + meta_len_in_byte, &_m, SIZE);
        memcpy(buff + meta_len_in_byte + 1 * SIZE, &vs, SIZE);
        memcpy(buff + meta_len_in_byte + 2 * SIZE, &hs, SIZE);
        memcpy(buff + meta_len_in_byte + 3 * SIZE, &kh, SIZE);
        memcpy(buff + meta_len_in_byte + 4 * SIZE, &kw, SIZE);
        memcpy(buff + meta_len_in_byte + 5 * SIZE, &vp, SIZE);
        memcpy(buff + meta_len_in_byte + 6 * SIZE, &hp, SIZE);

        *len_in_byte = meta_len_in_byte + 7 * SIZE;

        printf("Pool: %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", _m, vs, hs, kh, kw, vp, hp);
    }
};
    
} //SuperNeurons namespace

#endif

