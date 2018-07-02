#if !defined(_CUDNN_CONV_H_)
#define _CUDNN_CONV_H_
#include <switch.h>
#include <util/common.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons{

template <class value_type>
class conv_layer_t:base_network_layer_t<value_type>
{
private:
    //conv param
    int stride;
    int scale_h;
    int scale_w;
    int padding_h;
    int padding_w;
    int num_output;
    int kernel_h, kernel_w;
    const int conv_dims;
    const value_type one;
    const value_type zero;

//    double std;
//    double mean;
    initializer_t<value_type> *weight_initializer;
    initializer_t<value_type> *bias_initializer;

    bool is_first_forward = false;
    bool is_first_backward = false;

//    weight_filler_t filler_t;

    //auxiliary conv space
    size_t      f_conv_buff_size;
    tensor_t<value_type>* f_conv_buff;
    size_t      b_conv_buff_size;
    tensor_t<value_type>* b_conv_buff;
    size_t      filter_buff_size;
    tensor_t<value_type>* filter_buff;

    //cudnn param
    cudnnConvolutionDescriptor_t    conv_desc;
    cudnnConvolutionFwdAlgo_t       f_conv_alg;
    cudnnConvolutionBwdDataAlgo_t   b_conv_alg;
    cudnnConvolutionBwdFilterAlgo_t filter_alg;
    cudnnFilterDescriptor_t         filter_desc;
    cudnnDataType_t                 cudnn_data_type;
    const cudnnConvolutionMode_t     cudnn_conv_mode;

    void createDesc() {
        checkCUDNN( cudnnCreateConvolutionDescriptor(&(this->conv_desc)) );
        checkCUDNN( cudnnCreateFilterDescriptor(&(this->filter_desc)) );
    }

    cudnnConvolutionFwdAlgoPerf_t search_fwd_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    cudnnConvolutionBwdDataAlgoPerf_t search_bwd_data_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    cudnnConvolutionBwdFilterAlgoPerf_t search_bwd_filter_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );

    void find_best_fwd_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );
    void find_best_bwd_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );

public:
    conv_layer_t(size_t num_output_,
                 size_t kernel_h_,
                 size_t kernel_w_,
                 size_t stride_,
                 size_t padding_h_,
                 size_t padding_w_,
                 initializer_t<value_type> *_initializer,
//                 weight_filler_t ft,
//                 double m,
//                 double s,
                 bool with_bias,
                 initializer_t<value_type> *bias_initializer = new constant_initializer_t<float>(0.0)): scale_h(1), scale_w(1), conv_dims(2), one(1), zero(0),
                                   cudnn_conv_mode(CUDNN_CROSS_CORRELATION),
                                   weight_initializer(_initializer),
                                   bias_initializer(bias_initializer),
                                   base_network_layer_t<value_type>(CONV)
    {
        //setup network
        this->enable_bias(with_bias);

        assert(num_output_  >= 1);
        assert(kernel_h_    >= 1);
        assert(kernel_w_    >= 1);
        assert(stride_      >= 1);
        //conv param
        this->stride      = stride_;
        this->num_output  = num_output_;
        this->kernel_h    = kernel_h_;
        this->kernel_w    = kernel_w_;
        this->padding_h   = padding_h_;
        this->padding_w   = padding_w_;

        switch (sizeof(value_type))
        {
            case 2: cudnn_data_type = CUDNN_DATA_HALF;   break;
            case 4: cudnn_data_type = CUDNN_DATA_FLOAT;  break;
            case 8: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }

        //cudnn
        createDesc();
    }

    conv_layer_t(size_t num_output_,
                 size_t kernel_size_,
                 size_t stride_,
                 size_t padding_h_,
                 size_t padding_w_,
                 initializer_t<value_type> *weight_initializer,
                 bool with_bias,
                 initializer_t<value_type> *bias_initializer = new constant_initializer_t<float>(0.0) ): scale_h(1), scale_w(1), conv_dims(2), one(1), zero(0),
                                   cudnn_conv_mode(CUDNN_CROSS_CORRELATION), weight_initializer(weight_initializer), bias_initializer(bias_initializer), base_network_layer_t<value_type>(CONV)
    {
        //setup network
        this->enable_bias(with_bias);
        assert(bias_initializer != NULL);

        assert(num_output_  >= 1);
        assert(kernel_size_ >= 1);
        assert(stride_      >= 1);
        //conv param
        this->stride      = stride_;
        this->num_output  = num_output_;
        this->kernel_h    = kernel_size_;
        this->kernel_w    = kernel_size_;
        this->padding_h   = padding_h_;
        this->padding_w   = padding_w_;

        switch (sizeof(value_type))
        {
            case 2: cudnn_data_type = CUDNN_DATA_HALF;   break;
            case 4: cudnn_data_type = CUDNN_DATA_FLOAT;  break;
            case 8: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }

        //cudnn
        createDesc();
    }

    ~conv_layer_t() {
        checkCUDNN( cudnnDestroyConvolutionDescriptor( this->conv_desc ));
        checkCUDNN( cudnnDestroyFilterDescriptor( this->filter_desc ));
    }

    void forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );

    void backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h );

    std::vector<value_type> forward (network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg);

    void backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg);

    /**
     * Meta data
     * value_type num_output;
     * value_type kernel_h, kernel_w;
     * value_type stride;
       value_type padding_h, padding_w;
       value_type weight_initializer
       tensor_t: weight, bias
     */
    void gen_description(char *buff, size_t *len_in_byte) {
        size_t meta_len_in_byte, t1, t2;
        size_t SIZE = sizeof(value_type);

        this->gen_meta_description(buff, &meta_len_in_byte);

        value_type _out = num_output,
            _kh = kernel_h, _kw = kernel_w,
            _sd = stride,
            _ph = padding_h, _pw = padding_w,
            _init = weight_initializer->get_type();

        memcpy(buff + meta_len_in_byte, &_out, SIZE);
        memcpy(buff + meta_len_in_byte + 1 * SIZE, &_kh, SIZE);
        memcpy(buff + meta_len_in_byte + 2 * SIZE, &_kw, SIZE);
        memcpy(buff + meta_len_in_byte + 3 * SIZE, &_sd, SIZE);
        memcpy(buff + meta_len_in_byte + 4 * SIZE, &_ph, SIZE);
        memcpy(buff + meta_len_in_byte + 5 * SIZE, &_pw, SIZE);
        memcpy(buff + meta_len_in_byte + 6 * SIZE, &_init, SIZE);

        printf("%.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", _out, _kh, _kw, _sd, _ph, _pw, _init);
        printf("(%zu %zu %zu %zu) (%zu %zu %zu %zu)\n",
               this->get_weight()->get_N(), this->get_weight()->get_C(), this->get_weight()->get_H(), this->get_weight()->get_W(),
               this->get_bias()->get_N(), this->get_bias()->get_C(), this->get_bias()->get_H(), this->get_bias()->get_W());

        this->get_weight()->gen_description(buff + meta_len_in_byte + 7 * SIZE, &t1);
        this->get_bias()->gen_description(buff + meta_len_in_byte + 7 * SIZE + t1, &t2);

        *len_in_byte = meta_len_in_byte + 7 * SIZE + t1 + t2;
    }
};

} // superneuron namespace
#endif // _CUDNN_CONV_H_
