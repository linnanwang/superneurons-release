#include <layer/cudnn_convolution_layer.h>
#include <util/mem_util.h>
#include <limits>
#include <cudnn.h>
#define CONV_DEBUG

namespace SuperNeurons{


template<class value_type>
cudnnConvolutionFwdAlgoPerf_t conv_layer_t<value_type>::search_fwd_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    int input_l = this->get_input_layer_id();
    int curt_l = this->get_id();
    tensor_t<value_type> *t_in = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type> *f_out = this->get_f_out();

    const int MAX_FWD_ALGO = 8;

    cudnnConvolutionFwdAlgoPerf_t* results = new cudnnConvolutionFwdAlgoPerf_t[MAX_FWD_ALGO];

    float min_time = std::numeric_limits<float>::max();
    cudnnConvolutionFwdAlgoPerf_t best_algo{};
    best_algo.time = -1;

    for (int i = 0; i < MAX_FWD_ALGO; ++i) {
        results[i].algo = (cudnnConvolutionFwdAlgo_t) i;
        results[i].status = cudnnGetConvolutionForwardWorkspaceSize(*cudnn_h,
                                                                       t_in->get_tensor_desc(),
                                                                       this->filter_desc,
                                                                       this->conv_desc,
                                                                       f_out->get_tensor_desc(),
                                                                       results[i].algo,
                                                                       &results[i].memory);
        if (results[i].status != CUDNN_STATUS_SUCCESS || results[i].memory >= query_free_mem()) {
            results[i].time = -1;
            continue;
        }

        this->f_conv_buff->resizeTensor(1,1,1,results[i].memory / sizeof(value_type) + 1);

        if (this->f_conv_buff->get_gpu_ptr() == NULL) {
            results[i].time = -1;
            continue;
        }

        cudaStreamSynchronize(stream_singleton::get_compute_stream());
        cudaDeviceSynchronize();

        double ts = get_cur_time();

        for (int step = 0; step < 5; ++step) {
            checkCUDNN( cudnnConvolutionForward(*(cudnn_h),
                                                &(this->one),
                                                t_in->get_tensor_desc(),
                                                t_in->get_gpu_ptr(),
                                                this->filter_desc,
                                                this->get_weight()->get_gpu_ptr(),
                                                this->conv_desc,
                                                results[i].algo,
                                                this->f_conv_buff->get_gpu_ptr(),
                                                this->f_conv_buff->get_mem_size(),
                                                &(this->zero),
                                                f_out->get_tensor_desc(),
                                                f_out->get_gpu_ptr()) );
            if (get_cur_time() - ts > min_time) {
                break;
            }
        }

        cudaStreamSynchronize(stream_singleton::get_compute_stream());

        results[i].time = (float)(get_cur_time() - ts);

        cudaDeviceSynchronize();

        if (results[i].time > 0 && results[i].time < min_time) {
            min_time = results[i].time;

            memcpy(&best_algo, &(results[i]), sizeof(cudnnConvolutionFwdAlgoPerf_t));
        }
    }

#ifdef CONV_DEBUG
    printf("--------FWD algo -------\n");
    for (int i = 0; i < MAX_FWD_ALGO; ++i) {
//        if (results[i].status != CUDNN_STATUS_SUCCESS) {
//            continue;
//        }
        printf("layer %d for Algo %d: %f time requiring %zu byte memory\n",
               this->get_id(), results[i].algo, results[i].time, results[i].memory);
    }
    printf("\n");

#endif

    delete[] results;

    return best_algo;
}

template<class value_type>
cudnnConvolutionBwdDataAlgoPerf_t conv_layer_t<value_type>::search_bwd_data_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {

    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();

    tensor_t<value_type>* weight = this->get_weight();
    tensor_t<value_type>* dEdD_n = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_c = this->get_b_data();

    const int MAX_BWD_DATA_ALGO = 6;

    cudnnConvolutionBwdDataAlgoPerf_t* results = new cudnnConvolutionBwdDataAlgoPerf_t[MAX_BWD_DATA_ALGO];

    float min_time = std::numeric_limits<float>::max();
    cudnnConvolutionBwdDataAlgoPerf_t best_algo{};
    best_algo.time = -1;

    for (int i = 0; i < MAX_BWD_DATA_ALGO; ++i) {
        results[i].algo = (cudnnConvolutionBwdDataAlgo_t)i;
        results[i].status = cudnnGetConvolutionBackwardDataWorkspaceSize(*cudnn_h,
                                                                            this->filter_desc,
                                                                            this->get_f_out()->get_tensor_desc(),
                                                                            this->conv_desc,
                                                                            this->get_b_data()->get_tensor_desc(),
                                                                            results[i].algo,
                                                                            &results[i].memory);
        if (results[i].status != CUDNN_STATUS_SUCCESS || results[i].memory >= query_free_mem()) {
            results[i].time = -1;
            continue;
        }

        this->b_conv_buff->resizeTensor(1,1,1, results[i].memory / sizeof(value_type) + 1);

        if (this->b_conv_buff->get_gpu_ptr() == NULL) {
            results[i].time = -1;
            continue;
        }

        cudaStreamSynchronize(stream_singleton::get_compute_stream());
        cudaDeviceSynchronize();

        double ts = get_cur_time();
        for (int step = 0; step < 5; ++step) {
            checkCUDNN( cudnnConvolutionBackwardData(*(cudnn_h),
                                                     &(this->one),
                                                     this->filter_desc,
                                                     weight->get_gpu_ptr(),
                                                     dEdD_n->get_tensor_desc(),
                                                     dEdD_n->get_gpu_ptr(),
                                                     this->conv_desc,
                                                     results[i].algo,
                                                     this->b_conv_buff->get_gpu_ptr(),
                                                     this->b_conv_buff->get_mem_size(),
                                                     &(this->zero),
                                                     dEdD_c->get_tensor_desc(),
                                                     dEdD_c->get_gpu_ptr()) );
            if (get_cur_time() - ts > min_time) {
                break;
            }
        }
        cudaStreamSynchronize(stream_singleton::get_compute_stream());

        results[i].time = (float)(get_cur_time() - ts);

        cudaDeviceSynchronize();

        if (results[i].time > 0 && results[i].time < min_time) {
            min_time = results[i].time;

            memcpy(&best_algo, &results[i], sizeof(cudnnConvolutionBwdDataAlgoPerf_t));
        }
    }


#ifdef CONV_DEBUG
    printf("--------BWD data algo -------\n");
    for (int i = 0; i < MAX_BWD_DATA_ALGO; ++i) {
        if (results[i].status != CUDNN_STATUS_SUCCESS) {
            continue;
        }
        printf("layer %d bwd Algo %d: %f time requiring %zu byte memory\n",
               this->get_id(), results[i].algo, results[i].time, results[i].memory);
    }
    printf("\n");

#endif

    delete[] results;

    return best_algo;
}


template<class value_type>
cudnnConvolutionBwdFilterAlgoPerf_t conv_layer_t<value_type>::search_bwd_filter_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {

    int input_l_id  = this->get_input_layer_id();
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* t_in = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_n = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* weight_grad = this->get_weight_grad();

    const int MAX_BWD_FILTER_ALGO = 6;

    cudnnConvolutionBwdFilterAlgoPerf_t* results = new cudnnConvolutionBwdFilterAlgoPerf_t[MAX_BWD_FILTER_ALGO];

    float min_time = std::numeric_limits<float>::max();
    cudnnConvolutionBwdFilterAlgoPerf_t best_algo{};
    best_algo.time = -1;

    for (int i = 0; i < MAX_BWD_FILTER_ALGO; ++i) {
        results[i].algo = (cudnnConvolutionBwdFilterAlgo_t)i;
        results[i].status = cudnnGetConvolutionBackwardFilterWorkspaceSize(*cudnn_h,
                                                                              t_in->get_tensor_desc(),
                                                                              this->get_f_out()->get_tensor_desc(),
                                                                              this->conv_desc,
                                                                              this->filter_desc,
                                                                              results[i].algo,
                                                                              &results[i].memory);
        if (results[i].status != CUDNN_STATUS_SUCCESS || results[i].memory >= query_free_mem()) {
            results[i].time = -1;
            continue;
        }

        this->filter_buff->resizeTensor(1,1,1, results[i].memory / sizeof(value_type) + 1);

        if (this->filter_buff->get_gpu_ptr() == NULL) {
            results[i].time = -1;
            continue;
        }

        cudaStreamSynchronize(stream_singleton::get_compute_stream());
        cudaDeviceSynchronize();

        double ts = get_cur_time();
        for (int step = 0; step < 5; ++step) {
            const value_type one_b = 1.0/(value_type) t_in->get_N();
            checkCUDNN( cudnnConvolutionBackwardFilter(*(cudnn_h),
                                                       &(this->one),
                                                       t_in->get_tensor_desc(),
                                                       t_in->get_gpu_ptr(),
                                                       dEdD_n->get_tensor_desc(),
                                                       dEdD_n->get_gpu_ptr(),
                                                       this->conv_desc,
                                                       results[i].algo,
                                                       this->filter_buff->get_gpu_ptr(),
                                                       this->filter_buff->get_mem_size(),
                                                       &(this->zero),
                                                       this->filter_desc,
                                                       weight_grad->get_gpu_ptr() ) );
            if (get_cur_time() - ts > min_time) {
                break;
            }
        }

        cudaStreamSynchronize(stream_singleton::get_compute_stream());

        results[i].time = (float)(get_cur_time() - ts);

        cudaDeviceSynchronize();


        if (results[i].time > 0 && results[i].time < min_time) {
            min_time = results[i].time;

            memcpy(&best_algo, &results[i], sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));
        }
    }

#ifdef CONV_DEBUG
    printf("--------BWD filter algo -------\n");
    for (int i = 0; i < MAX_BWD_FILTER_ALGO; ++i) {
        if (results[i].status != CUDNN_STATUS_SUCCESS) {
            continue;
        }
        printf("layer %d bwd1 Algo %d: %f time requiring %zu byte memory\n",
               this->get_id(), results[i].algo, results[i].time, results[i].memory);
    }

    printf("\n");

#endif

    delete[] results;

    return best_algo;
}

template<class value_type>
void conv_layer_t<value_type>::find_best_fwd_algo( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {

#ifdef CONV_DEBUG
    printf("free memory : %zu byte\n", query_free_mem());
#endif

    cudnnConvolutionFwdAlgoPerf_t best_algo = search_fwd_algo(reg, cudnn_h);

    if (best_algo.time > 0) {
        this->f_conv_alg = best_algo.algo;
        this->f_conv_buff->resizeTensor(1, 1, 1, best_algo.memory / sizeof(value_type) + 1);
        this->f_conv_buff_size = this->f_conv_buff->get_mem_size();
    }

#ifdef CONV_DEBUG
    printf("Conv: choose fwd algo: %d, buff size %zu\n", this->f_conv_alg, this->f_conv_buff_size);
#endif
}


template <class value_type>
void conv_layer_t<value_type>::find_best_bwd_algo(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {

#ifdef CONV_DEBUG
    printf("free memory : %zu byte\n", query_free_mem());
#endif

    cudnnConvolutionBwdDataAlgoPerf_t best_bwd_data = search_bwd_data_algo(reg, cudnn_h);
    if (best_bwd_data.time > 0) {
        this->b_conv_alg = best_bwd_data.algo;
        this->b_conv_buff->resizeTensor(1, 1, 1, best_bwd_data.memory / sizeof(value_type) + 1);
        this->b_conv_buff_size = this->b_conv_buff->get_mem_size();
    }

#ifdef CONV_DEBUG
    printf("free memory : %zu byte\n", query_free_mem());
#endif
    cudnnConvolutionBwdFilterAlgoPerf_t best_bwd_filter = search_bwd_filter_algo(reg, cudnn_h);
    if (best_bwd_filter.time > 0) {
        this->filter_alg = best_bwd_filter.algo;
        this->filter_buff->resizeTensor(1, 1, 1, best_bwd_filter.memory / sizeof(value_type) + 1);
        this->filter_buff_size = this->filter_buff->get_mem_size();
    }

#ifdef CONV_DEBUG
    printf("Conv: choose bwd data algo: %d, buff size %zu\n", this->b_conv_alg, this->b_conv_buff_size);
    printf("Conv: choose bwd filter algo: %d, buff size %zu\n", this->filter_alg, this->filter_buff_size);
    printf("Conv: choose bwd plus %zu\n", this->b_conv_buff_size + this->filter_buff_size);
#endif
}


template <class value_type>
void conv_layer_t<value_type>::forward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    printf("======>setup the forward convolution layer:%d\n", this->get_id());

    assert( reg != NULL );
    assert( cudnn_h != NULL );

    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();

    tensor_t<value_type>* t_in = reg->get_reg_output(input_l, curt_l);

    //get the previous layer forward output
    assert( t_in != NULL );

    const int conv_input_channels    = t_in->get_C();
    const int conv_outputs           = this->num_output;
    const int conv_kernel_h          = this->kernel_h;
    const int conv_kernel_w          = this->kernel_w;
    const int conv_dims              = 2;
    const int padding[2]             = { this->padding_h, this->padding_w };
    const int upscales[conv_dims]    = { this->scale_h, this->scale_w   };
    const int strides[conv_dims]     = { this->stride, this->stride };
    //filter description
    const int filter_dim = 4;
    const int filter_dims[filter_dim] = {(int) conv_outputs, (int) conv_input_channels, (int) conv_kernel_h,
                                         (int) conv_kernel_w};
    //create filter weight tensor
    tensor_t<value_type> *weight = new tensor_t<value_type>(conv_outputs, conv_input_channels, conv_kernel_h,
                                                            conv_kernel_w, reg->get_vector(), PARAM, this->get_id());
    this->set_weight(weight, reg);

    weight->init(this->weight_initializer);

    checkCUDNN( cudnnSetFilterNdDescriptor(this->filter_desc,
                                           this->cudnn_data_type,
                                           t_in->get_tensor_format(),
                                           filter_dim,
                                           filter_dims) );

    //v 1.0 only supports image convolution
    checkCUDNN( cudnnSetConvolutionNdDescriptor(this->conv_desc,
                                                this->conv_dims,
                                                padding,
                                                strides,
                                                upscales,
                                                this->cudnn_conv_mode,
                                                this->cudnn_data_type) );

    int t_output_dim[4] = {0, 0, 0, 0};
    checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(this->conv_desc,
                                                      t_in->get_tensor_desc(),
                                                      this->filter_desc,
                                                      filter_dim,
                                                      t_output_dim) );
    //create output tensor
    tensor_t<value_type> *t_out = new tensor_t<value_type>(t_output_dim[0], t_output_dim[1], t_output_dim[2],
                                                           t_output_dim[3], reg->get_vector(), DATA, this->get_id());
    this->set_f_out(t_out, reg);
    //create bias tensor
    tensor_t<value_type> *bias = new tensor_t<value_type>(1, weight->get_N(), 1, 1, reg->get_vector(), PARAM,
                                                          this->get_id());
    // TODO: bias == 0 ?!
    bias->init(this->bias_initializer);
    this->set_bias(bias, reg);

    // we first get the memory-save algorithm to set the f_conv_buff as placeholder
    checkCUDNN( cudnnGetConvolutionForwardAlgorithm(*cudnn_h,
                                                    t_in->get_tensor_desc(),
                                                    this->filter_desc,
                                                    this->conv_desc,
                                                    t_out->get_tensor_desc(),
                                                    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                                                    0,
                                                    &(this->f_conv_alg)
                                                    ) );
    checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(*cudnn_h,
                                                        t_in->get_tensor_desc(),
                                                        this->filter_desc,
                                                        this->conv_desc,
                                                        t_out->get_tensor_desc(),
                                                        this->f_conv_alg,
                                                        &(this->f_conv_buff_size) ) );

    size_t buff_W = (this->f_conv_buff_size) / sizeof(value_type) + 1;
    this->f_conv_buff = new tensor_t<value_type>(1, 1, 1, buff_W, reg->get_vector(), CONV_BUFF, this->get_id() );

    printf("\n------------------conv-layer forward setup %d \n", this->get_id());
    printf("output tensor dims:%d %d %d %d \n", t_output_dim[0], t_output_dim[1] ,t_output_dim[2], t_output_dim[3]);
    printf("Fastest forward conv is Algo %d\n", this->f_conv_alg);
    printf("the size of forward CNN buffer space is %.3f MB\n", (double) this->f_conv_buff_size/1024.0f / 1024.0f);

    //make sure all the necessary tensors are properly set
    assert( this->get_f_out()       != NULL );
    assert( this->get_bias()        != NULL );
    assert( this->get_weight()      != NULL );
    assert( this->f_conv_buff       != NULL );

    //register the forward dependency
    t_in   = reg->get_reg_output(input_l, curt_l);
    weight = this->get_weight();
    t_out  = this->get_f_out();

    reg->register_forward_dependency( this->get_id(), t_in );
    reg->register_forward_dependency( this->get_id(), weight );
    reg->register_forward_dependency( this->get_id(), t_out );
    reg->register_forward_dependency( this->get_id(), this->f_conv_buff );

    if( this->is_bias_enable() ) {
        reg->register_forward_dependency(this->get_id(), this->get_bias() );
    }

}

template <class value_type>
void conv_layer_t<value_type>::backward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    //Find the BACKWARD fastest convolution algorithm
    //create dEdD tensor
    int input_l_id  = this->get_input_layer_id();
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* t_in = reg->get_reg_output(input_l_id, curt_l_id);
    printf("======>setup the backward convolution layer:%d\n", this->get_id());

    tensor_t<value_type> *b_data = new tensor_t<value_type>(t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(),
                                                            reg->get_vector(), DATA, this->get_id());
    this->set_b_data(b_data, reg);

    assert(this->get_f_out()  != NULL);
    assert(this->get_weight() != NULL);
    //already setup in the forward setup
    tensor_t<value_type>* dEdD   = this->get_f_out();
    tensor_t<value_type>* weight = this->get_weight();
    //create weight_grad tensor
    tensor_t<value_type> *weight_grad = new tensor_t<value_type>(weight->get_N(), weight->get_C(), weight->get_H(),
                                                                 weight->get_W(), reg->get_vector(), GRAD,
                                                                 this->get_id());
    tensor_t<value_type> *weight_prev = new tensor_t<value_type>(weight->get_N(), weight->get_C(), weight->get_H(),
                                                                 weight->get_W(), reg->get_vector(), GRAD,
                                                                 this->get_id());
    weight_prev->init(new constant_initializer_t<value_type>(0.0));
    this->set_weight_grad(weight_grad, reg);
    this->set_weight_prev(weight_prev, reg);

    tensor_t<value_type>* f_out  = this->get_f_out();
    assert(f_out != NULL);
    //create bias_grad tensor
    tensor_t<value_type> *bias_grad = new tensor_t<value_type>(1, weight_grad->get_N(), 1, 1, reg->get_vector(), GRAD,
                                                               this->get_id());
    tensor_t<value_type> *bias_prev = new tensor_t<value_type>(1, weight_grad->get_N(), 1, 1, reg->get_vector(), GRAD,
                                                               this->get_id());
    bias_prev->init(new constant_initializer_t<value_type>(0.0));

    this->set_bias_grad(bias_grad, reg);
    this->set_bias_prev(bias_prev, reg);

    checkCUDNN( cudnnGetConvolutionBackwardDataAlgorithm(*cudnn_h,
                                                         this->filter_desc,
                                                         dEdD->get_tensor_desc(),
                                                         this->conv_desc,
                                                         b_data->get_tensor_desc(),
                                                         CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
                                                         0,
                                                         &(this->b_conv_alg) ) );

    checkCUDNN( cudnnGetConvolutionBackwardDataWorkspaceSize(*cudnn_h,
                                                             this->filter_desc,
                                                             dEdD->get_tensor_desc(),
                                                             this->conv_desc,
                                                             b_data->get_tensor_desc(),
                                                             this->b_conv_alg,
                                                             &(this->b_conv_buff_size) ) );

    size_t buff_W = (this->b_conv_buff_size) / sizeof(value_type) + 1;
    this->b_conv_buff = new tensor_t<value_type>(1, 1, 1, buff_W, reg->get_vector(), CONV_BUFF, this->get_id() );


    checkCUDNN( cudnnGetConvolutionBackwardFilterAlgorithm(*cudnn_h,
                                                           t_in->get_tensor_desc(),
                                                           dEdD->get_tensor_desc(),
                                                           this->conv_desc,
                                                           this->filter_desc,
                                                           CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
                                                           0,
                                                           &(this->filter_alg) ) );

    checkCUDNN( cudnnGetConvolutionBackwardFilterWorkspaceSize(*cudnn_h,
                                                               t_in->get_tensor_desc(),
                                                               dEdD->get_tensor_desc(),
                                                               this->conv_desc,
                                                               this->filter_desc,
                                                               this->filter_alg,
                                                               &(this->filter_buff_size)) );

    size_t fbuff_W = (this->filter_buff_size) / sizeof(value_type) + 1;
    this->filter_buff = new tensor_t<value_type>(1, 1, 1, fbuff_W, reg->get_vector(), CONV_BUFF, this->get_id() );

    printf("\n------------------conv-layer backward setup %d \n", this->get_id());
    printf("Fastest backward conv is Algo %d\n", this->b_conv_alg);
    printf("the size of backward CNN buffer space is %.3f MB\n", (double)(this->b_conv_buff_size)/1024.0 / 1024.0);
    printf("Fastest backward CONV filter is Algo %d\n", this->filter_alg);
    printf("the size of backward CNN buffer space is %.3f MB\n", (double)(this->filter_buff_size)/1024.0 / 1024.0);
    //final check
    assert( this->get_b_data()      != NULL );
    assert( this->get_weight_grad() != NULL );
    assert( this->get_bias_grad()   != NULL );

    //register the backward dependency
    weight      = this->get_weight();
    bias_grad   = this->get_bias_grad();
    weight_grad = this->get_weight_grad();
    t_in        = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_c = this->get_b_data();
    tensor_t<value_type>* dEdD_n = reg->get_reg_b_data(output_l_id, curt_l_id);

    assert( t_in != NULL );
    assert( dEdD_c != NULL );
    assert( dEdD_n != NULL );
    assert( weight != NULL );
    assert( bias_grad != NULL );
    assert( weight_grad != NULL );

    reg->register_backward_dependency(this->get_id(), weight );
    reg->register_backward_dependency(this->get_id(), dEdD_n );
    reg->register_backward_dependency(this->get_id(), dEdD_c );
    reg->register_backward_dependency(this->get_id(), t_in );
    reg->register_backward_dependency(this->get_id(), weight_grad );
    reg->register_backward_dependency(this->get_id(), this->b_conv_buff );
    reg->register_backward_dependency(this->get_id(), this->filter_buff );
    if( this->is_bias_enable() ) {
        reg->register_backward_dependency(this->get_id(), bias_grad );
    }
}


template<class value_type>
std::vector<value_type>
conv_layer_t<value_type>::forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
                                  registry_t<value_type> *reg) {

    assert(cudnn_h != NULL);
    assert(reg != NULL);

    //setup each parameters on GPUs
#ifdef DEBUG
    double start = get_cur_time();
#endif

    //prepare the input tensors
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input  = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* weight = this->get_weight();
    tensor_t<value_type>* f_out  = this->get_f_out();

    assert(input  != NULL);
    assert(weight != NULL);
    assert(f_out  != NULL);

    if (is_first_forward) {

        is_first_forward = false;

        this->find_best_fwd_algo(reg, cudnn_h);
    }

    assert(input->get_gpu_ptr() != NULL);
    assert(weight->get_gpu_ptr() != NULL);
    assert(this->f_conv_buff->get_gpu_ptr() != NULL);
    assert(f_out->get_gpu_ptr() != NULL);


    checkCUDNN( cudnnConvolutionForward(*(cudnn_h),
                                        &(this->one),
                                        input->get_tensor_desc(),
                                        input->get_gpu_ptr(),
                                        this->filter_desc,
                                        weight->get_gpu_ptr(),
                                        this->conv_desc,
                                        this->f_conv_alg,
                                        this->f_conv_buff->get_gpu_ptr(),
                                        this->f_conv_buff_size,
                                        &(this->zero),
                                        f_out->get_tensor_desc(),
                                        f_out->get_gpu_ptr()) );

    if(this->is_bias_enable()) {
        checkCUDNN( cudnnAddTensor(*(cudnn_h),
                                   &(this->one),
                                   this->get_bias()->get_tensor_desc(),
                                   this->get_bias()->get_gpu_ptr(),
                                   &(this->one),
                                   f_out->get_tensor_desc(),
                                   f_out->get_gpu_ptr() ) );
    }

#ifdef DEBUG
    double end = get_cur_time();
    input->printTensor("INPUT of Convolution");
    weight->printTensor("WEIGHT of Convolution");
    f_out->printTensor("OUTPUT of Convolution");
    if(this->is_bias_enable()) this->get_bias()->printTensor("Bias of Convolution");
#endif
    return std::vector<value_type>();
}

template<class value_type>
void conv_layer_t<value_type>::backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
                                        registry_t<value_type> *reg) {
    assert(cudnn_h != NULL);
    assert(reg != NULL);

    //prepare the input tensors
    int input_l_id  = this->get_input_layer_id();
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* t_in = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_n = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_c = this->get_b_data();
    tensor_t<value_type>* weight = this->get_weight();
    tensor_t<value_type>* weight_grad = this->get_weight_grad();
    tensor_t<value_type>* bias_grad   = this->get_bias_grad();


    assert( t_in->get_gpu_ptr() != NULL );
    assert( weight->get_gpu_ptr() != NULL );
    assert( dEdD_n->get_gpu_ptr() != NULL );
    assert( dEdD_c->get_gpu_ptr() != NULL );
    assert( bias_grad->get_gpu_ptr() != NULL );
    assert( weight_grad->get_gpu_ptr() != NULL );
    assert( this->b_conv_buff->get_gpu_ptr() != NULL);
    assert(this->filter_buff->get_gpu_ptr() != NULL);


    if (is_first_backward) {
        is_first_backward = false;

        this->find_best_bwd_algo(reg, cudnn_h);
    }

    checkCUDNN( cudnnConvolutionBackwardData(*(cudnn_h),
                                             &(this->one),
                                             this->filter_desc,
                                             weight->get_gpu_ptr(),
                                             dEdD_n->get_tensor_desc(),
                                             dEdD_n->get_gpu_ptr(),
                                             this->conv_desc,
                                             this->b_conv_alg,
                                             this->b_conv_buff->get_gpu_ptr(),
                                             this->b_conv_buff_size,
                                             &(this->zero),
                                             dEdD_c->get_tensor_desc(),
                                             dEdD_c->get_gpu_ptr()) );

    const value_type one_b = 1.0/(value_type) t_in->get_N();
    checkCUDNN( cudnnConvolutionBackwardFilter(*(cudnn_h),
                                               &(this->one),
                                               t_in->get_tensor_desc(),
                                               t_in->get_gpu_ptr(),
                                               dEdD_n->get_tensor_desc(),
                                               dEdD_n->get_gpu_ptr(),
                                               this->conv_desc,
                                               this->filter_alg,
                                               this->filter_buff->get_gpu_ptr(),
                                               this->filter_buff_size,
                                               &(this->zero),
                                               this->filter_desc,
                                               weight_grad->get_gpu_ptr() ) );

    if(this->is_bias_enable()) {
        checkCUDNN( cudnnConvolutionBackwardBias(*(cudnn_h),
                                                 &(this->one),
                                                 dEdD_n->get_tensor_desc(),
                                                 dEdD_n->get_gpu_ptr(),
                                                 &(this->zero),
                                                 bias_grad->get_tensor_desc(),
                                                 bias_grad->get_gpu_ptr() ) );
    }

#ifdef DEBUG
    double end = get_cur_time();
    this->get_bias_grad()->printTensor("Bias Gradient");
    this->get_weight_grad()->printTensor("Weight Gradient");
    dEdD_c->printTensor("@convolution layer backward data");
#endif
}

INSTANTIATE_CLASS(conv_layer_t);

} // superneurons namespace
