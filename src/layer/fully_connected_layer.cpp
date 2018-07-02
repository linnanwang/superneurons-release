#include <layer/fully_connected_layer.h>

namespace SuperNeurons {
    
template<class value_type>
void fully_connected_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the backward fully connected layer:%d\n", this->get_id());
    assert(reg != NULL);
    assert(cudnn_h != NULL);
    
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    
    //setup backward data
    tensor_t<value_type>* input   = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* weight  = this->get_weight();
    assert(input  != NULL);
    assert(weight != NULL);
    
    tensor_t<value_type>* b_data  = new tensor_t<value_type>(input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), DATA, this->get_id());
    this->set_b_data(b_data, reg);
    
    //setup backward weight grad
    tensor_t<value_type>* weight_grad = new tensor_t<value_type>(weight->get_N(), weight->get_C(), weight->get_H(), weight->get_W(), reg->get_vector(), GRAD, this->get_id());
    tensor_t<value_type>* weight_prev = new tensor_t<value_type>(weight->get_N(), weight->get_C(), weight->get_H(), weight->get_W(), reg->get_vector(), GRAD, this->get_id());
    weight_prev->init(new constant_initializer_t<value_type>(0));
//    weight_prev->const_fill(0); //zero init
    this->set_weight_grad(weight_grad, reg);
    this->set_weight_prev(weight_prev, reg);
    
    //setup backward bias grad
    tensor_t<value_type>* bias      = this->get_bias();
    tensor_t<value_type>* bias_grad = new tensor_t<value_type>(bias->get_N(), bias->get_C(), bias->get_H(), bias->get_W(), reg->get_vector(), GRAD, this->get_id());
    tensor_t<value_type>* bias_prev = new tensor_t<value_type>(bias->get_N(), bias->get_C(), bias->get_H(), bias->get_W(), reg->get_vector(), GRAD, this->get_id());
    bias_prev->init(new constant_initializer_t<value_type>(0));
//    bias_prev->const_fill(0); //zero init
    this->set_bias_grad(bias_grad, reg);
    this->set_bias_prev(bias_prev, reg);
    
    assert( this->get_weight_grad() != NULL );
    assert( this->get_b_data()      != NULL );
    assert( this->get_bias_grad()   != NULL );
    
    //register the backward dependency
    int curt_l_id   = this->get_id();
    int input_l_id  = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    
    tensor_t<value_type>* t_in            = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_n          = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_c          = this->get_b_data();
    weight          = this->get_weight();
    bias_grad       = this->get_bias_grad();
    weight_grad     = this->get_weight_grad();
    
    assert( t_in        != NULL );
    assert( dEdD_n      != NULL );
    assert( dEdD_c      != NULL );
    assert( weight      != NULL );
    assert( bias_grad   != NULL );
    assert( weight_grad != NULL );
    
    reg->register_backward_dependency(this->get_id(), t_in        );
    reg->register_backward_dependency(this->get_id(), dEdD_n      );
    reg->register_backward_dependency(this->get_id(), dEdD_c      );
    reg->register_backward_dependency(this->get_id(), weight      );
    reg->register_backward_dependency(this->get_id(), bias_grad   );
    reg->register_backward_dependency(this->get_id(), weight_grad );
}

template<class value_type>
void fully_connected_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the forward fully connected layer:%d\n", this->get_id());
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
    
#ifdef DEBUG
    printf("the layer's input tensor:%p, size : %zu %zu %zu %zu\n", input, input->get_N(), input->get_C(), input->get_H(), input->get_W());
#endif
    assert( input != NULL);

    const size_t N  = input->get_N();
    const size_t C  = input->get_C();
    const size_t H  = input->get_H();
    const size_t W  = input->get_W();
    const size_t input_dim = C*H*W;
    const size_t ouput_dim = this->output_dim;
    //right align rule applied to tensor
    //setup weight
    tensor_t<value_type>* weight = new tensor_t<value_type>(ouput_dim, input_dim, 1, 1, reg->get_vector(), PARAM, this->get_id());
    this->set_weight(weight, reg);
    weight->init(this->weight_initializer);
    
    //setup bias
    tensor_t<value_type>* bias    = new tensor_t<value_type>(output_dim, 1, 1, 1, reg->get_vector(), PARAM, this->get_id());
    tensor_t<value_type>* bias_multiplier = new tensor_t<value_type>(1, 1, 1, N, reg->get_vector(), AUX, this->get_id());
    bias->init(this->bias_initializer);
    //to remove
    for(size_t i = 0; i < N; i++)  bias_multiplier->set_scalar(0, 0, 0, i, 1);
    //bias_multiplier->init(new constant_initializer_t<value_type>(1.0f));
    
    this->set_bias(bias, reg);
    this->bias_multiplier = bias_multiplier;
    
    //setup output tensor
    tensor_t<value_type>* output  = new tensor_t<value_type>(N, output_dim, 1, 1, reg->get_vector(), DATA, this->get_id());
    this->set_f_out(output, reg);
    
    assert( this->get_weight() != NULL );
    assert( this->get_bias()   != NULL );
    assert( this->get_f_out()  != NULL );
    assert( this->bias_multiplier != NULL );
    
    //register the forward dependency
    tensor_t<value_type>* t_in   = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* t_out  = this->get_f_out();
    bias                         = this->get_bias();
    weight                       = this->get_weight();
    
    assert( t_in   != NULL );
    assert( weight != NULL );
    assert( t_out  != NULL );
    assert( bias   != NULL );
    
    reg->register_forward_dependency( this->get_id(), t_in );
    reg->register_forward_dependency( this->get_id(), weight );
    reg->register_forward_dependency( this->get_id(), t_out );
    reg->register_forward_dependency( this->get_id(), bias );
}

template<>
void fully_connected_layer_t<float>::mat_multiply(cublasHandle_t* cublas_h,
                                                 int m, int n, int k,
                                                 cublasOperation_t TransA, cublasOperation_t TransB,
                                                 float alpha, float beta,
                                                 float* A, int lda,
                                                 float* B, int ldb,
                                                 float* C, int ldc) {
    checkCublasErrors(
                      cublasSgemm(*(cublas_h),
                                  TransA, TransB,
                                  m, n, k,
                                  &alpha,
                                  A, lda,
                                  B, ldb,
                                  &beta,
                                  C, ldc)
                      );
}
    
template<>
void fully_connected_layer_t<double>::mat_multiply(cublasHandle_t* cublas_h,
                                                 int m, int n, int k,
                                                 cublasOperation_t TransA, cublasOperation_t TransB,
                                                 double alpha, double beta,
                                                 double* A, int lda,
                                                 double* B, int ldb,
                                                 double* C, int ldc)
{
    checkCublasErrors(
                      cublasDgemm(*(cublas_h),
                                  TransA, TransB,
                                  m, n, k,
                                  &alpha,
                                  A, lda,
                                  B, ldb,
                                  &beta,
                                  C, ldc)
                      );
}
    
template<class value_type>
std::vector<value_type> fully_connected_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) {
    //--------forward data operation--------//
    const int m_d = (int) this->get_f_out()->get_N();  //output dim
    const int k_d = (int) this->get_weight()->get_C(); //@line14, input dim
    const int n_d = (int) this->get_weight()->get_N();  //@line14, total images
    
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in   = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* weight = this->get_weight();
    tensor_t<value_type>* t_out  = this->get_f_out();
    tensor_t<value_type>* bias   = this->get_bias();
    
#ifdef DEBUG
    printf("input tensor from %d to %d\n", input_l, curt_l);
#endif
    //forward data
    mat_multiply(cublas_h,
                 n_d, m_d, k_d,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 this->one, this->zero,
                 weight->get_gpu_ptr(), k_d,
                 t_in->get_gpu_ptr(),   k_d,
                 t_out->get_gpu_ptr(),  n_d );
    
    //--------forward bias operation--------//
    if(this->is_bias_enable()) {
        mat_multiply(cublas_h,
                     n_d, m_d, this->one,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     this->one, this->one,
                     bias->get_gpu_ptr(),
                     (int) n_d,
                     this->bias_multiplier->get_gpu_ptr(),
                     (int) this->one,
                     t_out->get_gpu_ptr(),
                     (int) n_d );
    }

#ifdef DEBUG
    this->get_weight()->printTensor("fully connected, weight");
    this->bias_multiplier->printTensor("fully connected, bias multiplier");
    t_in->printTensor("fully connected, input");
    this->get_bias()->printTensor("fully connected, bias");
    this->get_f_out()->printTensor("fully connected, after bias output");
#endif
    return std::vector<value_type>();
}
    


template<class value_type>
void fully_connected_layer_t<value_type>::backward_data(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) {
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();

    tensor_t<value_type>* dEdD_n   = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_c   = this->get_b_data();
    tensor_t<value_type>* weight      = this->get_weight();
    // w[inputxoutput]*sigma(l+1)'[outputxN] = sigma[inputxN] column format
    // takes another transpose to the row format
    
    const int m_d = (int) this->get_f_out()->get_N();  //output dim
    const int k_d = (int) this->get_weight()->get_C(); //@line14, input dim
    const int n_d = (int) this->get_weight()->get_N();  //@line14, total images
    int ld_a = (int) weight->get_W();
    int ld_b = (int) dEdD_n->get_W();
    int ld_c = (int) (dEdD_c->get_C() * dEdD_c->get_H() * dEdD_c->get_W());
    value_type one   = 1;
    value_type zero  = 0;
    
    mat_multiply(cublas_h, k_d, m_d, n_d,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 this->one, this->zero,
                 weight->get_gpu_ptr(), k_d,
                 dEdD_n->get_gpu_ptr(), n_d,
                 dEdD_c->get_gpu_ptr(), k_d);

#ifdef DEBUG
    this->get_b_data()->printTensor("fully connected, backward data");
#endif
}

template<class value_type>
void fully_connected_layer_t<value_type>::backward_weight(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) {
    int input_l_id  = this->get_input_layer_id();
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();


    tensor_t<value_type>* t_in        = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* weight_grad = this->get_weight_grad();
    tensor_t<value_type>* dEdD_n      = reg->get_reg_b_data(output_l_id, curt_l_id);
    //input[flatten_dim, N] * dEdD[N, y] = weight_grad[flatten_dim,y]
    
    const int m_d = (int) this->get_f_out()->get_N();  //output dim
    const int k_d = (int) this->get_weight()->get_C(); //@line14, input dim
    const int n_d = (int) this->get_weight()->get_N();  //@line14, total images
    
    mat_multiply(cublas_h, k_d, n_d, m_d,
                 CUBLAS_OP_N, CUBLAS_OP_T,
                 this->one, this->one,
                 t_in->get_gpu_ptr(),   //bottom data
                 (int) k_d,
                 dEdD_n->get_gpu_ptr(), //top diff
                 (int) n_d,
                 weight_grad->get_gpu_ptr(),
                 (int) k_d );
    
#ifdef DEBUG
    printf("m:%d n:%d, k:%d: ld_dEdD:%zu ld_weight_grad:%zu\n",m_d, n_d, k_d, dEdD_n->get_W(), weight_grad->get_W());
    this->get_weight_grad()->printTensor("fully connected, backward weight grad");
#endif
}
    
template<class value_type>
void fully_connected_layer_t<value_type>::backward_bias(cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg)
{
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    
    tensor_t<value_type>* dEdD            = reg->get_reg_b_data(output_l_id, curt_l_id);
    tensor_t<value_type>* bias_multiplier = this->bias_multiplier;
    //input[1, N] * dEdD[N, output] = bias_diff[1, output]
    tensor_t<value_type>* bias_grad       = this->get_bias_grad();
    
    const int m_d = (int) this->get_f_out()->get_N();  //output dim
    const int n_d = (int) this->get_weight()->get_N();  //@line14, total images
    
    cublas_gemv(cublas_h, CUBLAS_OP_N,
                n_d, m_d,
                &(this->one),
                dEdD->get_gpu_ptr(), n_d,
                bias_multiplier->get_gpu_ptr(), 1,
                &(this->one),
                bias_grad->get_gpu_ptr(), 1 );
#ifdef DEBUG
    this->get_bias_grad()->printTensor("fully connected, backward bias grad");
#endif
}
    
template<class value_type>
void fully_connected_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) {
    
    backward_weight(cublas_h, cudnn_h, reg);
    backward_data(cublas_h, cudnn_h, reg);
    if(this->is_bias_enable() ) {
        backward_bias(cublas_h, cudnn_h, reg);
    }
}

INSTANTIATE_CLASS(fully_connected_layer_t);

}// SuperNeurons namespace
