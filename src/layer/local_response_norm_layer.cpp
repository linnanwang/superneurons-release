#include <layer/local_response_norm_layer.h>

namespace SuperNeurons {
    
template <class value_type>
void LRN_layer_t<value_type>::forward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    //hook the output of previous layer
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in = reg->get_reg_output(input_l, curt_l);
    assert( t_in != NULL );
    printf("======>setup the forward local response normalization layer:%d\n", this->get_id());

    checkCUDNN( cudnnSetLRNDescriptor(this->norm_desc,
                                      this->lrnN,
                                      this->lrnAlpha,
                                      this->lrnBeta,
                                      this->lrnK) );
    
    tensor_t<value_type>* t_out  = new tensor_t<value_type>( t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(), reg->get_vector(), DATA, this->get_id());
    
    //setup the output tensor
    this->set_f_out( t_out, reg );
    
    //forward hookup check
    assert( this->get_f_out() != NULL );
    
    reg->register_forward_dependency( this->get_id(), t_in );
    reg->register_forward_dependency( this->get_id(), t_out );

}
    
template <class value_type>
void LRN_layer_t<value_type>::backward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    //setup the backward data
    printf("======>setup the backward local response normalization layer:%d\n", this->get_id());
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    int input_l_id  = this->get_input_layer_id();
    
    tensor_t<value_type>* dEdD   = reg->get_reg_b_data(output_l_id, curt_l_id);
    assert(dEdD != NULL);
    this->set_b_data( dEdD, reg );
    assert( this->get_b_data() != NULL );
    
    
    tensor_t<value_type>* f_out  = this->get_f_out();
    tensor_t<value_type>* b_data = this->get_b_data();
    tensor_t<value_type>* f_in   = reg->get_reg_output(input_l_id, curt_l_id);
    dEdD   = reg->get_reg_b_data(output_l_id, curt_l_id);
    
    reg->register_backward_dependency( this->get_id(), f_out );
    reg->register_backward_dependency( this->get_id(), b_data );
    reg->register_backward_dependency( this->get_id(), f_in );
    reg->register_backward_dependency( this->get_id(), dEdD );
    

}

template <class value_type>
std::vector<value_type> LRN_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input  = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* output = this->get_f_out();
#ifdef DEBUG
    printf("input tensor from %d to %d\n", input_l, curt_l);
#endif

    assert(input->get_gpu_ptr() != NULL);
    assert(output->get_gpu_ptr() != NULL);

    checkCUDNN( cudnnLRNCrossChannelForward(*cudnn_h,
                                            this->norm_desc,
                                            this->LRN_mode,
                                            &(this->one),
                                            input->get_tensor_desc(),
                                            input->get_gpu_ptr(),
                                            &(this->zero),
                                            output->get_tensor_desc(),
                                            output->get_gpu_ptr() )
               );
#ifdef DEBUG
    this->get_f_out()->printTensor("output of local response normalization");
#endif
    return std::vector<value_type>();
}

template <class value_type>
void LRN_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    int input_l_id  = this->get_input_layer_id();
    
    tensor_t<value_type>* f_out  = this->get_f_out();
    tensor_t<value_type>* b_data = this->get_b_data();
    tensor_t<value_type>* f_in   = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* dEdD   = reg->get_reg_b_data(output_l_id, curt_l_id);
    
    checkCUDNN( cudnnLRNCrossChannelBackward(*cudnn_h,
                                             this->norm_desc,
                                             this->LRN_mode,
                                             &(this->one),
                                             f_out->get_tensor_desc(),
                                             f_out->get_gpu_ptr(),
                                             dEdD->get_tensor_desc(),
                                             dEdD->get_gpu_ptr(),
                                             f_in->get_tensor_desc(),
                                             f_in->get_gpu_ptr(),
                                             &(this->zero),
                                             b_data->get_tensor_desc(),
                                             b_data->get_gpu_ptr() )
               );

#ifdef DEBUG
    printf( "@%d prev %d next %d\n", curt_l_id, input_l_id, output_l_id );
    this->get_b_data()->printTensor("backward local response normalization results");
#endif
}
    
INSTANTIATE_CLASS(LRN_layer_t);

} //SuperNeurons namespace


