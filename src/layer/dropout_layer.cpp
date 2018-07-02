#include <layer/dropout_layer.h>

namespace SuperNeurons {

template <class value_type>
void dropout_layer_t<value_type>::forward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    //hook the output of previous layer
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);
    assert( input != NULL );
    printf("======>setup the forward drop out layer:%d\n", this->get_id());

    checkCUDNN( cudnnDropoutGetStatesSize(*cudnn_h, &state_size_bytes) );
    size_t state_size = state_size_bytes / sizeof(value_type);
    dropout_state = new tensor_t<value_type>( state_size, 1, 1, 1, reg->get_vector(), PARAM, this->get_id());
    unsigned long seed = (unsigned long) rand();
    checkCUDNN( cudnnSetDropoutDescriptor(dropout_desc,
                                          *cudnn_h,
                                          dropout_rate,
                                          dropout_state->get_gpu_ptr(),
                                          state_size_bytes,
                                          seed) );
    
    checkCUDNN( cudnnDropoutGetReserveSpaceSize(input->get_tensor_desc(),
                                                &buff_size_bytes ) );
    size_t buff_size = buff_size_bytes / sizeof(value_type);
    dropout_buff = new tensor_t<value_type>(buff_size, 1, 1, 1, reg->get_vector(), PARAM, this->get_id());
    
    tensor_t<value_type>* f_out  = new tensor_t<value_type>( input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), DATA, this->get_id());
    //setup the output tensor
    this->set_f_out( f_out, reg );
    
    //forward hookup check
    assert( this->get_f_out() != NULL );
    assert( dropout_state != NULL );
    assert( dropout_buff  != NULL );

    // register the forward dependency
    reg->register_forward_dependency(this->get_id(), input);
    reg->register_forward_dependency(this->get_id(), dropout_state);
    reg->register_forward_dependency(this->get_id(), dropout_buff);
    reg->register_forward_dependency(this->get_id(), f_out);
}
    
template <class value_type>
void dropout_layer_t<value_type>::backward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    //setup the backward data
    printf("======>setup the backward dropout layer:%d\n", this->get_id());
    int curt_l_id   = this->get_id();
    int prev_l_id   = this->get_input_layer_id();
    int output_l_id = this->get_output_layer_id();
    tensor_t<value_type>* f_in = reg->get_reg_output(prev_l_id, curt_l_id);
    assert( f_in != NULL );
    tensor_t<value_type>* b_data = new tensor_t<value_type>(f_in->get_N(), f_in->get_C(), f_in->get_H(), f_in->get_W(), reg->get_vector(), DATA, this->get_id());

    this->set_b_data( b_data, reg );
    assert( this->get_b_data() != NULL );

    // register the backward dependency
    tensor_t<value_type>* dEdD   = reg->get_reg_b_data(output_l_id, curt_l_id);
    reg->register_backward_dependency(this->get_id(), b_data);
    reg->register_backward_dependency(this->get_id(), dropout_buff);
    reg->register_backward_dependency(this->get_id(), dEdD);
}

template <class value_type>
std::vector<value_type> dropout_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input  = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* output = this->get_f_out();
#ifdef DEBUG
    printf("@ dropout layer input tensor from %d to %d\n", input_l, curt_l);
#endif

    assert(dropout_state->get_gpu_ptr() != NULL);

    if (stage == NET_INFER) {
        output->copy(input);
    } else {
        
        checkCUDNN( cudnnDropoutForward(*cudnn_h,
                                        dropout_desc,
                                        input->get_tensor_desc(),
                                        input->get_gpu_ptr(),
                                        output->get_tensor_desc(),
                                        output->get_gpu_ptr(),
                                        dropout_buff->get_gpu_ptr(),
                                        buff_size_bytes)
                   );
    }
    

#ifdef DEBUG
    this->get_f_out()->printTensor("output of dropout layer");
#endif
    return std::vector<value_type>();
}

template <class value_type>
void dropout_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    int input_l_id  = this->get_input_layer_id();
    
//    tensor_t<value_type>* f_out  = this->get_f_out();
    tensor_t<value_type>* b_data = this->get_b_data();
//    tensor_t<value_type>* f_in   = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* dEdD   = reg->get_reg_b_data(output_l_id, curt_l_id);
    
    checkCUDNN( cudnnDropoutBackward(*cudnn_h,
                                     dropout_desc,
                                     dEdD->get_tensor_desc(),
                                     dEdD->get_gpu_ptr(),
                                     b_data->get_tensor_desc(),
                                     b_data->get_gpu_ptr(),
                                     dropout_buff->get_gpu_ptr(),
                                     buff_size_bytes)
               );

#ifdef DEBUG
    printf( "@%d prev %d next %d\n", curt_l_id, input_l_id, output_l_id );
    this->get_b_data()->printTensor("backward dropout results");
#endif
}
    
INSTANTIATE_CLASS(dropout_layer_t);

} //SuperNeurons namespace
