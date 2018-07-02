#include <util/common.h>
#include <layer/cudnn_pooling_layer.h>

namespace SuperNeurons {
    
template <class value_type>
void pool_layer_t<value_type>::forward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    //hook the output of previous layer
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in = reg->get_reg_output(input_l, curt_l);
    printf("======>setup the forward pooling layer:%d\n", this->get_id());
    
    checkCUDNN( cudnnSetPooling2dDescriptor(this->pool_desc,
                                            this->mode,
                                            this->p_nan,
                                            this->kernel_height,
                                            this->kernel_width,
                                            this->vertical_padding,
                                            this->horizontal_padding,
                                            this->vertical_stride,
                                            this->horizontal_stride) );
    int output_tensor_dim[4] = { 0, 0, 0, 0 };
    
    checkCUDNN( cudnnGetPooling2dForwardOutputDim(this->pool_desc,
                                                  t_in->get_tensor_desc(),
                                                  &output_tensor_dim[0],
                                                  &output_tensor_dim[1],
                                                  &output_tensor_dim[2],
                                                  &output_tensor_dim[3]) );
    
    tensor_t<value_type>* f_out  = new tensor_t<value_type>(output_tensor_dim[0], output_tensor_dim[1], output_tensor_dim[2], output_tensor_dim[3], reg->get_vector(), DATA, this->get_id());
    this->set_f_out( f_out, reg );
    
    //make sure all the necessary tensors are properly set
    assert( this->get_f_out() != NULL );
    assert( t_in != NULL );
    
    //register the forward dependency
    t_in = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* t_out = this->get_f_out();
    
    reg->register_forward_dependency( this->get_id(), t_in );
    reg->register_forward_dependency( this->get_id(), t_out );
    
}
    
template <class value_type>
void pool_layer_t<value_type>::backward_setup(registry_t<value_type>* reg, cudnnHandle_t* cudnn_h) {
    //setup the backward data
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* f_in   = reg->get_reg_output(input_l, curt_l);
    assert( f_in != NULL );
    printf("======>setup the backward pooling layer:%d\n", this->get_id());
    
    tensor_t<value_type>* b_data = new tensor_t<value_type>(f_in->get_N(), f_in->get_C(), f_in->get_H(), f_in->get_W(), reg->get_vector(), DATA, this->get_id());
    this->set_b_data( b_data, reg );
    assert( this->get_b_data() != NULL );
    
    //register the backward dependency
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    int input_l_id  = this->get_input_layer_id();
    
    tensor_t<value_type>* t_out    = this->get_f_out();
    tensor_t<value_type>* dEdD_c   = this->get_b_data();
    tensor_t<value_type>* t_in     = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_n   = reg->get_reg_b_data(output_l_id, curt_l_id);
    
    assert( t_out  != NULL );
    assert( t_in   != NULL );
    assert( dEdD_n != NULL );
    assert( dEdD_c != NULL );

    reg->register_backward_dependency(this->get_id(), t_out  );
    reg->register_backward_dependency(this->get_id(), t_in   );
    reg->register_backward_dependency(this->get_id(), dEdD_n );
    reg->register_backward_dependency(this->get_id(), dEdD_c );

}

template <class value_type>
std::vector<value_type> pool_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in  = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type>* t_out = this->get_f_out();
#ifdef DEBUG
    printf("input tensor from %d to %d\n", input_l, curt_l);
#endif

    checkCUDNN( cudnnPoolingForward(*(cudnn_h),
                                    this->pool_desc,
                                    &(this->one),
                                    t_in->get_tensor_desc(),
                                    t_in->get_gpu_ptr(),
                                    &(this->zero),
                                    t_out->get_tensor_desc(),
                                    t_out->get_gpu_ptr()) );
#ifdef DEBUG
    this->get_f_out()->printTensor("OUTPUT of Pooling");
#endif
    return std::vector<value_type>();
}

template <class value_type>
void pool_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    int curt_l_id   = this->get_id();
    int output_l_id = this->get_output_layer_id();
    int input_l_id  = this->get_input_layer_id();
    
    tensor_t<value_type>* t_out    = this->get_f_out();
    tensor_t<value_type>* dEdD_c   = this->get_b_data();
    tensor_t<value_type>* t_in     = reg->get_reg_output(input_l_id, curt_l_id);
    tensor_t<value_type>* dEdD_n   = reg->get_reg_b_data(output_l_id, curt_l_id);
    
    checkCUDNN( cudnnPoolingBackward(*(cudnn_h),
                                     this->pool_desc,
                                     &(this->one),
                                     t_out->get_tensor_desc(),
                                     t_out->get_gpu_ptr(),
                                     dEdD_n->get_tensor_desc(),
                                     dEdD_n->get_gpu_ptr(),
                                     t_in->get_tensor_desc(),
                                     t_in->get_gpu_ptr(),
                                     &(this->zero),
                                     dEdD_c->get_tensor_desc(),
                                     dEdD_c->get_gpu_ptr())
               );
#ifdef DEBUG
    printf( "@%d prev %d next %d\n", curt_l_id, input_l_id, output_l_id );
    this->get_b_data()->printTensor("Backward Pooling Results");
#endif
}
    
INSTANTIATE_CLASS(pool_layer_t);

} //SuperNeurons namespace


