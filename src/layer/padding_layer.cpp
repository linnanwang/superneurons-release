//
// Created by ay27 on 7/4/17.
//

#include <layer/padding_layer.h>

namespace SuperNeurons {


// gpu function //

template<class value_type>
void padding_forward(size_t N, size_t C, size_t H, size_t W, size_t padC, size_t padH, size_t padW,
                     const value_type *src, value_type *dst);

template<class value_type>
void padding_backward(size_t N, size_t C, size_t H, size_t W, size_t padC, size_t padH, size_t padW,
                      const value_type *src, value_type *dst);


template<class value_type>
void padding_layer_t<value_type>::forward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {
    //hook the output of previous layer
    int input_l = this->get_input_layer_id();
    int curt_l = this->get_id();
    tensor_t<value_type> *input = reg->get_reg_output(input_l, curt_l);
    assert(input != NULL);
    printf("======>setup the forward padding layer:%d\n", this->get_id());

    size_t output_tensor_dim[4] = {input->get_N(), input->get_C() + 2 * padC, input->get_H() + 2 * padH,
                                   input->get_W() + 2 * padW};

    tensor_t<value_type> *f_out = new tensor_t<value_type>(output_tensor_dim[0], output_tensor_dim[1],
                                                           output_tensor_dim[2], output_tensor_dim[3],
                                                           reg->get_vector(), DATA, this->get_id());

    //setup the output tensor
    this->set_f_out(f_out, reg);

    //forward hookup check
    assert(this->get_f_out() != NULL);

    // register the forward dependency
    input = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type> *output = this->get_f_out();
    reg->register_forward_dependency(this->get_id(), input);
    reg->register_forward_dependency(this->get_id(), output);

}

template<class value_type>
void padding_layer_t<value_type>::backward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {
    //setup the backward data
    int curt_l = this->get_id();
    int input_l = this->get_input_layer_id();
    int output_l = this->get_output_layer_id();
    tensor_t<value_type> *f_in = reg->get_reg_output(input_l, curt_l);
    assert(f_in != NULL);
    printf("======>setup the backward pooling layer:%d\n", this->get_id());

    tensor_t<value_type> *b_data = new tensor_t<value_type>(f_in->get_N(), f_in->get_C(), f_in->get_H(),
                                                            f_in->get_W(), reg->get_vector(), DATA, this->get_id());
    this->set_b_data(b_data, reg);
    assert(this->get_b_data() != NULL);

    // register the backward dependency
    tensor_t<value_type> *dEdD_curt = this->get_b_data();
    tensor_t<value_type> *dEdD_next = reg->get_reg_b_data(output_l, curt_l);
    reg->register_backward_dependency(this->get_id(), dEdD_curt);
    reg->register_backward_dependency(this->get_id(), dEdD_next);

}

template<class value_type>
std::vector<value_type>
padding_layer_t<value_type>::forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
                                     registry_t<value_type> *reg) {
    int input_l = this->get_input_layer_id();
    int curt_l = this->get_id();
    tensor_t<value_type> *input = reg->get_reg_output(input_l, curt_l);
    tensor_t<value_type> *output = this->get_f_out();

    size_t N = input->get_N(), C = input->get_C(), H = input->get_H(), W = input->get_W();

    padding_forward(N, C, H, W, padC, padH, padW, input->get_gpu_ptr(), output->get_gpu_ptr());

#ifdef DEBUG
    input->printTensor("input @ padding layer");
    output->printTensor("output @ padding layer");
#endif

    return std::vector<value_type>();
}

template<class value_type>
void padding_layer_t<value_type>::backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
                                           registry_t<value_type> *reg) {
    int curt_l_id = this->get_id();
    int output_l_id = this->get_output_layer_id();
    int input_l_id = this->get_input_layer_id();

    tensor_t<value_type> *dEdD_curt = this->get_b_data();
    tensor_t<value_type> *dEdD_next = reg->get_reg_b_data(output_l_id, curt_l_id);

    size_t N = dEdD_curt->get_N(), C = dEdD_curt->get_C(), H = dEdD_curt->get_H(), W = dEdD_curt->get_W();

    padding_backward(N, C, H, W, padC, padH, padW, dEdD_next->get_gpu_ptr(), dEdD_curt->get_gpu_ptr());

#ifdef DEBUG
    printf("@%d prev %d next %d\n", curt_l_id, input_l_id, output_l_id);
    dEdD_next->printTensor("Backward padding dEdD_next");
    dEdD_curt->printTensor("Backward padding Results");
#endif
}

INSTANTIATE_CLASS(padding_layer_t);
}
