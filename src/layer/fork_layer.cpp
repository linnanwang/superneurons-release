#include <layer/fork_layer.h>

namespace SuperNeurons {
    
template <class value_type>
void fork_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the forward fork layer@%d\n", this->get_id() );
    std::pair<int, int> input_key = (this->get_inputs_keys())[0];
    tensor_t<value_type>* input = reg->get_reg_output(input_key.first, input_key.second);
    this->set_input( input );
    // it has to be hard copy to avoid multiple writes
    std::vector<std::pair<int, int> > output_keys = this->get_outputs_keys();

    // should not overlap tensor !!!!
//    this->set_output( input, output_keys[0], reg);
    for(size_t i = 0; i < output_keys.size(); i++) {
        tensor_t<value_type>* tmp = new tensor_t<value_type>(input->get_N(), input->get_C(), input->get_H(), input->get_W(), reg->get_vector(), DATA, this->get_id());
        this->set_output(tmp, output_keys[i], reg);
    }
    
    // register the forward dependency
    // please be noted the input is outputs[0]
    std::vector<tensor_t<value_type>* > outputs = this->get_outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        reg->register_forward_dependency( this->get_id(), outputs[i] );
    }
    reg->register_forward_dependency( this->get_id(), input);
}

template <class value_type>
void fork_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    //backward
    printf("======>setup the backward fork layer@%d\n", this->get_id() );
    //we use the first output_keys as the output b_data
    std::pair<int, int> b_data_key = (this->get_inputs_keys())[0];
    std::vector<std::pair<int, int> > dEdD_next_keys = this->get_outputs_keys();

    tensor_t<value_type>* b_data  = reg->get_reg_b_data( dEdD_next_keys[0].second, dEdD_next_keys[0].first );
    assert(b_data != NULL);
    this->set_b_data(b_data, b_data_key, reg);
    
    for(size_t i = 1; i < dEdD_next_keys.size(); i++) {
        tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
        assert(tmp != NULL);
        assert( tmp->get_N() == b_data->get_N() );
        assert( tmp->get_C() == b_data->get_C() );
        assert( tmp->get_H() == b_data->get_H() );
        assert( tmp->get_W() == b_data->get_W() );
    }
    //register the backward dependency
    dEdD_next_keys = this->get_outputs_keys();
    b_data = (this->get_b_data())[0];
    for(size_t i = 1; i < dEdD_next_keys.size(); i++) {
        tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
        reg->register_backward_dependency(this->get_id(), tmp );
    }
    reg->register_backward_dependency(this->get_id(), b_data  );
}


template <class value_type>
std::vector<value_type> fork_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    //the input layer set the output tensor in the idx 0,
    //the subsequents should copy from idx0
    tensor_t<value_type>* input = this->get_inputs()[0];
    std::vector<tensor_t<value_type>* > outputs = this->get_outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i]->copy( input );
    }

#ifdef DEBUG
    printf("f fork layer : %d\n", this->get_id());
    input->printTensor("input @ fork layer");
    for (size_t i=0; i<outputs.size(); i++) {
        printf("output %zu tensor %p, layer %d\n", i, outputs[i], outputs[i]->get_layer_id());
        outputs[i]->printTensor("output @ fork layer");
    }
#endif

    return std::vector<value_type>();
}

template <class value_type>
void fork_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    std::vector<std::pair<int, int> > dEdD_next_keys = this->get_outputs_keys();
    tensor_t<value_type>* b_data = (this->get_b_data())[0];

#ifdef DEBUG
    printf("b fork layer : %d\n", this->get_id());
    b_data->printTensor("backward data src @ fork layer");
#endif
    
    for(size_t i = 1; i < dEdD_next_keys.size(); i++) {
        tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
        b_data->sum( tmp );
    }
    //const value_type scale_factor = 1.0f/ (value_type)dEdD_next_keys.size();
    //b_data->scale( scale_factor );
#ifdef DEBUG
    printf("backward src 0, tensor %p layer %d\n", b_data, b_data->get_layer_id());
    for(size_t i = 1; i < dEdD_next_keys.size(); i++) {
        tensor_t<value_type>* tmp = reg->get_reg_b_data(dEdD_next_keys[i].second, dEdD_next_keys[i].first);
        tmp->printTensor("backward data src @ fork layer");
        printf("backward src %zu, tensor %p layer %d\n", i, tmp, tmp->get_layer_id());
    }

    b_data->printTensor("backward data dst @ fork layer");
#endif
    
}

INSTANTIATE_CLASS(fork_layer_t);
    
} //SuperNeurons namespace
