#include <layer/join_layer.h>

namespace SuperNeurons {
    
template <class value_type>
void join_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the forward join layer@%d\n", this->get_id() );
    std::vector<std::pair<int, int> > input_keys = this->get_inputs_keys();
    for (size_t i = 0; i < input_keys.size(); i++) {
        std::pair<int, int> input_key = input_keys[i];
        tensor_t<value_type>* input = reg->get_reg_output(input_key.first, input_key.second);
        assert(input != NULL);
        this->set_input( input );
    }
    
    // join layer only has 1 output
    // reduce all the inputs into input[0]

    // should not overlap tensor !!!
    tensor_t<value_type>* t = this->get_inputs()[0];
    tensor_t<value_type>* output = new tensor_t<value_type>(t->get_N(), t->get_C(), t->get_H(), t->get_W(), reg->get_vector(), DATA, this->get_id());
//    tensor_t<value_type>* output = (this->get_inputs())[0];
    std::pair<int, int> output_key = (this->get_outputs_keys())[0];
    this->set_output( output, output_key, reg);
    // register the forward dependency
    std::vector<tensor_t<value_type>* > inputs = this->get_inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        reg->register_forward_dependency( this->get_id(), inputs[i] );
    }
    reg->register_forward_dependency(this->get_id(), output);
}

template <class value_type>
void join_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    //backward
    printf("======>setup the backward join layer@%d\n", this->get_id() );
    std::pair<int, int>    output_k  = (this->get_outputs_keys())[0];
    tensor_t<value_type>* dEdD_next = reg->get_reg_b_data(output_k.second, output_k.first);
    assert(dEdD_next != NULL);
    
    std::vector<std::pair<int, int> > inputs  = this->get_inputs_keys();
    this->set_b_data(dEdD_next, inputs[0], reg);
    for(size_t i = 1; i < inputs.size(); i++) {
        tensor_t<value_type>* tmp = new tensor_t<value_type>( dEdD_next->get_N(), dEdD_next->get_C(), dEdD_next->get_H(), dEdD_next->get_W(), reg->get_vector(), DATA, this->get_id());
        this->set_b_data(tmp, inputs[i], reg);
    }
    // register the backward dependency
    inputs  = this->get_inputs_keys();
    tensor_t<value_type>* source = reg->get_reg_b_data(inputs[0].second, inputs[0].first);
    for(size_t i = 1; i < inputs.size(); i++) {
        tensor_t<value_type>* target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
        reg->register_backward_dependency(this->get_id(), target );
    }
    reg->register_backward_dependency(this->get_id(), source );
}


template <class value_type>
std::vector<value_type> join_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    //we use the input tensor of idx 0 as the output,
    //the subsequents should reduce to idx 0
    std::vector<tensor_t<value_type>* > inputs = this->get_inputs();
#ifdef DEBUG
    printf("@join layer forward:\n");
    printf("input 0 : %p, layer %d\n", inputs[0], inputs[0]->get_layer_id());
    inputs[0]->printTensor("@ join layer inputs");
#endif
    for (size_t i = 1; i < inputs.size(); i++) {
        inputs[0]->sum( inputs[i] );
#ifdef DEBUG
        inputs[i]->printTensor("@ join layer inputs");
        printf("input %zu : %p, layer %d\n", i, inputs[i], inputs[i]->get_layer_id());
#endif
    }
    this->get_outputs()[0]->copy(inputs[0]);
    //const value_type scale_factor = 1.0f/ (value_type) inputs.size();
    //inputs[0]->scale( scale_factor );
#ifdef DEBUG
    tensor_t<value_type>* output = (this->get_outputs())[0];
    output->printTensor("the output of join layer");
#endif
    return std::vector<value_type>();
}

template <class value_type>
void join_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    assert( cudnn_h != NULL );
    assert( reg     != NULL );

#ifdef DEBUG
    printf("@join layer backward:\n");
#endif
    
    std::vector<std::pair<int, int> > inputs  = this->get_inputs_keys();
    tensor_t<value_type>* source = reg->get_reg_b_data(inputs[0].second, inputs[0].first);
#ifdef DEBUG
    printf("source tensor %p, layer %d\n", source, source->get_layer_id());
#endif
    for(size_t i = 1; i < inputs.size(); i++) {
        tensor_t<value_type>* target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
        target->copy(source);
#ifdef DEBUG
        printf(" %zu dest tensor %p, layer %d\n",
               i, reg->get_reg_b_data(inputs[i].second, inputs[i].first), (reg->get_reg_b_data(inputs[i].second, inputs[i].first))->get_layer_id());
#endif
    }
#ifdef DEBUG
    source->printTensor("Source b_data @ join layer");
    for(size_t i = 0; i < inputs.size(); i++) {
        tensor_t<value_type>* target = reg->get_reg_b_data(inputs[i].second, inputs[i].first);
        target->printTensor("destination b_data@ join layer");
    }
#endif
}

INSTANTIATE_CLASS(join_layer_t);
    
} //SuperNeurons namespace
