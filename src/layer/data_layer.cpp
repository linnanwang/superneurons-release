//
// Created by ay27 on 17/3/31.
//

#include <layer/data_layer.h>

namespace SuperNeurons {
    
template<class value_type>
void data_layer_t<value_type>::forward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {
    // create tensor to store data and label
    printf("======>setup the forward data layer:%d\n", this->get_id());
    tensor_t<value_type> *f_out = new tensor_t<value_type>(this->N, this->C, this->H, this->W, reg->get_vector(), DATA_SOURCE, this->get_id());
    tensor_t<value_type> *label = new tensor_t<value_type>(this->N, 1, 1, 1, reg->get_vector(), DATA_SOURCE, this->get_id());

    int cur_layer_id = this->get_id();
    int dst_layer_id = this->get_output_layer_id();
    
    if(this->mode == DATA_TRAIN) {
        reg->set_train_label(label);
    } else if(this->mode == DATA_TEST) {
        reg->set_test_label(label);
    }
    this->set_f_out(f_out, reg);
    
    //register the forward dependency
    tensor_t<value_type>* output = this->get_f_out();
    reg->register_forward_dependency( this->get_id(), output );
}

template<class value_type>
std::vector<value_type> data_layer_t<value_type>::forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg) {
    tensor_t<value_type>* label = NULL;
    if(this->mode == DATA_TRAIN) {
        label = reg->get_train_label();
    } else {
        label = reg->get_test_label();
    }
    
    tensor_t<value_type>* output = this->get_f_out();

    reader->get_batch(output, label);

#ifdef DEBUG
    output->printTensor("output from data layer");
#endif
#ifdef PRINT_DATA
    if(this->mode == DATA_TEST) {
        printf("------------------testing@layer%p output%p label%p---------------------\n", this, output, label);
        output->printTensorNoDebug("test input image");
        label->printTensorNoDebug("test labels");
    } else {
        printf("------------------training@layer%p output%p label%p---------------------\n", this, output, label);
        output->writeToFile("training_tensor");
        label->printTensorNoDebug("train labels");
    }
#endif

    return std::vector<value_type>();
}

INSTANTIATE_CLASS(data_layer_t);
}
