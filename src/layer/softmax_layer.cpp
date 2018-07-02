#include <layer/softmax_layer.h>

namespace SuperNeurons {
//------GPU functions-----//
template <class value_type>
float softmax_loss(value_type* pred_gpu_ptr, value_type* label_gpu_ptr, int N, int C, int H, int W);
    
template <class value_type>
void softmax_grad(value_type* pred_gpu_ptr, value_type* label_gpu_ptr, int N, int C, int H, int W);
    
template <class value_type>
value_type softmax_top1_accuracy(value_type* label, value_type* predict, int N, int C, int H, int W);
template <class value_type>
value_type softmax_top5_accuracy(value_type* label, value_type* predict, int N, int C, int H, int W);
//------------------------//

template <class value_type>
void softmax_layer_t<value_type>::forward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    
    printf("======>setup the forward softmax layer:%d\n", this->get_id());
    //in-place operations, no need to create tensors
    //forward
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* t_in = reg->get_reg_output(input_l, curt_l);

    tensor_t<value_type>* t_out = new tensor_t<value_type>(t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(), reg->get_vector(), DATA, this->get_id());

    assert(t_in != NULL);
    this->set_f_out( t_out, reg ); //use the inplace operation.
    assert( this->get_f_out()  != NULL );
    
    //register the forward dependency
    t_in                               = reg->get_reg_output(input_l, curt_l);
//    tensor_t<value_type>* t_out        = this->get_f_out();
    tensor_t<value_type>* label_train  = reg->get_train_label();
    tensor_t<value_type>* label_test   = reg->get_test_label();
    
    assert( t_in        != NULL );
    assert( t_out       != NULL );
    assert( label_train != NULL );
    
    reg->register_forward_dependency( this->get_id(), t_in        );
    reg->register_forward_dependency( this->get_id(), t_out       );
    reg->register_forward_dependency( this->get_id(), label_train );
}
    
template <class value_type>
void softmax_layer_t<value_type>::backward_setup( registry_t<value_type>* reg, cudnnHandle_t* cudnn_h ) {
    printf("======>setup the backward softmax layer:%d\n", this->get_id());
    //in-place operations, no need to create tensors
    //backward hookup
    this->set_b_data( this->get_f_out(), reg );
    //hookup checks
    assert( this->get_b_data() != NULL );
    
    tensor_t<value_type>* t_out        = this->get_f_out();
    tensor_t<value_type>* label_train  = reg->get_train_label();
    tensor_t<value_type>* t_in         = reg->get_reg_output(this->get_input_layer_id(), this->get_id());
    
    assert( t_out != NULL );
    assert( label_train != NULL );
    
    //register the backward dependency
    reg->register_backward_dependency(this->get_id(), t_out        );
    reg->register_backward_dependency(this->get_id(), label_train  );
    reg->register_backward_dependency(this->get_id(), t_in);    // we should add the backward dependency to avoid freeing in recompute part

}

bool has_elem(int* array, int size, size_t target) {
    for(int i = 0; i < size; i++) {
        if( (size_t)array[i] == target ) {
            return true;
        }
    }
    return false;
}
    
template <class value_type>
std::vector<value_type> softmax_layer_t<value_type>::forward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    
    assert( cudnn_h != NULL );
    assert( reg     != NULL );
    
    int input_l = this->get_input_layer_id();
    int curt_l  = this->get_id();
    tensor_t<value_type>* input = reg->get_reg_output(input_l, curt_l);

#ifdef DEBUG
    printf("input tensor from %d to %d\n", input_l, curt_l);
#endif

    checkCUDNN( cudnnSoftmaxForward(*(cudnn_h),
                                    this->softmax_alg,
                                    this->mode,
                                    &(this->alpha),
                                    input->get_tensor_desc(),
                                    input->get_gpu_ptr(),
                                    &(this->beta),
                                    this->get_f_out()->get_tensor_desc(),
                                    this->get_f_out()->get_gpu_ptr() ) );
#ifdef DEBUG
    printf("output tensor from %d to %d\n", input_l, curt_l);
    this->get_f_out()->printTensor("softmax result");
#endif
    if (stage == NET_TRAIN) {
        //loss we will compute the loss
        tensor_t<value_type>* label  = reg->get_train_label();
        tensor_t<value_type>* output = this->get_f_out();
        /*
        value_type loss = 0;
        const value_type normalizer = (value_type) label->get_N();
        label->GPUtoCPU();  //TO DO:this should be done on GPU
        output->GPUtoCPU(); //TO DO:this should be done on GPU
        for(size_t i = 0; i < label->get_N(); i++) {
            loss -= log(output->get_scalar(i, 0, 0, label->get_scalar(i, 0, 0, 0)));
        }
        loss = loss / normalizer;
        return loss;
         */
        
         #ifdef BENCHMARK
         double gstart = get_cur_time();
         #endif
         
         value_type gpu_loss = (value_type) softmax_loss<value_type>(output->get_gpu_ptr(), label->get_gpu_ptr(), (int) output->get_N(), (int) output->get_C(), (int) output->get_H(), (int) output->get_W());
         
         #ifdef BENCHMARK
         double gend   = get_cur_time();
         printf("softmax loss time:%f\n", gend - gstart);
         #endif
        std::vector<value_type> loss;
        loss.push_back(gpu_loss);
        return loss;
    } else if( stage == NET_INFER ) {
        //loss we will compute the loss
        tensor_t<value_type>* label  = reg->get_test_label();
        tensor_t<value_type>* output = this->get_f_out();
        const value_type normalizer = (value_type) label->get_N();
       // label->GPUtoCPU();  //TO DO:this should be done on GPU
       // output->GPUtoCPU(); //TO DO:this should be done on GPU
        value_type corr_counter = 0;
        
#ifdef DEBUG
        output->printTensor("output of softmax layer");
#endif
        /*-------top 1 accuracy---------*/
        /*
        for(size_t i = 0; i < output->get_N(); i++) {
            value_type max  = 0;
            size_t pred     = 0;
            value_type corr = label->get_scalar(i, 0, 0, 0);
            for( size_t j = 0; j < output->get_C(); j++ ) {
                value_type curt = output->get_scalar(i, j, 0, 0);
                if( max < curt ) {
                    max   = curt;
                    pred  = j;
                }
            }
            if (corr == pred) {
                corr_counter++;
            }
        }
        value_type accuracy_cpu = corr_counter / (value_type) output->get_N();
        */
        //printf("accuracy cpu:%f\n", accuracy_cpu);

        /*-------top 5 accuracy---------*/
        /*
         corr_counter = 0;
        for(size_t i = 0; i < output->get_N(); i++)
        {
            int idx_top5[5] = {-1, -1, -1, -1, -1};
            int corr = (int) label->get_scalar(i, 0, 0, 0);
            for(int j = 0; j < 5; j++)
            {
                value_type max = 0;
                size_t pred = 0;
                for( size_t k = 0; k < output->get_W(); k++ )
                {
                    if( has_elem(idx_top5, 5, k) ) {
                        continue;
                    }
                    value_type curt = output->get_scalar(i, 0, 0, k);
                    if( max < curt )
                    {
                        max  = curt;
                        pred = k;
                    }
                }
                idx_top5[j] = pred;
            }
            if ( has_elem(idx_top5, 5, corr) ) {
                corr_counter++;
            }
        }
         value_type accuracy_top5_cpu = corr_counter / (value_type) output->get_N();
         */
        value_type accuracy_top1_gpu = softmax_top1_accuracy(label->get_gpu_ptr(), output->get_gpu_ptr(), output->get_N(), output->get_C(), output->get_H(), output->get_W());
        value_type accuracy_top5_gpu = softmax_top5_accuracy(label->get_gpu_ptr(), output->get_gpu_ptr(), output->get_N(), output->get_C(), output->get_H(), output->get_W());
        //printf("accuracy cpu:%f gpu:%f\n", accuracy_top5_cpu, accuracy_top5_gpu);
        std::vector<value_type> accuracy;
        accuracy.push_back(accuracy_top1_gpu);
        accuracy.push_back(accuracy_top5_gpu);

        /*-------------------------------*/
        return accuracy;
    } else {
        printf("Not supported network stage at softmax_layer.cpp@line 62\n");
        return std::vector<value_type>();
    }
}
    
template <class value_type>
void softmax_layer_t<value_type>::backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type>* reg ) {
    
    tensor_t<value_type>* output = this->get_f_out();
    tensor_t<value_type>* label  = reg->get_train_label();
    softmax_grad( output->get_gpu_ptr(), label->get_gpu_ptr(), (int) output->get_N(), (int) output->get_C(), (int) output->get_H(), (int) output->get_W() );
    output->scale( 1.0f / output->get_N() );
#ifdef DEBUG
    output->printTensor("Gradient from Softmax");
#endif
}

INSTANTIATE_CLASS(softmax_layer_t);
    
} //SuperNeurons namespace


