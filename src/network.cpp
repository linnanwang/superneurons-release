#include <network.h>
#include <tensor.h>
#include <fstream>

namespace SuperNeurons{
    
template <class value_type>
std::vector<double> network_t<value_type>::network_perf_profile() {
    std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
    std::map<int, void* > net_layers  = reg->get_net_layers();
    double max_mem_used = 0;
    for(size_t i = 0; i < net_comp_route.size(); i++) {
        int layer_id = net_comp_route[i].first;
        net_comp dir = net_comp_route[i].second;
        // stash tensors
        std::pair<double, double> stat = mem_controller.stash_tensors_for_profile( layer_id, dir );
        double mem_stage_time =  stat.first;
        double total_mem      =  stat.second;
        double curt_mem = BYTE_TO_MB(query_used_mem());
        if(curt_mem > max_mem_used) max_mem_used = curt_mem;
        // execution
        base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
        double start = get_cur_time();
        for(size_t j = 0; j < 10; j++) {
            if(dir == FORWARD) {
                b->forward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            } else if(dir == BACKWARD) {
                b->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            }
            //cudaStreamSynchronize(stream);
        }
        LAYER layer_type = b->get_layer_type();
        double end  = get_cur_time();
        double avg_time = (end - start)/10.0f;
        mem_controller.free_tensors_for_profile(layer_id, dir);
        double mem_time = mem_stage_time;
        if (dir == FORWARD) {
            printf("at layer id:%3d, type:%3d, compute_time:%5.5f, memory_time:%5.5f, total_mem:%5.5f\n", layer_id, layer_type, avg_time, mem_time, total_mem);
        } else {
            printf("at layer id:%3d, type:%3d, compute_time:%5.5f, memory_time:%5.5f, total_mem:%5.5f\n", layer_id*-1, layer_type, avg_time, mem_time, total_mem);
        }
    }
    printf("Max Memory used in profile:%f\n", max_mem_used);
    return std::vector<double>();
}

template <class value_type>
void network_t<value_type>::gradient_check_kernel(int l_id, size_t n, size_t c, size_t h, size_t w, tensor_t<value_type>* data, tensor_t<value_type>* diff, const char* str) {

    if( n >= data->get_N() ) {
        printf("n: %zu exceeds layer %d ~ N:%zu \n", n, l_id, data->get_N());
        return;
    }
    if( c >= data->get_C() ) {
        printf("c: %zu exceeds layer %d ~ C:%zu \n", c, l_id, data->get_C());
        return;
    }
    if( h >= data->get_H() ) {
        printf("h: %zu exceeds layer %d ~ H:%zu \n", h, l_id, data->get_H());
        return;
    }
    if( w >= data->get_W() ) {
        printf("w:%zu exceeds layer %d ~ W:%zu \n", w, l_id, data->get_W());
        return;
    }

    //analytical gradient
    value_type epsilon = 0.0001;
    value_type tmp     = data->get_scalar(n, c, h, w);
    data->set_scalar( n, c, h, w, tmp+epsilon );

    value_type loss1 = this->forward(NET_TRAIN);
    data->set_scalar( n, c, h, w, tmp-epsilon );
    value_type loss2 = this->forward(NET_TRAIN);

    value_type anly_grad = (loss1 - loss2)/(2*epsilon);
    //numerical gradient
    data->set_scalar( n, c, h, w, tmp); //set the weight
    this->forward(NET_TRAIN);
    this->backward();
    double num_grad = diff->get_scalar(n, c, h, w);

    printf("=> layer %d,n:%zu c:%zu h:%zu w:%zu num_grad:%f anly_grad:%f measure:%f loss1:%f loss2:%f %s\n", l_id, n, c, h, w, num_grad, anly_grad, math.abs_(anly_grad-num_grad)/math.max_(anly_grad, num_grad), loss1, loss2, str);
}

template <class value_type>
void network_t<value_type>::gradient_check(int layer_id) {
    tensor_t<value_type>* weight      = reg->get_reg_weight(layer_id);
    tensor_t<value_type>* weight_grad = reg->get_reg_weight_grad(layer_id);
    if(weight == NULL) {
        printf("layer:%d does not have params\n", layer_id);
        return;
    }
    size_t N = weight->get_N();
    size_t C = weight->get_C();
    size_t H = weight->get_H();
    size_t W = weight->get_W();

    for(size_t i = 0; i < N; i++) {
        for(size_t j = 0; j < C; j++) {
            for(size_t k = 0; k < H; k++) {
                for(size_t m = 0; m < W; m++) {
                    gradient_check_kernel( layer_id, i, j, k, m, weight, weight_grad, "weight");
                }
            }
        }
    }

    tensor_t<value_type>* bias        = reg->get_reg_bias(layer_id);
    tensor_t<value_type>* bias_grad   = reg->get_reg_bias_grad(layer_id);

    N = bias->get_N();
    C = bias->get_C();
    H = bias->get_H();
    W = bias->get_W();

    for(size_t i = 0; i < N; i++) {
        for(size_t j = 0; j < C; j++) {
            for(size_t k = 0; k < H; k++) {
                for(size_t m = 0; m < W; m++) {
                    gradient_check_kernel( layer_id, i, j, k, m, bias, bias_grad, "bias");
                }
            }
        }
    }
}

template <class value_type>
void network_t<value_type>::setup_test(base_layer_t<value_type>* test_data_layer, size_t iter) {

    if(!(this->is_forward_setup && this->is_backward_setup)) {
        printf("please setup the training before testing\n");
        exit(1);
    }

    this->test_iter              = iter;
    this->test_data_layer        = test_data_layer;

    test_data_layer->forward_setup (reg, &cudnn_handle);
    assert(this->test_data_layer != NULL);

    this->reg->register_net_layers(this->test_data_layer->get_base_id(), (void*) this->test_data_layer);
    this->reg->register_net_test_route(this->test_data_layer->get_base_id());
    this->reg->print_net_test_route();
    //points to the same layer, but the first network layer shall switch between these two
    this->is_testing_ready = true;
}


    

template <class value_type>
void network_t<value_type>::forward_kernel(network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* loss) {
    std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
    std::map<int, void* > net_layers  = reg->get_net_layers();
    for(size_t i = 0; i < net_comp_route.size(); i++) {
        if( net_comp_route[i].second == FORWARD ) {
            int layer_id = net_comp_route[i].first;
            // stash tensors
            mem_controller.stash_tensor( layer_id, FORWARD , NET_TRAIN);

            // execution
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
            *loss = b->forward(stage, &cublas_handle, &cudnn_handle, reg);
            // update tensors
            mem_controller.update_tensor_state(layer_id, FORWARD, stage);

#ifdef DEBUG
            printf("forward finish layer %zu %d\n", i, layer_id);
#endif
        }
    }
}

template <class value_type>
void network_t<value_type>::backward_with_update_kernel(base_layer_t<value_type>* l, size_t iter) {
    std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
    std::map<int, void* > net_layers  = reg->get_net_layers();

    for( size_t i = 0; i < net_comp_route.size(); i++ ) {
        if( net_comp_route[i].second == BACKWARD ) {
            int layer_id = net_comp_route[i].first;
            // get tensors
            mem_controller.stash_tensor( layer_id, BACKWARD , NET_TRAIN);
            // execution
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
            b->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            // when backward finish, do the update immediately
            /*
            tensor_t<value_type>* weight_grad = this->reg->get_reg_weight_grad( layer_id );
            if( weight_grad != NULL ) {
                weight_grad->clip(this->get_cublas_handle());
                std::string idx = std::to_string( i );
                std::string filename = "gradient" + idx;
                weight_grad->writeToFile(filename.c_str());
            }
            */
            b->update(&cublas_handle, iter, solver);
            // update tensors
            mem_controller.update_tensor_state(layer_id, BACKWARD, NET_TRAIN);

#ifdef DEBUG
            printf("backward finish layer %d\n", layer_id);
#endif
        }
    }
    mem_controller.reset_tensor_state();
}
    
template <class value_type>
void network_t<value_type>::update_kernel(base_layer_t<value_type>* b, size_t iter) {
    for(size_t i = 0; i < reg->get_net_comp_route().size(); i++) {
        if( reg->get_net_comp_route()[i].second == FORWARD ) {
            int key = reg->get_net_comp_route()[i].first;
            base_layer_t<value_type>* b = (base_layer_t<value_type>*)reg->get_net_layers().find(key)->second;
            b->update(&cublas_handle, iter, solver);
        }
    }
}
    
template <class value_type>
void network_t<value_type>::backward_kernel(base_layer_t<value_type>* b) {
    
    std::vector<std::pair<int, net_comp> > net_comp_route = reg->get_net_comp_route();
    std::map<int, void* > net_layers  = reg->get_net_layers();
    
    for( size_t i = 0; i < net_comp_route.size(); i++ ) {
        if( net_comp_route[i].second == BACKWARD ) {
            int layer_id = net_comp_route[i].first;
            // get tensors
            mem_controller.stash_tensor( layer_id, BACKWARD , NET_TRAIN);
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
            b->backward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
            mem_controller.update_tensor_state(layer_id, BACKWARD, NET_TRAIN);
        }
    }
    value_type sum = reg->get_grad_sqrsum();
    //LOG(INFO)<<"###grad sum"<<sum<<"sqrted:"<<std::sqrt(sum);
}

template <class value_type>
void network_t<value_type>::forward_test(network_stage stage, base_layer_t<value_type>* b, std::vector<value_type>* acc) {
    
//NEED REPLACE the data layer in net_comp_route!!!!
    std::vector<std::pair<int, net_comp> > net_test_route = reg->get_net_test_route();
    std::map<int, void* > net_layers  = reg->get_net_layers();
    
    for(size_t i = 0; i < net_test_route.size(); i++) {
        if( net_test_route[i].second == FORWARD ) {
            // get the necessary tensors sorted out before calling forward
            int layer_id = net_test_route[i].first;
            // stash tensors
            mem_controller.stash_tensor( layer_id, FORWARD , NET_INFER);
            
            base_layer_t<value_type>* b = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
            *acc = b->forward(stage, &cublas_handle, &cudnn_handle, reg);
            
            // update tensors
            mem_controller.update_tensor_state(layer_id, FORWARD, stage);
        }
    }
    mem_controller.reset_tensor_state();
}

template <class value_type>
void network_t<value_type>::fsetup_kernel(base_layer_t<value_type>* b) {
    if(b == NULL) return;

    //conduct forward computation
    b->fcounter_inc();

    if(b->get_fcounter() < b->get_prev_size() ) {
        return;
    }
    b->forward_setup(reg, &cudnn_handle);
    this->reg->register_net_layers(b->get_base_id(), (void*) b);
    this->reg->register_net_comp_route(b->get_base_id(), FORWARD);
    
    std::vector<base_layer_t<value_type>*> next = b->get_next();
    if(next.size() == 1) {
        //regular network layer
        fsetup_kernel(next[0]);
    } else if(next.size() > 1) {
        //fork layer
        for(size_t i = 1; i < next.size(); i++) {
            fsetup_kernel(next[i]);
        }
        fsetup_kernel(next[0]);
    }
    b->reset_fc_counter();
}

template <class value_type>
void network_t<value_type>::bsetup_kernel(base_layer_t<value_type>* b) {
    if(b == NULL) return;
    //conduct forward computation
    b->fcounter_inc();
#ifdef DEBUG
    printf("@layer %p:%d fcounter:%zu get_next_size:%zu \n", b, b->get_base_id(), b->get_fcounter(), b->get_next_size());
#endif

    if(b->get_fcounter() < b->get_next_size() ) {
        return;
    }
    b->backward_setup(reg, &cudnn_handle);
    this->reg->register_net_comp_route(b->get_base_id(), BACKWARD);

    std::vector<base_layer_t<value_type>*> prev = b->get_prev();
    if(prev.size() == 1) {
        //regular network layer
        bsetup_kernel(prev[0]);
    } else if(prev.size() > 1) {
        //fork layer
        for(size_t i = 1; i < prev.size(); i++) {
            bsetup_kernel(prev[i]);
        }
        bsetup_kernel(prev[0]);
    }
    b->reset_fc_counter();
}


template <class value_type>
void network_t<value_type>::test() {
    assert( this->test_data_layer != NULL );
    assert( this->train_data_layer != NULL );
    //let's swap the head of data layer
    //please note prev matters!!!
    //the first network layer will need prev to figure out the input
    base_layer_t<value_type>* train_l = this->train_data_layer;
    base_layer_t<value_type>* test_l  = this->test_data_layer;
    std::vector<base_layer_t<value_type>*> next_ls = train_l->get_next();
    assert(next_ls.size() == 1);
    base_layer_t<value_type>* next_l = next_ls[0];
    next_l->switch_prev_l_to(test_l);

    value_type cumulative_acc_top1 = 0;
    value_type cumulative_acc_top5 = 0;
    base_layer_t<value_type>* start = this->test_data_layer;

    for(size_t i = 0; i < this->test_iter; i++) {
        checkCudaErrors( cudaDeviceSynchronize() );
        std::vector<value_type> tmp;
        forward_test(NET_INFER, start, &tmp);
        cumulative_acc_top1 += tmp[0];
        cumulative_acc_top5 += tmp[1];
    }

    value_type test_accuracy_top1 = cumulative_acc_top1 / (value_type) this->test_iter;
    value_type test_accuracy_top5 = cumulative_acc_top5 / (value_type) this->test_iter;

    printf("-------test accuracy--top 1 %f top 5 %f-------\n", test_accuracy_top1, test_accuracy_top5);
    next_l->switch_prev_l_to(train_l);

}

INSTANTIATE_CLASS(network_t);

} //SuperNeuron namespace
