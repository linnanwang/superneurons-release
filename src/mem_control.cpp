#include <mem_control.h>
#include <util/mem_util.h>
//#define MEM_DEBUG


namespace SuperNeurons{


template <class value_type>
void mem_controller_t<value_type>::init(registry_t<value_type> *r) {
    this->reg = r;

    set_regulated_tensors();

    // set liveness, prefetch, recompute

    max_layer_id = -1;
    for (auto it = r->get_net_layers().begin(); it!=r->get_net_layers().end(); ++it) {
        if (it->first > max_layer_id) {
            max_layer_id = it->first;
        }
    }

    // scan the subsequent_forward and subsequent_backward
    subsequent_forward.resize(max_layer_id+1);
    subsequent_backward.resize(max_layer_id+1);

    std::vector<std::pair<int, net_comp> > net_route = reg->get_net_comp_route();
    for (int layer_id = 1; layer_id <= max_layer_id; ++layer_id) {
        subsequent_forward[layer_id].resize(0);
        subsequent_backward[layer_id].resize(0);
        bool is_subsequent = false;
        for(size_t i = 0; i < net_route.size(); i++) {
            if( is_subsequent == true ) {
                subsequent_forward[layer_id].push_back(net_route[i]);
            }
            if( net_route[i].first == layer_id && net_route[i].second == FORWARD ) {
                is_subsequent = true;
            }
        }
        is_subsequent = false;
        for(size_t i = 0; i < net_route.size(); i++) {
            if( is_subsequent == true ) {
                subsequent_backward[layer_id].push_back(net_route[i]);
            }
            if( net_route[i].first == layer_id && net_route[i].second == BACKWARD ) {
                is_subsequent = true;
            }
        }
    }

#ifdef LIVENESS
    live_anls = new liveness_analysis_t<value_type>(reg,
                                                    (std::map<void *, mem_mode>*)&regulated_tensors,
                                                    &subsequent_forward,
                                                    &subsequent_backward,
                                                    CHECKPOINT_LAYERS,
                                                    max_layer_id);
#endif
#ifdef RECOMPUTE_ON
    recomp = new recompute_t<value_type>(reg, (std::map<void *, mem_mode>*)&regulated_tensors, CHECKPOINT_LAYERS, max_layer_id);
#endif
}


/*--------network profile---------*/
template <class value_type>
std::pair<double, double> mem_controller_t<value_type>::stash_tensors_for_profile(int curt_layer_id, net_comp dir) {

    std::vector<tensor_t<value_type>* >* tensors = NULL;
    if(dir == FORWARD) {
        tensors = reg->get_forward_dependency(curt_layer_id); // TO DO, we don't track the data layer!!
    } else if(dir == BACKWARD) {
        tensors = reg->get_backward_dependency(curt_layer_id);
    }
    if(tensors == NULL) return std::make_pair(0, 0);

    size_t total_mem = 0;
    double total_time  = 0;
    for(size_t i = 0; i < tensors->size(); i++) {
        tensor_t<value_type>* t = tensors->operator[](i);
        if( t->get_type() != DATA && t->get_type() != CONV_BUFF ) continue;
        if( t->get_type() == DATA) {
            total_mem += t->get_mem_size();
        }
//        t->atomic_set_state(GPU);
        t->stash_gpu_space();
        for(size_t j = 0; j < 10; j++) {
            double start = get_cur_time();
            t->CPUtoGPU();
            double end   = get_cur_time();
            total_time += (end - start);
        }
    }
    double avg_time = total_time / 10.0f;

    double mem_in_mb = total_mem/1000000.0f;

#ifdef MEM_DEBUG
    double curt_mem = query_gpu_mem();
    if (dir == FORWARD) {
        printf("------------forward:%d(%f MB) stash------------\n", curt_layer_id, curt_mem);
    } else {
        printf("------------backward:%d(%f MB) stash-----------\n", curt_layer_id, curt_mem);
    }
    this->print_regulated_tensors();
#endif
    return std::make_pair(avg_time, mem_in_mb);
}

template <class value_type>
void mem_controller_t<value_type>::free_tensors_for_profile(int curt_layer_id, net_comp dir) {
    std::vector<tensor_t<value_type>* >* tensors = NULL;
    if(dir == FORWARD) {
        tensors = reg->get_forward_dependency(curt_layer_id); // TO DO, we don't track the data layer!!
    } else if(dir == BACKWARD) {
        tensors = reg->get_backward_dependency(curt_layer_id);
    }
    if(tensors == NULL) return;

    size_t total_mem = 0;
    double total_time  = 0;
    for(size_t i = 0; i < tensors->size(); i++) {
        tensor_t<value_type>* t = tensors->operator[](i);
        if( t->get_type() != DATA && t->get_type() != CONV_BUFF ) continue;
        if( t->get_type() == DATA) {
            total_mem += t->get_mem_size();
        }
//        t->atomic_set_state(VOID);
        t->free_gpu_space();
    }

    double avg_time = total_time / 10.0f;
#ifdef MEM_DEBUG
    double curt_mem = query_gpu_mem();
    if (dir == FORWARD) {
        printf("------------forward:%d(%f MB) update------------\n", curt_layer_id, curt_mem);
    } else {
        printf("------------backward:%d(%f MB) update-----------\n", curt_layer_id, curt_mem);
    }
    this->print_regulated_tensors();
#endif
}
/*-----------------------------------*/


template <class value_type>
void mem_controller_t<value_type>::reset_tensor_state() {
#ifdef LIVENESS
    for (auto it = regulated_tensors.begin(); it != regulated_tensors.end(); it++ ) {
        it->first->free_gpu_space(VOID);
    }
#endif
}


template <class value_type>
void mem_controller_t<value_type>::print_layer_type(int layer_id, net_comp dir) {
    std::map<int, void* > net_layers     = reg->get_net_layers();
    base_layer_t<value_type>* curt_layer = (base_layer_t<value_type>*) net_layers.find(layer_id)->second;
    LAYER curt_layer_type                = curt_layer->get_layer_type();
    /*
     CONV    = 0,
     POOL    = 1,
     ACT     = 2,
     BN      = 3,
     FC      = 4,
     LRN     = 5,
     PADDING = 6,
     DATA_L  = 7,
     DROPOUT = 8,
     SOFTMAX = 9,
    CONCAT  = 10,
    FORK_L  = 11,
    JOIN_L  = 12
     */

    if( curt_layer_type == CONV ) {
        printf("@layer %d, CONV\n", layer_id);
    } else if( curt_layer_type == POOL ) {
        printf("@layer %d, POOL\n", layer_id);
    } else if( curt_layer_type == ACT ) {
        printf("@layer %d, ACT\n", layer_id);
    } else if( curt_layer_type == BN ) {
        printf("@layer %d, BN\n", layer_id);
    } else if( curt_layer_type == FC ) {
        printf("@layer %d, FC\n", layer_id);
    } else if( curt_layer_type == LRN ) {
        printf("@layer %d, LRN\n", layer_id);
    } else if( curt_layer_type == PADDING ) {
        printf("@layer %d, PADDING\n", layer_id);
    } else if( curt_layer_type == DATA_L ) {
        printf("@layer %d, DATA_L\n", layer_id);
    } else if( curt_layer_type == DROPOUT ) {
        printf("@layer %d, DROPOUT\n", layer_id);
    } else if( curt_layer_type == CONCAT ) {
        printf("@layer %d, CONCAT\n", layer_id);
    } else if( curt_layer_type == FORK_L ) {
        printf("@layer %d, FORK_L\n", layer_id);
    } else if( curt_layer_type == JOIN_L ) {
        printf("@layer %d, JOIN_L\n", layer_id);
    } else if( curt_layer_type == SOFTMAX ) {
        printf("@layer %d, SOFTMAX\n", layer_id);
    } else {
        printf("@layer %d, UNKNOWN.....\n", layer_id);
    }
}

/*--------control--start--------------------*/
template <class value_type>
void mem_controller_t<value_type>::stash_tensor(int layer_id, net_comp dir, network_stage stage) {

#ifdef DEBUG
    auto tmp = reg->get_net_layers().find(layer_id);
#endif

#ifdef LIVENESS
    live_anls->stash(layer_id, dir);
    print_regulated_tensors();
#endif

#ifdef RECOMPUTE_ON
    recomp->upload_to_reconstruct(layer_id, dir, stage);
#endif
}


template <class value_type>
void mem_controller_t<value_type>::update_tensor_state(int layer_id, net_comp dir, network_stage stage) {

#ifdef BENCHMARK
    // collect conv buff size
    auto it = reg->get_net_layers().find(layer_id);
    if (it != reg->get_net_layers().end()) {
        base_layer_t<value_type>* l = (base_layer_t<value_type>*)it->second;
        if (l->get_layer_type() == CONV) {

        }
    }
#endif

#ifdef LIVENESS
    live_anls->update(layer_id, dir);
#endif
#ifdef RECOMPUTE_ON
    recomp->offload_to_recompute(layer_id, dir, stage);
#endif

}

/*--------control--end--------------------*/



template <class value_type>
void mem_controller_t<value_type>::set_regulated_tensors() {
    std::map<int, std::vector< tensor_t<value_type>* > > tensor_by_layer = reg->get_tensor_by_layer();
    typename std::map<int, std::vector< tensor_t<value_type>* > >::iterator it = tensor_by_layer.begin();
    for (it = tensor_by_layer.begin(); it != tensor_by_layer.end(); ++it) {
        std::vector<tensor_t<value_type>* > tensors = it->second;
        for(size_t i = 0; i < tensors.size(); i++) {
            if(tensors[i]->get_type() != DATA && tensors[i]->get_type() != CONV_BUFF ) {
                continue;
            }
            typename std::map<tensor_t<value_type>*, mem_mode>::iterator it2 = regulated_tensors.find(tensors[i]);
            if (it2 == regulated_tensors.end()) {
                regulated_tensors.insert( std::make_pair(tensors[i], VOID) );
            }
        }
    }
}

/*--------print helper start------------*/

template <class value_type>
void mem_controller_t<value_type>::print_required_tensor(int layer_id, net_comp dir) {
    std::vector<tensor_t<value_type>* >* tensors = NULL;
    if(dir == FORWARD) {
        tensors = reg->get_forward_dependency(layer_id); // TO DO, we don't track the data layer!!
    } else if(dir == BACKWARD) {
        tensors = reg->get_backward_dependency(layer_id);
    }
    if(tensors == NULL) return;
    for(size_t i = 0; i < tensors->size(); i++) {
        tensor_t<value_type>* t = tensors->operator[](i);
        printf("TENSOR NEEDED layer %d : %p state:%d type:%d gpu_ptr:%p\n",
               t->get_layer_id(), t, t->get_state(), t->get_type(), t->get_gpu_ptr() );
    }
}

template <class value_type>
void mem_controller_t<value_type>::print_regulated_tensors(bool log, int layer_id) {
    if (log) {
        int total=0, hit=0, miss=0;
        typename std::map<tensor_t<value_type> *, mem_mode>::iterator it = regulated_tensors.begin();
        for (it = regulated_tensors.begin(); it != regulated_tensors.end(); it++) {
            if (it->first->get_type() == CONV_BUFF || it->first->into_cnt == 0) {
                continue;
            }
            printf("layer %d tensor %p type %d: total=%d, hit=%d, miss=%d\n",
                   it->first->get_layer_id(), it->first, it->first->get_type(),
                   it->first->into_cnt, it->first->hit_cnt, it->first->miss_cnt);
            total += it->first->into_cnt;
            hit += it->first->hit_cnt;
            miss += it->first->miss_cnt;
        }
        printf("Summary total=%d, hit=%d, miss=%d\n", total, hit, miss);
        printf("hit / total = %f\n", (double)hit / (double) total);

        size_t conv_buff = 0;
        for (it = regulated_tensors.begin(); it != regulated_tensors.end(); it++) {
            if (it->first->get_type() == CONV_BUFF || it->first->into_cnt == 0) {
                conv_buff += it->first->get_mem_size();
                printf("layer %d  conv buff %zu  %.3fMB\n",
                       it->first->get_layer_id(), it->first->get_mem_size(), (double)(it->first->get_mem_size())/1024.0/1024.0);
            }
        }
        printf("conv buff = %zu  %.3fMB\n", conv_buff, (double)conv_buff/1024.0/1024.0);

    }

#ifdef DEBUG

    int  data_c          = 0;
    int  conv_buff_c     = 0;

    long total_data_type = 0;
    long total_data_type_gpu = 0;

    long total_conv_buff = 0;
    long total_conv_buff_gpu = 0;

    long total_gpu_data  = 0;
    long total_cpu_data  = 0;
    int total_live_cnt   = 0;
    int ff = 0;
    int bb = 0;

    typename std::map<tensor_t<value_type>*, mem_mode>::iterator it = regulated_tensors.begin();
    for( it = regulated_tensors.begin(); it != regulated_tensors.end(); it++ ) {
        tensor_t<value_type>* t = it->first;
        // or t->get_state() == CPU2GPU or t->get_state() == GPU2CPU
        // do not accumulate other state
        if ( t->get_gpu_ptr() != NULL ) {
            total_gpu_data += t->get_scalar_count();

            if (t->get_type() == DATA) {
                total_live_cnt += 1;

                bool flag = true;
                for (int layer = 3; layer <= max_layer_id; ++layer) {
                    std::vector<tensor_t<value_type> *> *f_tensors = reg->get_forward_dependency(layer);
                    for (auto tt = f_tensors->begin(); tt != f_tensors->end(); ++tt) {
                        if (t == (*tt)) {
                            ff += 1;
                            flag = false;
                            break;
                        }
                    }
                    if (!flag) {
                        break;
                    }
                }

                flag = true;
                for (int layer = 3; layer <= max_layer_id; ++layer) {
                    std::vector<tensor_t<value_type> *> *b_tensors = reg->get_backward_dependency(layer);
                    for (auto tt = b_tensors->begin(); tt != b_tensors->end(); ++tt) {
                        if (t == (*tt)) {
                            bb += 1;
                            flag = false;
                            break;
                        }
                    }
                    if (!flag) {
                        break;
                    }
                }
            }


        } else if( t->get_gpu_ptr() == NULL ) {
            total_cpu_data += t->get_scalar_count();
        }

        double tensor_size = t->get_mem_size()/1024.0f/1024.0f;
        if(t->get_type() == DATA) {
            data_c += 1;
            if (t->get_gpu_ptr() != NULL) {
                printf("size:%5.2fMB mem_mode:%2d layer:%d DATA tensor:%p gpu_ptr:%p\n", tensor_size, t->get_state(),
                       t->get_layer_id(), t, t->get_gpu_ptr());
                total_data_type_gpu += ((long) t->get_scalar_count());
            }
            total_data_type += ((long) t->get_scalar_count());
        } else if(t->get_type() == CONV_BUFF) {
            conv_buff_c += 1;
            if (t->get_gpu_ptr() != NULL) {
                printf("size:%5.2fMB mem_mode:%2d layer:%d BUFF tensor:%p gpu_ptr:%p\n", tensor_size, t->get_state(),
                       t->get_layer_id(), t, t->get_gpu_ptr());

                total_conv_buff_gpu += ((long) t->get_scalar_count());
            }
            total_conv_buff += ((long) t->get_scalar_count());
        } else {
            printf("ERROR UNTRACKED tensor:%p size:%5.2ffMB mem_mode:%d gpu_ptr:%p\n", t, tensor_size, t->get_state(), t->get_gpu_ptr());
        }
    }
    double total_data_type_mem   = total_data_type*sizeof(value_type)/1024.0f/1024.0f;
    double total_data_gpu_mem    = total_data_type_gpu* sizeof(value_type) / 1024.0f / 1024.0f;

    double total_conv_buff_mem   = total_conv_buff*sizeof(value_type)/1024.0f/1024.0f;
    double total_conv_buff_gpu_mem = total_conv_buff_gpu* sizeof(value_type) / 1024.0f / 1024.0f;

    double total_gpu_data_mem   = total_gpu_data*sizeof(value_type)/1024.0f/1024.0f;
    double total_cpu_data_mem    = total_cpu_data*sizeof(value_type)/1024.0f/1024.0f;

    printf("TOTAL CPU TENSOR:%8.2fMB, TOTAL GPU TENSOR:    %8.2fMB\n",
           total_cpu_data_mem, total_gpu_data_mem);
    printf("TOTAL DATA:      %8.2fMB, TOTAL GPU DATA:      %8.2fMB\n",
           total_data_type_mem, total_data_gpu_mem);
    printf("TOTAL CONV_BUFF: %8.2fMB, TOTAL GPU CONV_BUFF: %8.2fMB\n",
            total_conv_buff_mem, total_conv_buff_gpu_mem);

#ifdef LRU_ON
    lru_singleton::get_lru()->print_list();
#endif

    printf("free gpu memory : %f MB\n", BYTE_TO_MB(query_free_mem()));


#endif

}

INSTANTIATE_CLASS(mem_controller_t);

} //SuperNeuron namespace
