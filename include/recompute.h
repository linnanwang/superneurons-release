//
// Created by ay27 on 9/14/17.
//

#ifndef SUPERNEURONS_RECOMPUTE_H
#define SUPERNEURONS_RECOMPUTE_H

#include <util/common.h>
#include <tensor.h>
#include <layer/base_layer.h>
#include <stream_singleton.h>

namespace SuperNeurons {

template<class value_type>
class recompute_t {
private:
    registry_t <value_type> *reg;
    std::map<void *, mem_mode> *regulated_tensors;
    std::vector<std::vector<void *> > offload_tensors;      // layer_id, free tensor
    std::vector<std::vector<void *> > recompute_free_map;   // layer id, tensor should be free in recompute forwarad
    std::vector<LAYER> CHECKPOINT_LAYERS;
    size_t bottleneck_mem_size;
    std::vector<size_t> f_mem_usage;    // layer_id & size
    std::vector<size_t> b_mem_usage;    // layer_id & size
    lru_list_t *lru = lru_singleton::get_lru();
    std::vector<std::pair<void *, int> > checkpoints;    // curt layer_id, <pre checkpoint, correct branch idx>
    std::vector<std::pair<size_t, bool> > seg_mems;  // <segment forward memory usage, larger than bottleneck>
    std::vector<bool> should_recompute;     // log the layer should recompute or not

    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
    cudaStream_t stream = stream_singleton::get_compute_stream();
    const int max_layer_id;
    bool is_re_init_finish = false;
    std::vector<bool> forward_flags;        // check if it's the second iter of first layer

    void find_bottleneck();

    inline bool is_checkpoint(base_layer_t <value_type> *l) {
        for (auto it = CHECKPOINT_LAYERS.begin(); it != CHECKPOINT_LAYERS.end(); ++it) {
            if (l->get_layer_type() == *it) {
                return true;
            }
            if (l->get_layer_type() == DATA_L || l->get_layer_type() == SOFTMAX) {
                return true;
            }
        }
        return false;
    }

    inline bool is_forward_dependency(tensor_t<value_type>* t, int layer_id) {
        for (size_t i = 0; i < reg->get_forward_dependency(layer_id)->size(); ++i) {
            if (t == reg->get_forward_dependency(layer_id)->operator[](i)) {
                return true;
            }
        }
        return false;
    }

    inline bool is_backward_dependency(tensor_t<value_type>* t, int layer_id) {
        for (size_t i = 0; i < reg->get_backward_dependency(layer_id)->size(); ++i) {
            if (t == reg->get_backward_dependency(layer_id)->operator[](i)) {
                return true;
            }
        }
        return false;
    }

    bool is_in_offload_list(void* t) {
        for (size_t i = 0; i < offload_tensors.size(); ++i) {
            for (size_t j = 0; j < offload_tensors[i].size(); ++j) {
                if (offload_tensors[i][j] == t) {
                    return true;
                }
            }
        }
        return false;
    }

    void log_should_recompute() {
        should_recompute.clear();
        should_recompute.resize(max_layer_id+1);

        for (int layer_id = 1; layer_id <= max_layer_id; ++layer_id) {
            auto tmp = reg->get_net_layers().find(layer_id);
            if (tmp == reg->get_net_layers().end()) {
                continue;
            }
            base_layer_t<value_type>* l = (base_layer_t<value_type>*)tmp->second;
            if (! l->get_prev().empty()) {
                tensor_t<value_type> *input_tensor = reg->get_reg_output(l->get_prev()[0]->get_base_id(), layer_id);
                if (is_in_offload_list(input_tensor) && is_backward_dependency(input_tensor, layer_id)) {
                    should_recompute[layer_id] = true;
                    continue;
                }
            }

            if ( ! l->get_next().empty()) {
                tensor_t<value_type> *output_tensor = reg->get_reg_output(layer_id, l->get_next()[0]->get_base_id());
                if (is_in_offload_list(output_tensor) && is_backward_dependency(output_tensor, layer_id)) {
                    should_recompute[layer_id] = true;
                }
            }
        }

#ifdef DEBUG
        printf("--------should recompute---------\n");
        for (int layer_id = 1; layer_id <= max_layer_id; ++layer_id) {
            printf("layer %d : %s\n", layer_id, should_recompute[layer_id] ? "true" : "false");
        }
#endif
    }

    void scan_checkpoint();

    void scan_offload_tensors();

    void scan_recompute_free_tensor();

    inline void init() {
        find_bottleneck();
        scan_checkpoint();
        scan_offload_tensors();
        scan_recompute_free_tensor();
        log_should_recompute();
    }


    void stash_f_tensor(base_layer_t <value_type> *l);

    void update_f_tensor(base_layer_t <value_type> *l, int curt_recomputed_layer_id);


    void recompute_for_dependency(int curt_layer_id, network_stage stage);

public:
    recompute_t(registry_t <value_type> *_reg,
                std::map<void *, mem_mode> *_regulated_tensors,
                const std::vector<LAYER> &_CHECKPOINT_LAYERS,
                const int _max_layer_id)
            : reg(_reg), regulated_tensors(_regulated_tensors),
              CHECKPOINT_LAYERS(_CHECKPOINT_LAYERS), max_layer_id(_max_layer_id) {

        checkCUDNN(cudnnCreate(&cudnn_handle));
        cudnnSetStream(cudnn_handle, stream);
        checkCublasErrors(cublasCreate(&cublas_handle));
        cublasSetStream(cublas_handle, stream);

        init();

        forward_flags.resize(max_layer_id+1);
        for (int i = 0; i <= max_layer_id; ++i) {
            forward_flags[i] = false;
        }
    }

    virtual ~recompute_t() {
        checkCUDNN(cudnnDestroy(cudnn_handle));
        checkCublasErrors(cublasDestroy(cublas_handle));
    }

    inline void re_init();

    void offload_to_recompute(int layer_id, net_comp dir, network_stage stage);

    void upload_to_reconstruct(int layer_id, net_comp dir, network_stage stage);
};


} // namespace SuperNeurons

#endif //SUPERNEURONS_RECOMPUTE_H
