//
// Created by ay27 on 9/14/17.
//

#ifndef SUPERNEURONS_LIVENESS_H
#define SUPERNEURONS_LIVENESS_H

#include <util/common.h>
#include <tensor.h>
#include <registry.h>
#include <layer/base_layer.h>

namespace SuperNeurons {


template<class value_type>
class liveness_analysis_t {
private:
    std::vector<std::vector<void *> > f_stash_tensors;
    std::vector<std::vector<void *> > b_stash_tensors;
    std::vector<std::vector<void *> > f_free_tensors;
    std::vector<std::vector<void *> > b_free_tensors;
    registry_t <value_type> *reg;
    std::map<void *, mem_mode> *regulated_tensors;

    std::vector<std::vector<std::pair<int, net_comp> > > *subsequent_forward;
    std::vector<std::vector<std::pair<int, net_comp> > > *subsequent_backward;

    std::vector<LAYER> CHECKPOINT_LAYERS;

    const int max_layer_id;

    inline std::vector<std::pair<int, net_comp> >& get_subsequent_layers(int curt_layer_id, net_comp dir);

    bool is_used_by_layer(int layer_id, net_comp dir, tensor_t <value_type> *t);

    bool is_freeable_afterwards(int curt_layer_id, net_comp dir, tensor_t <value_type> *t);

    void set_ins(std::vector<std::vector<void *> > *ins, net_comp dir);

    void set_outs(std::vector<std::vector<void *> > *outs, net_comp dir);

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


public:
    liveness_analysis_t(registry_t <value_type> *_reg, std::map<void *, mem_mode> *_regulated_tensors,
                        std::vector<std::vector<std::pair<int, net_comp>>> *_subsequent_forward,
                        std::vector<std::vector<std::pair<int, net_comp>>> *_subsequent_backward,
                        const std::vector<LAYER> &_CHECKPOINT_LAYERS,
                        int _max_layer_id) : reg(_reg), regulated_tensors(_regulated_tensors),
                                             subsequent_forward(_subsequent_forward),
                                             subsequent_backward(_subsequent_backward),
                                             CHECKPOINT_LAYERS(_CHECKPOINT_LAYERS),
                                             max_layer_id(_max_layer_id) {
        set_ins(&f_stash_tensors, FORWARD);
        set_ins(&b_stash_tensors, BACKWARD);
        set_outs(&f_free_tensors, FORWARD);
        set_outs(&b_free_tensors, BACKWARD);

#ifdef DEBUG
        printf("--------f_stash_tensors-----------\n");
        for (int layer_id = 1; layer_id < (int)f_stash_tensors.size(); ++layer_id) {
            printf("layer : %d\n", layer_id);
            for (size_t i = 0; i < f_stash_tensors[layer_id].size(); ++i) {
                tensor_t<value_type>* t = (tensor_t<value_type>*)f_stash_tensors[layer_id][i];
                printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            }
        }
        printf("\n\n");
        printf("--------b_stash_tensors-----------\n");
        for (int layer_id = 1; layer_id < (int)b_stash_tensors.size(); ++layer_id) {
            printf("layer : %d\n", layer_id);
            for (size_t i = 0; i < b_stash_tensors[layer_id].size(); ++i) {
                tensor_t<value_type>* t = (tensor_t<value_type>*)b_stash_tensors[layer_id][i];
                printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            }
        }
        printf("\n\n");
        printf("--------f_free_tensors-----------\n");
        for (int layer_id = 1; layer_id < (int)f_free_tensors.size(); ++layer_id) {
            printf("layer : %d\n", layer_id);
            for (size_t i = 0; i < f_free_tensors[layer_id].size(); ++i) {
                tensor_t<value_type>* t = (tensor_t<value_type>*)f_free_tensors[layer_id][i];
                printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            }
        }
        printf("\n\n");
        printf("--------b_free_tensors-----------\n");
        for (int layer_id = 1; layer_id < (int)b_free_tensors.size(); ++layer_id) {
            printf("layer : %d\n", layer_id);
            for (size_t i = 0; i < b_free_tensors[layer_id].size(); ++i) {
                tensor_t<value_type>* t = (tensor_t<value_type>*)b_free_tensors[layer_id][i];
                printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            }
        }
        printf("\n\n");
#endif
    }

    void stash(int layer_id, net_comp dir);

    void update(int layer_id, net_comp dir);
};


} // namespace SuperNeurons

#endif //SUPERNEURONS_LIVENESS_H
