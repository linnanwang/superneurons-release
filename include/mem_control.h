#ifndef SUPERNEURONS_MEM_CONTROL_H
#define SUPERNEURONS_MEM_CONTROL_H

#include <thread>
#include <tensor.h>
#include <registry.h>
#include <util/common.h>
#include <layer/base_layer.h>
#include <util/thread_routine.h>
#include <util/lru.h>
#include <liveness.h>
#include <recompute.h>

namespace SuperNeurons {

template<class value_type>
class mem_controller_t {
private:

    std::vector<LAYER> CHECKPOINT_LAYERS;

    registry_t<value_type>* reg;
    
    std::map<tensor_t<value_type>*, mem_mode> regulated_tensors;

    std::vector<std::vector<std::pair<int, net_comp> > > subsequent_forward;
    std::vector<std::vector<std::pair<int, net_comp> > > subsequent_backward;

    
    void print_required_tensor(int layer_id, net_comp dir);
    
    void print_layer_type(int layer_id, net_comp dir);

    int max_layer_id=0;

    liveness_analysis_t<value_type>* live_anls;
    recompute_t<value_type>*         recomp;

    void set_regulated_tensors();

public:
    mem_controller_t() {
        CHECKPOINT_LAYERS.push_back(CONV);
        CHECKPOINT_LAYERS.push_back(FC);
        CHECKPOINT_LAYERS.push_back(JOIN_L);
        CHECKPOINT_LAYERS.push_back(FORK_L);
        CHECKPOINT_LAYERS.push_back(CONCAT);
    }
    
    ~mem_controller_t() {
//        print_regulated_tensors(true);
    }
    
    void init(registry_t<value_type> *r);

    /***************************************/
    // for profile
    std::pair<double, double> stash_tensors_for_profile(int curt_layer_id, net_comp dir);
    void free_tensors_for_profile(int curt_layer_id, net_comp dir);
    /***************************************/

    void print_regulated_tensors(bool log=false, int layer_id=-1);
    
    void reset_tensor_state();
    
    void stash_tensor(int layer_id, net_comp dir, network_stage stage);
    
    void update_tensor_state(int layer_id, net_comp dir, network_stage stage);
};
    
} // namespace SuperNeurons

#endif //SUPERNEURONS_INITIALIZER_H
