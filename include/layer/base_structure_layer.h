#if !defined(_BASE_STRUCTURE_LAYER_H_)
#define _BASE_STRUCTURE_LAYER_H_
#include <tensor.h>
#include <registry.h>
#include <util/common.h>
#include <layer/base_layer.h>

namespace SuperNeurons{
    
template <class value_type>
class base_structure_t:base_layer_t<value_type>
{
private:
    std::vector<tensor_t<value_type>* > inputs;  //JOIN layer inputs
    std::vector<tensor_t<value_type>* > outputs; //FORK layer outputs
    std::vector<tensor_t<value_type>* > b_data;  //FORK layer has one, while JOIN has multiples
    structure_type type;
    
public:
    
    base_structure_t(LAYER lt):base_layer_t<value_type>(lt) {}
    
    inline int get_id() {
        return this->get_base_id();
    }
    
    virtual std::vector<value_type> forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg)  = 0;
    virtual void backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg) = 0;
    virtual void forward_setup(registry_t<value_type>* reg = NULL, cudnnHandle_t* cudnn_h = NULL)  = 0;
    virtual void backward_setup(registry_t<value_type>* reg = NULL, cudnnHandle_t* cudnn_h = NULL) = 0;
    virtual void gen_description(char* buff, size_t* len_in_byte) = 0;

    void gen_meta_description(char* buff, size_t* len_in_byte) {
        this->_gen_meta_description(buff, len_in_byte);
    }

    void update(cublasHandle_t *cublas_h, size_t iter, base_solver_t<value_type>* solver ) {
    //structure layer do not update
    };
    
    std::vector<std::pair<int, int> > get_inputs_keys() {
        std::vector<base_layer_t<value_type>* > prev_layers = this->get_prev();
        int curt_l_id = this->get_base_id();
        std::vector<std::pair<int, int> > result;
        
        for(size_t i = 0; i < prev_layers.size(); i++) {
            int prev_id = prev_layers[i]->get_base_id();
            d_key_t key(prev_id, curt_l_id);
            result.push_back(key);
        }
        //verification
        if (type == FORK) {
            assert(result.size() <= 1);
        } else {
            assert(result.size() >= 1);
        }
        return result;
    }
    
    std::vector<std::pair<int, int> > get_outputs_keys() {
        std::vector<base_layer_t<value_type>* > next_layers = this->get_next();
        int curt_l_id = this->get_base_id();
        std::vector<std::pair<int, int> > result;
        
        if(next_layers.size() == 0) {
            d_key_t key(curt_l_id, 0);
            result.push_back(key);
        } else {
            for(size_t i = 0; i < next_layers.size(); i++) {
                int next_id = next_layers[i]->get_base_id();
                d_key_t key(curt_l_id, next_id);
                result.push_back(key);
            }
        }
        
        //verification
        if (type == FORK) {
            assert(result.size() >= 1);
        } else {
            assert(result.size() <= 1);
        }
        return result;
    }
    
    void set_input(tensor_t<value_type>* t) {
        //the input a FORK layer has to be one
        if(type == FORK) {
            inputs.push_back(t);
            assert(inputs.size() <= 1);
        } else {
            inputs.push_back(t);
        }
        printf("@layer%d curt_input size %lu\n", get_id(), inputs.size() );
        if(inputs.size() == 0) {
            return;
        } else {
            tensor_t<value_type>* tmp = inputs[0];
            assert(tmp->get_N() == t->get_N());
            // we don't check the channel size
            // assert(tmp->get_C() == t->get_C());
            assert(tmp->get_H() == t->get_H());
            assert(tmp->get_W() == t->get_W());
        }
    }
    
    //by convention, each layer only holds the output tensors
    void set_output(tensor_t<value_type>* t, std::pair<int, int> idx, registry_t<value_type>* reg) {
        //the output of a FORK layer has to be one
        if(type == JOIN) {
            outputs.push_back(t);
            assert(outputs.size() <= 1);
        } else {
            outputs.push_back(t);
        }
        reg->register_output(idx.first, idx.second, t);
        printf("@layer%d curt_output size %lu\n", get_id(), inputs.size() );
    }
    
    void set_b_data(tensor_t<value_type>* t, std::pair<int, int> idx, registry_t<value_type>* reg) {
        //the output of a FORK layer has to be one
        if(type == JOIN) {
            b_data.push_back(t);
            assert(b_data.size() >= 1);
            //b data is the reverse of input pair
        } else if(type == FORK) {
            b_data.push_back(t);
            assert(b_data.size() <= 1);
        }
        reg->register_b_data(idx.second, idx.first, t);
        printf("@layer%d curt_b_data size %lu\n", get_id(), b_data.size() );
    }
    
    void set_structure_type(structure_type t) {
        type = t;
    }
    
    std::vector<tensor_t<value_type>* > get_b_data() {
        if (type == FORK) {
            assert(inputs.size()  <= 1);
            assert(outputs.size() >= 1);
            assert(b_data.size()  <= 1);
            return this->b_data;
        } else if(type == JOIN) {
            assert(inputs.size()  >= 1);
            assert(outputs.size() <= 1);
            assert(b_data.size()  >= 1);
            return this->b_data;
        }
        return std::vector<tensor_t<value_type>* >();
    }
    
    std::vector<tensor_t<value_type>* > get_inputs() {
        if (type == FORK) {
            assert(inputs.size()  <= 1);
            assert(outputs.size() >= 1);
            return this->inputs;
        } else if(type == JOIN) {
            assert(inputs.size()  >= 1);
            assert(outputs.size() <= 1);
            return this->inputs;
        }
        return std::vector<tensor_t<value_type>* >();
    }
    
    std::vector<tensor_t<value_type>* > get_outputs() {
        if (type == FORK) {
            assert(inputs.size()  <= 1);
            assert(outputs.size() >= 1);
            return this->outputs;
        } else if(type == JOIN) {
            assert(inputs.size()  >= 1);
            assert(outputs.size() <= 1);
            return this->outputs;
        }
        return std::vector<tensor_t<value_type>* >();
    }

    

};


} // superneuron namespace
#endif // _BASE_STRUCTURE_LAYER_H_
