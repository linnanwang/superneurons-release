#if !defined(_BASE_NETWORK_LAYER_H_)
#define _BASE_NETWORK_LAYER_H_
#include <tensor.h>
#include <registry.h>
#include <util/common.h>
#include <layer/base_layer.h>
#include <util/superneurons_math.h>

namespace SuperNeurons{

    
template <class value_type>
class base_network_layer_t:base_layer_t<value_type>
{
private:
    //forward output tensor, each layer only holds the output tensor
    /*----------data---------*/
    tensor_t<value_type>* f_out              = NULL;
    tensor_t<value_type>* b_data             = NULL;
    /*-------parameters------*/
    tensor_t<value_type>* weight             = NULL;
    tensor_t<value_type>* bias               = NULL;
    /*---solver variables---*/
    tensor_t<value_type>* weight_grad        = NULL;
    tensor_t<value_type>* bias_grad          = NULL;
    tensor_t<value_type>* weight_prev        = NULL;
    tensor_t<value_type>* bias_prev          = NULL;
    
    bool use_bias;
    
public:
    base_network_layer_t(LAYER lt):use_bias(false), base_layer_t<value_type>(lt) {
        
    }
    
    ~base_network_layer_t() {
    }
    
    int get_input_layer_id() {
        //single in
        std::vector<base_layer_t<value_type>*> inputs_l = this->get_prev();
        if (inputs_l.size() == 0) {
            return 0;
        } else {
            assert(inputs_l.size() == 1);
            base_layer_t<value_type>* input_l = inputs_l[0];
            int prev_id = input_l->get_base_id();
            return prev_id;
        }
    }
    
    int get_output_layer_id() {
        //single out
        std::vector<base_layer_t<value_type>*> outputs_l = this->get_next();
        if (outputs_l.size() == 0) {
            return 0;
        } else {
            assert(outputs_l.size() == 1);
            base_layer_t<value_type>* output_l = outputs_l[0];
            int prev_id = output_l->get_base_id();
            return prev_id;
        }
    }

    
    //needs setup before forward and backward
    virtual std::vector<value_type> forward (network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg) = 0;
    virtual void backward(network_stage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h, registry_t<value_type> *reg) = 0;
    virtual void forward_setup(registry_t<value_type>* reg = NULL, cudnnHandle_t* cudnn_h = NULL) = 0;
    virtual void backward_setup(registry_t<value_type>* reg = NULL, cudnnHandle_t* cudnn_h = NULL) = 0;
    virtual void gen_description(char* buff, size_t* len_in_byte) = 0;

    void gen_meta_description(char* buff, size_t* len_in_byte) {
        this->_gen_meta_description(buff, len_in_byte);
    }
    
    void update( cublasHandle_t *cublas_h, size_t iter, base_solver_t<value_type>* solver) {
        if(this->weight == NULL) return;
        solver->update(cublas_h, iter,
                       get_weight(), get_bias(),
                       get_weight_grad(), get_bias_grad(),
                       get_weight_prev(), get_bias_prev(), use_bias);
    }
    
    //GET
    inline bool is_bias_enable() {
        return this->use_bias;
    }
    
    inline int get_id() {
        return this->get_base_id();
    }
    
    inline tensor_t<value_type>* get_b_data() {
        return this->b_data;
    }
    
    inline tensor_t<value_type>* get_f_out() {
        return this->f_out;
    }
    
    inline tensor_t<value_type>* get_bias() {
        return this->bias;
    }
    
    inline tensor_t<value_type>* get_weight() {
        return this->weight;
    }
    
    inline tensor_t<value_type>* get_bias_grad() {
        return this->bias_grad;
    }
    
    inline tensor_t<value_type>* get_weight_grad() {
        return this->weight_grad;
    }
    
    inline tensor_t<value_type>* get_weight_prev() {
        return this->weight_prev;
    }
    
    inline tensor_t<value_type>* get_bias_prev() {
        return this->bias_prev;
    }
    
    //SET
    inline void enable_bias(bool s) {
        this->use_bias = s;
    }
    
    inline void set_f_out(tensor_t<value_type>* t, registry_t<value_type>* reg) {
       int cur_layer_id = this->get_id();
       int dst_layer_id = get_output_layer_id();
       reg->register_output(cur_layer_id, dst_layer_id, t);
       this->f_out = t;
       assert(this->get_f_out() == reg->get_reg_output(cur_layer_id, dst_layer_id) );
    }
    
    inline void set_b_data(tensor_t<value_type>* t, registry_t<value_type>* reg) {
        int cur_layer_id = this->get_id();
        int dst_layer_id = get_input_layer_id();
        reg->register_b_data(cur_layer_id, dst_layer_id, t);
        this->b_data = t;
        assert(this->get_b_data() == reg->get_reg_b_data(cur_layer_id, dst_layer_id) );
    }
    
    inline void set_bias(tensor_t<value_type>* t, registry_t<value_type>* reg) {
        this->bias = t;
        int cur_layer_id = this->get_id();
        reg->register_bias(cur_layer_id, t);
        assert(this->get_bias() == reg->get_reg_bias(cur_layer_id) );
    }
    
    inline void set_bias_grad(tensor_t<value_type>* t, registry_t<value_type>* reg) {
        int cur_layer_id = this->get_id();
        reg->register_bias_grad(cur_layer_id, t);
        this->bias_grad = t;
        assert(this->get_bias() == reg->get_reg_bias(cur_layer_id) );
    }

    inline void set_weight(tensor_t<value_type>* t, registry_t<value_type>* reg) {
        int cur_layer_id = this->get_id();
        reg->register_weight(cur_layer_id, t);
        this->weight = t;
        assert(this->get_weight() == reg->get_reg_weight(cur_layer_id) );
    }
    
    inline void set_weight_grad(tensor_t<value_type>* t, registry_t<value_type>* reg) {
        int cur_layer_id = this->get_id();
        reg->register_weight_grad(cur_layer_id, t);
        this->weight_grad = t;
        assert(this->get_weight_grad() == reg->get_reg_weight_grad(cur_layer_id) );
    }

    inline void set_weight_prev(tensor_t<value_type>* t, registry_t<value_type>* reg) {
        int cur_layer_id = this->get_id();
        reg->register_weight_prev(cur_layer_id, t);
        this->weight_prev = t;
        assert(this->get_weight_prev() == reg->get_reg_weight_prev(cur_layer_id) );
    }
    
    inline void set_bias_prev(tensor_t<value_type>* t, registry_t<value_type>* reg) {
        int cur_layer_id = this->get_id();
        reg->register_bias_prev(cur_layer_id, t);
        this->bias_prev = t;
        assert(this->get_bias_prev() == reg->get_reg_bias_prev(cur_layer_id) );
    }
    
    
};

} // superneuron namespace
#endif // _BASE_NETWORK_LAYER_H_
