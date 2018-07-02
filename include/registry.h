#if !defined(_REGISTRY_H_)
#define _REGISTRY_H_
#include <map>
#include <vector>
#include <string>
#include <tensor.h>
#include <util/common.h>

namespace SuperNeurons{

template <class value_type>
class registry_t
{
private:
    //by convention, the NULL layer is 0
    std::vector<tensor_t<value_type>* > tensors_to_free;
    //d_key_t: [src_layer, dst_layer] currnt layer outputs to next layer(+1), forward
    std::map<d_key_t, tensor_t<value_type>* > outputs; //the outputs of a layer
    //d_key_t: [src_layer, dst_layer] currnt layer outputs to prev layer(-1), backward
    std::map<d_key_t, tensor_t<value_type>* > b_data;  //backward data

    std::map<int, tensor_t<value_type>* > bias;
    std::map<int, tensor_t<value_type>* > weight;
    std::map<int, tensor_t<value_type>* > bias_grad;
    std::map<int, tensor_t<value_type>* > weight_grad;

    std::map<int, tensor_t<value_type>* > bias_prev;
    std::map<int, tensor_t<value_type>* > weight_prev;
    //this tracks the dependency tensors in the forward and backward flow of a network
    std::map<int, std::vector<tensor_t<value_type>* > > forward_dependency;
    std::map<tensor_t<value_type>*, std::vector<int>  > forward_dependency_by_tensor;
    std::map<int, std::vector<tensor_t<value_type>* > > backward_dependency;
    std::map<tensor_t<value_type>*, std::vector<int>  > backward_dependency_by_tensor;
    
    cublasHandle_t  cublas_handle;
    
    //this tracks the computation route of a unrolled DAG/network
    std::vector<std::pair<int, net_comp> > net_comp_route;
    std::vector<std::pair<int, net_comp> > net_test_route;
    std::map<int, void* >                  net_layers;
    std::map<int, std::vector< tensor_t<value_type>* > > tensor_by_layer;


    //train data shall be registered as the first layer's output
    tensor_t<value_type>* test_data;
    tensor_t<value_type>* train_data;
    tensor_t<value_type>* test_label;
    tensor_t<value_type>* train_label;

    bool is_included(std::vector<tensor_t<value_type>* > &v, tensor_t<value_type>* t);

    void print_registry(std::map<d_key_t, tensor_t<value_type>* > m, const char* str) {
        for(auto const &ent : m) {
            printf("%s layer[from %d to %d] %p\n", str, ent.first.first, ent.first.second, ent.second);
        }
    }

    void print_registry(std::map<int, tensor_t<value_type>* > m, const char* str) {
        for(auto const &ent : m) {
            printf("%s layer[%d] %p\n", str, ent.first, ent.second);
        }
    }
    
    void print_dependency_by_tensors(std::map<tensor_t<value_type>*, std::vector<int> > &m, net_comp dir);
    
    void print_dependency(std::map<int, std::vector< tensor_t<value_type>* > > &m, net_comp dir);

    void register_layer_param(int layer_id, tensor_t<value_type>* t,  std::map<int, tensor_t<value_type>* >& m, const char* str);

    tensor_t<value_type>* get_layer_param(int layer_id, std::map<int, tensor_t<value_type>* >& m);
    
    void print_net_route(std::vector<std::pair<int, net_comp> >& route) {
        printf("computation route of this network:\n");
        for(size_t i = 0; i < route.size(); i++) {
            if (route[i].second == FORWARD) {
                printf("(%d, forward)->", route[i].first);
            } else {
                printf("(%d, backward)->", route[i].first);
            }
        }
        printf("\n");
    }


public:
    
    
    std::vector<tensor_t<value_type>* >* get_forward_dependency(int layer_id) {
        typename std::map<int, std::vector<tensor_t<value_type>* > >::iterator it = forward_dependency.find(layer_id);
        if (it != forward_dependency.end() ) {
            return &(it->second);
        } else {
            return NULL;
        }
    }
    
    std::vector<tensor_t<value_type>* >* get_backward_dependency(int layer_id) {
        typename std::map<int, std::vector<tensor_t<value_type>* > >::iterator it = backward_dependency.find(layer_id);
        if (it != backward_dependency.end() ) {
            return &(it->second);
        } else {
            return NULL;
        }
    }
    
    std::map<int, std::vector< tensor_t<value_type>* > >& get_tensor_by_layer() {
        return this->tensor_by_layer;
    }

    
    std::map<tensor_t<value_type>*, std::vector<int> >& get_forward_tensors() {
        return this->forward_dependency_by_tensor;
    }
    
    std::map<tensor_t<value_type>*, std::vector<int> >& get_backward_tensors() {
        return this->backward_dependency_by_tensor;
    }

    registry_t():train_data(NULL), train_label(NULL), test_data(NULL), test_label(NULL) {
        checkCublasErrors( cublasCreate(&cublas_handle) );
    }

    ~registry_t() {
        for(size_t i = 0; i < tensors_to_free.size(); i++) {
            delete tensors_to_free[i];
        }
    }
    
    void print_forward_dependency() {
        print_dependency(forward_dependency, FORWARD);
    }
    
    void print_backward_dependency() {
        print_dependency(backward_dependency, BACKWARD);
    }
    
    void print_forward_dependency_by_tensor() {
        print_dependency_by_tensors(forward_dependency_by_tensor, FORWARD);
    }
    
    void print_backward_dependency_by_tensor() {
        print_dependency_by_tensors(backward_dependency_by_tensor, BACKWARD);
    }
    
    void print_net_comp_route() {
        print_net_route(this->net_comp_route);
    }
    
    void print_net_test_route() {
        print_net_route(this->net_test_route);
    }
    
    
    void print_net_layers() {
        typename std::map<int, void* >::iterator it = net_layers.begin();
        printf("layers of this network:\n");
        for (it = net_layers.begin(); it != net_layers.end(); ++it) {
            printf("layer id:%d -> %p \n", it->first, it->second);
        }
    }
    
    void print_tensors_by_layers();

    std::map<int, tensor_t<value_type>* >* get_all_weight() {
        return &(this->weight);
    };

    std::map<int, tensor_t<value_type>* >* get_all_bias() {
        return &(this->bias);
    };

    tensor_t<value_type>* get_train_label() {
        return train_label;
    }

    tensor_t<value_type>* get_train_data() {
        return train_data;
    }

    tensor_t<value_type>* get_test_data() {
        return test_data;
    }

    tensor_t<value_type>* get_test_label() {
        return test_label;
    }

    std::vector<tensor_t<value_type>* >* get_vector() {
        return &(this->tensors_to_free);
    }
    

    void set_train_data(tensor_t<value_type>* t) {
        if (train_data != NULL) {
            printf("TRAIN DATA has been set\n");
            exit(1);
        } else {
            printf("SET TRAIN DATA\n");
            train_data = t;
            assert( get_train_data() != NULL );
        }
    }

    void set_test_data(tensor_t<value_type>* t) {
        if (test_data != NULL) {
            printf("TEST DATA has been set\n");
            exit(1);
        } else {
            printf("SET TEST DATA\n");
            test_data = t;
            assert( get_test_data() != NULL );
        }
    }

    void set_train_label(tensor_t<value_type>* t) {
        if (train_label != NULL) {
            printf("TRAIN LABEL has been set\n");
            exit(1);
        } else {
            printf("SET TRAIN LABEL@%p\n", t);
            train_label = t;
            assert( get_train_label() != NULL );
        }
    }

    void set_test_label(tensor_t<value_type>* t) {
        if (test_label != NULL) {
            printf("TEST LABEL has been set\n");
            exit(1);
        } else {
            printf("SET TEST LABEL@%p\n", t);
            test_label = t;
            assert( get_test_label() != NULL );
        }
    }
    
    void register_net_test_route(const int layer_id) {
        assert(net_test_route.size() == 0);
        net_test_route.push_back( std::make_pair( layer_id, FORWARD) );
        for( size_t i = 1; i < net_comp_route.size(); i++ ) {
            if( net_comp_route[i].second == FORWARD ) {
                net_test_route.push_back( net_comp_route[i] );
            }
        }
    }
    
    void register_net_comp_route( const int layer_id,  const net_comp &nc) {
        this->net_comp_route.push_back( std::make_pair( layer_id, nc) );
    }
    
    void register_net_layers(const int layer_id, void* b) {
        this->net_layers.insert( std::make_pair( layer_id, b) );
    }
    
    void register_tensors_by_layers();

    void register_forward_dependency(  int layer_id, tensor_t<value_type>* t);
    
    void register_backward_dependency( int layer_id, tensor_t<value_type>* t);

    void register_output( int src_layer_id, int dest_layer_id, tensor_t<value_type>* t );

    void register_b_data( int src_layer_id, int dest_layer_id, tensor_t<value_type>* t );

    void register_weight(int layer_id, tensor_t<value_type>* t) {
        assert(t->get_type() == PARAM);
        register_layer_param(layer_id, t, weight, "weight");
    }
    
    void register_bias(int layer_id, tensor_t<value_type>* t) {
        assert(t->get_type() == PARAM);
        register_layer_param(layer_id, t, bias, "bias");
    }

    void register_weight_grad(int layer_id, tensor_t<value_type>* t) {
        assert(t->get_type() == GRAD);
        register_layer_param(layer_id, t, weight_grad, "weight_grad");
    }

    void register_bias_grad(int layer_id, tensor_t<value_type>* t ) {
        assert(t->get_type() == GRAD);
        register_layer_param(layer_id, t, bias_grad, "bias_grad");
    }

    void register_weight_prev(int layer_id, tensor_t<value_type>* t ) {
        assert(t->get_type() == GRAD);
        register_layer_param(layer_id, t, weight_prev, "weight_prev");
    }

    void register_bias_prev(int layer_id, tensor_t<value_type>* t ) {
        assert(t->get_type() == GRAD);
        register_layer_param(layer_id, t, bias_prev, "bias_prev");
    }
    
    std::vector<std::pair<int, net_comp> >& get_net_test_route() {
        return this->net_test_route;
    }
    
    std::vector<std::pair<int, net_comp> >& get_net_comp_route() {
        return this->net_comp_route;
    }
    
    std::map<int, void* >& get_net_layers() {
        return this->net_layers;
    }

    tensor_t<value_type>* get_reg_bias(int layer_id) {
        return get_layer_param(layer_id, bias);
    }

    tensor_t<value_type>* get_reg_weight(int layer_id) {
        return get_layer_param(layer_id, weight);
    }

    tensor_t<value_type>* get_reg_bias_grad(int layer_id) {
        return get_layer_param(layer_id, bias_grad);
    }

    tensor_t<value_type>* get_reg_weight_grad(int layer_id) {
        return get_layer_param(layer_id, weight_grad);
    }

    tensor_t<value_type>* get_reg_weight_prev(int layer_id) {
        return get_layer_param(layer_id, weight_prev);
    }

    tensor_t<value_type>* get_reg_bias_prev(int layer_id) {
        return get_layer_param(layer_id, bias_prev);
    }

    //by convention source layer holds the tensor
    tensor_t<value_type>* get_reg_output(int source_layer_id, int dest_layer_id);

    tensor_t<value_type>* get_reg_b_data(int source_layer_id, int dest_layer_id);

    value_type get_grad_sqrsum();
};

}//SuperNeurons namespace

#endif

