//
// Created by ay27 on 7/12/17.
//

#ifndef SUPERNEURONS_CHAINER_H
#define SUPERNEURONS_CHAINER_H

#include <layer/base_network_layer.h>
#include <vector>
#include <initializer_list>
#include <layer/fork_layer.h>
#include <layer/join_layer.h>

namespace SuperNeurons {


template <class value_type>
class chainer {
private:
    base_layer_t<value_type>* root;
    base_layer_t<value_type>* end;

public:
    chainer(base_layer_t<value_type>* root_layer) : root(root_layer), end(root_layer) {
    }

    ~chainer() {
    }

    chainer<value_type>* push_layer(std::initializer_list<base_layer_t<value_type>* > layers);

    chainer<value_type>* fork_layer( base_structure_t<value_type>* fork_layer,
                                     std::initializer_list< std::initializer_list<base_layer_t<value_type>* > > branches,
    base_structure_t<value_type>* join_layer);

};

} // namespace SuperNeurons

#endif //SUPERNEURONS_CHAINER_H
