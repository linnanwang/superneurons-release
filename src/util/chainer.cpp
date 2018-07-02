//
// Created by ay27 on 7/12/17.
//

#include <util/chainer.h>

namespace SuperNeurons {

template<class value_type>
chainer<value_type> *chainer<value_type>::push_layer(std::initializer_list<base_layer_t<value_type> *> layers) {
    for (typename std::initializer_list<base_layer_t<value_type> *>::iterator it = layers.begin();
         it != layers.end(); ++it) {
        end->hook(*it);
        end = *it;
    }

    return this;
}

template<class value_type>
chainer<value_type> *chainer<value_type>::fork_layer(base_structure_t<value_type> *fork_layer,
                                                     std::initializer_list<std::initializer_list<base_layer_t<value_type> *>> branches,
                                                     base_structure_t<value_type> *join_layer) {
    end->hook((base_layer_t<value_type> *) fork_layer);

    for (typename std::initializer_list<std::initializer_list<base_layer_t<value_type> *> >::iterator br = branches.begin();
         br != branches.end(); ++br) {
        end = (base_layer_t<value_type> *) fork_layer;
        push_layer(*br);
        end->hook((base_layer_t<value_type> *) join_layer);
    }

    end = (base_layer_t<value_type> *) join_layer;

    return this;
}

INSTANTIATE_CLASS(chainer);

} //namespace SuperNeurons