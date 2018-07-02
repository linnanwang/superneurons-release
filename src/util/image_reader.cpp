//
// Created by ay27 on 7/24/17.
//

#include <util/image_reader.h>

namespace SuperNeurons {

template<class value_type>
void image_reader_t<value_type>::get_batch_no_shuffle(value_type *p_data, value_type *p_label) {
    img_reader->get_batch(img_tmp);
    // should be noticed that the third argument in expand_byte is the count of data point, not length of data in-byte.
    expand_byte<image_t, value_type>(img_tmp, p_data,
                                     img_reader->get_batch_size() *
                                     img_reader->getC() * img_reader->getH() * img_reader->getW());

    label_reader->get_batch(label_tmp);
    expand_byte<label_t, value_type>(label_tmp, p_label, label_reader->get_batch_size());
}

INSTANTIATE_CLASS(image_reader_t);

} // namespace SuperNeurons