//
// Created by ay27 on 17/3/29.
//

#ifndef SUPERNEURONS_DATA1_READER_H
#define SUPERNEURONS_DATA1_READER_H

#include <util/base_reader.h>
#include <util/preprocess.h>
#include <algorithm>

namespace SuperNeurons {


/**
 * DO NOT USE IT DIRECTLY!!!!
 * please use the parallel_reader_t.
 * @tparam value_type
 */
template<class value_type>
class image_reader_t {
private:

    void shuffle_every_epoch() {
        unsigned int seed;
        seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(shuffled_idxs.begin(), shuffled_idxs.end(), std::default_random_engine(seed));
    }


    base_reader_t<image_t> *img_reader;
    base_reader_t<label_t> *label_reader;

    image_t *img_tmp;
    label_t *label_tmp;

    bool shuffle, in_memory;
    std::vector<size_t> shuffled_idxs;

    void get_batch_with_shuffle(value_type *p_data, value_type *p_label) {
        FatalError("get batch with shuffle does not prepared yet!!\n");
    }

    void get_batch_no_shuffle(value_type *p_data, value_type *p_label);

public:
    image_reader_t(const char *data_path, const char *label_path, size_t batch_size, bool _in_memory, bool _shuffle)
            : shuffle(_shuffle), in_memory(_in_memory) {
        if (_in_memory) {
            img_reader = new memory_reader_t<image_t>(data_path, batch_size);
            label_reader = new memory_reader_t<label_t>(label_path, batch_size);
        } else {
            img_reader = new base_reader_t<image_t>(data_path, batch_size);
            label_reader = new base_reader_t<label_t>(label_path, batch_size);
        }

        for (size_t i = 0; i < img_reader->get_total_data_cnt(); ++i) {
            shuffled_idxs.push_back(i);
        }
        checkCudaErrors(cudaMallocHost((void**)&img_tmp, batch_size * img_reader->getC() * img_reader->getH() * img_reader->getW() *
                sizeof(image_t)));
        checkCudaErrors(cudaMallocHost((void**)&label_tmp, batch_size * sizeof(label_t)));
    }

    ~image_reader_t() {
        delete img_reader;
        delete label_reader;
        checkCudaErrors(cudaFreeHost(img_tmp));
        checkCudaErrors(cudaFreeHost(label_tmp));
    }

    inline void seek(size_t item_idx) {
        assert(item_idx < img_reader->get_total_data_cnt());
        size_t c = img_reader->getC(), h = img_reader->getH(), w = img_reader->getW();
        img_reader->seek(MAGIC_LEN + item_idx * (c*h*w) * sizeof(image_t));
        label_reader->seek(MAGIC_LEN + item_idx * sizeof(label_t));
    }

    inline size_t get_total_data_cnt() {
        return img_reader->get_total_data_cnt();
    }

    inline size_t get_data_batch_size_in_byte() {
        return img_reader->get_batch_size_in_byte();
    }

    inline size_t get_label_batch_size_in_byte() {
        return label_reader->get_batch_size_in_byte();
    }

    inline size_t get_batch_size() const {
        return img_reader->get_batch_size();
    }

    inline size_t getC() const {
        return img_reader->getC();
    }

    inline size_t getH() const {
        return img_reader->getH();
    }

    inline size_t getW() const {
        return img_reader->getW();
    }

    inline void get_batch(value_type *p_data, value_type *p_label) {
        if (shuffle) {
            this->get_batch_with_shuffle(p_data, p_label);
        } else {
            this->get_batch_no_shuffle(p_data, p_label);
        }
    }
};


} // namespace SuperNeurons

#endif //SUPERNEURONS_DATA_READER_H
