//
// Created by ay27 on 7/24/17.
//

#include <util/base_reader.h>
#include <util/error_util.h>

namespace SuperNeurons {

/**
 * read meta data, following to the NCHW format.
 */
template <class value_type>
void base_reader_t<value_type>::read_meta() {
    char meta[MAGIC_LEN];

    src_file.clear();

    this->src_file.read(meta, MAGIC_LEN);
    this->total_data_cnt = bin_to_uint(meta);
    this->C = bin_to_uint(meta + META_ATOM);
    this->H = bin_to_uint(meta + META_ATOM * 2);
    this->W = bin_to_uint(meta + META_ATOM * 3);

    printf("read from file, N=%zu, C=%zu, H=%zu, W=%zu\n", total_data_cnt, C, H, W);
}



/**
 * read data according to the given start index and length-in-byte
 * @param data : data pointer
 * @param start_idx : start index of data point. for image, it's the start_idx-th image
 * @param len_in_byte : read length in byte
 * @param reseek : re-seek reading position according to the giving start_idx
 *                 if set as false, the start_idx will be ignored
 */

template <class value_type>
void base_reader_t<value_type>::read_data(value_type *data, size_t start_idx, size_t len_in_byte, bool reseek) {
    if (reseek) {
        seek(MAGIC_LEN + start_idx*(this->getC()*this->getH()*this->getW()* sizeof(value_type)));
    }

    std::ifstream& src_file = get_file();

    src_file.read((char *) data, len_in_byte);

    // if read to the end, go back to very first.
    size_t readed_size = (size_t) src_file.gcount();
    if (readed_size < len_in_byte) {
        seek(MAGIC_LEN);
        src_file.read((char *) (data + readed_size), len_in_byte - readed_size);
        if ((size_t )src_file.gcount() < len_in_byte - readed_size) {
            FatalError("read size must smaller than the whole data size.\n");
        }
//
//        for (int i = 0; i < 10; ++i) {
//            printf("%d ", data[i]);
//        }
//        for (int i = 0; i < 10; ++i) {
//            printf("%d ", data[this->batch_size*this->C*this->H*this->W - i-1]);
//        }
//        printf("\n");
    }
}


template class base_reader_t<image_t >;
template class base_reader_t<label_t >;

template class memory_reader_t<image_t >;
template class memory_reader_t<label_t >;

} // namespace SuperNeurons