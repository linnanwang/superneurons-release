//
// Created by ay27 on 7/20/17.
//

#ifndef SUPERNEURONS_BASE_READER_H
#define SUPERNEURONS_BASE_READER_H

#include <util/common.h>
#include <fstream>
#include <util/error_util.h>

namespace SuperNeurons {

/** the begin of binary file, which is NCHW, each size in byte of which is 8. */
const int MAGIC_LEN = 32;

/** for every NCHW, its type is meta_t */
typedef uint64_t meta_t;
const size_t META_ATOM = sizeof(meta_t);

/** for the image binary file, every data point stored in image_t */
typedef uint8_t image_t;

/** for the label binary file, every label stored in label_t */
typedef int32_t label_t;

/** a helper marco to convert the binary code to meta_t */
#define bin_to_uint(arr) (*((meta_t*)(arr)))

/** helper function to convert data type */
template<typename src_type, typename dst_type>
void expand_byte(const src_type *src, dst_type *dst, unsigned long len) {
    for (unsigned long i = 0; i < len; ++i) {
        dst[i] = (dst_type) src[i];
    }
}


template<class value_type>
class base_reader_t {
protected:

    void read_meta();

    size_t total_data_cnt, C, H, W;
    size_t batch_size;
    std::ifstream src_file;

public:
    base_reader_t(const char *data_path, size_t batch_size) : batch_size(batch_size) {
        src_file.open(data_path, std::ios::in | std::ios::binary);
        if (src_file.fail()) {
            src_file.close();
            fprintf(stderr, "read file %s failed!\n", data_path);
            exit(-1);
        }
        read_meta();
    }

    virtual ~base_reader_t() {
        if (src_file.is_open()) {
            src_file.close();
        }
    }

    virtual void read_data(value_type *data, size_t start_idx, size_t len_in_byte, bool reseek);

    virtual /**
     * change reading position
     * @param pos
     */
    inline void seek(size_t pos) {
        src_file.clear();
        src_file.seekg(pos, std::ios::beg);
    }

    /**
     * get batch data without shuffle
     * @param data
     */
    virtual inline void get_batch(value_type *data) {
        read_data(data, 0, get_batch_size_in_byte(), false);
    }

    inline std::ifstream &get_file() {
        return src_file;
    }

    inline size_t get_batch_size() const {
        return batch_size;
    }

    inline size_t get_batch_size_in_byte() {
        return batch_size * C * H * W * sizeof(value_type);
    }

    inline size_t get_total_data_cnt() {
        return total_data_cnt;
    }

    inline size_t getC() const {
        return C;
    }

    inline size_t getH() const {
        return H;
    }

    inline size_t getW() const {
        return W;
    }
};

template<class value_type>
class memory_reader_t : public base_reader_t<value_type> {
private:
    /** read all data into memory */
    value_type* raw_data;
    size_t pos;
    size_t total_scalar;
public:
    memory_reader_t(const char *data_path, size_t batch_size) : base_reader_t<value_type>(data_path, batch_size) {
        total_scalar = this->get_total_data_cnt() * this->getC() * this->getH() * this->getW();

        printf("total scalar = %zu\n", total_scalar);

        // heavy data read !!
        raw_data = (value_type*) malloc(total_scalar * sizeof(value_type));
        if (raw_data == NULL) {
            FatalError("can not malloc enough memory in memory reader!!");
        }
        this->get_file().read((char*)raw_data, total_scalar * sizeof(value_type));

        pos = 0;
    }

    ~memory_reader_t() {
        free(raw_data);
    }

    inline void get_batch(value_type *data) {
        size_t batch_scalar = this->get_batch_size()*this->getC()*this->getH()*this->getW();
        if (pos + batch_scalar > total_scalar) {
//            printf("roll back, pos=%zu, batch_size=%zu, total_scalar=%zu\n", pos, this->get_batch_size(), total_scalar);
            memcpy(data, raw_data + pos, (total_scalar - pos) * sizeof(value_type));
            memcpy(data+(total_scalar - pos), raw_data,
                   (batch_scalar - (total_scalar - pos)) * sizeof(value_type));
            pos = batch_scalar - (total_scalar - pos);
//            printf("roll back success, pos=%zu\n", pos);
//            for (int i = 0; i < 10; ++i) {
//                printf("%d ", data[i]);
//            }
//            for (int i = 0; i < 10; ++i) {
//                printf("%d ", data[batch_scalar - i-1]);
//            }
//            printf("\n");
        } else {
            memcpy(data, raw_data + pos, this->get_batch_size_in_byte());
            pos += batch_scalar;
        }
    }

    inline void seek(size_t pos) {
        this->pos = (size_t)((pos-MAGIC_LEN) / sizeof(value_type));
    }

};

} // namespace SuperNeurons

#endif //SUPERNEURONS_BASE_READER_H
