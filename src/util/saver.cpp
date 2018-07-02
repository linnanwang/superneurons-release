//
// Created by ay27 on 17/4/19.
//

#include <util/saver.h>
#include <cstdio>
#include <fstream>
#include <csignal>
#include <cinttypes>

namespace SuperNeurons {

#define bin_to_uint64_t(arr) (*((uint64_t*)(arr)))
#define bin_to_int32_t(arr)  (*((int32_t*)(arr)))

template<typename value_type>
void network_saver_impl<value_type>::write_tensor(int32_t layer_id, tensor_t<value_type> *t,
                                                  std::ofstream *out) {
    char meta[36];
    memset(meta, 0, 36);
    uint64_t n = t->get_N(), c = t->get_C(), h = t->get_H(), w = t->get_W();

#ifdef DEBUG
    printf("in write, layerid:%d, n:%" PRIu64 ", c:%" PRIu64 ", h:%" PRIu64 ", w:%" PRIu64 "\n", layer_id, n, c, h, w);
#endif

    memcpy(meta, &layer_id, 4);
    memcpy(meta + 4, &n, 8);
    memcpy(meta + 12, &c, 8);
    memcpy(meta + 20, &h, 8);
    memcpy(meta + 28, &w, 8);
    out->write(meta, 36);

    // fetch data from gpu!!!
    t->GPUtoCPU();

    size_t total_size = sizeof(value_type) * n * c * h * w;
    out->write((const char *) t->get_cpu_ptr(), total_size);
    out->flush();
}

template<typename value_type>
void network_saver_impl<value_type>::read_tensor(std::ifstream *in,
                                                 int32_t *layer_id,
                                                 size_t *N, size_t *C, size_t *H, size_t *W,
                                                 value_type **data) {
    char meta[36];
    in->read(meta, 36);
    *layer_id = bin_to_int32_t(meta);
    *N = (size_t) bin_to_uint64_t(meta + 4);
    *C = (size_t) bin_to_uint64_t(meta + 12);
    *H = (size_t) bin_to_uint64_t(meta + 20);
    *W = (size_t) bin_to_uint64_t(meta + 28);

#ifdef DEBUG
    printf("in read, layerid:%d, n:%zu, c:%zu, h:%zu w:%zu", *layer_id, *N, *C, *H, *W);
#endif

    // here do not use tensor_t and registry, the data array will be created here,
    // then replace the target tensor.
    *data = new value_type[(*N) * (*C) * (*H) * (*W)];

    size_t total_size = sizeof(value_type) * (*N) * (*C) * (*H) * (*W);
    in->read((char *) (*data), total_size);
}


template<typename value_type>
void network_saver_impl<value_type>::save() {



    std::map<int, tensor_t<value_type> *> *weight = this->reg->get_all_weight();
    std::map<int, tensor_t<value_type> *> *bias = this->reg->get_all_bias();

    // write to file
    std::ofstream out(file_path, std::ios::out | std::ios::binary);
    if (out.fail()) {
        fprintf(stderr, "can not open file %s to write\n", file_path);
        exit(-1);
    }

    uint8_t size_of_type = sizeof(value_type);
    uint64_t length1 = weight->size(), length2 = bias->size();

    out.write((const char *) &size_of_type, 1);
    out.write((const char *) &length1, 8);
    out.write((const char *) &length2, 8);

#ifdef DEBUG
    printf("in save, size of type: %d, leng1: %" PRIu64 ", leng2: %" PRIu64 "\n", size_of_type, length1, length2);
#endif

    for (typename std::map<int, tensor_t<value_type> *>::iterator it = weight->begin(); it != weight->end(); ++it) {
        write_tensor(it->first, it->second, &out);
    }
    for (typename std::map<int, tensor_t<value_type> *>::iterator it = bias->begin(); it != bias->end(); ++it) {
        write_tensor(it->first, it->second, &out);
    }
    out.close();

    printf("save network to checkpoint file finish\n");
}

template<typename value_type>
void network_saver_impl<value_type>::load() {

    std::ifstream in(file_path, std::ios::in | std::ios::binary);
    if (in.fail()) {
        printf("no checkpoint file to load\n");
        return;
    }

    char meta[17];
    in.read(meta, 17);

    uint8_t size_of_type = (uint8_t) meta[0];
    uint64_t weight_length = *((uint64_t *) (meta + 1)),
            bias_length = *((uint64_t *) (meta + 9));

#ifdef DEBUG
    printf("in save, size of type: %d, leng1: %" PRIu64 ", leng2: %" PRIu64 "\n", size_of_type, weight_length, bias_length);
#endif

    assert(size_of_type == sizeof(value_type));

    std::map<int, tensor_t<value_type> *> *weight = this->reg->get_all_weight();
    std::map<int, tensor_t<value_type> *> *bias = this->reg->get_all_bias();

    // Now, without the track of registry, we can safely replace the cpu ptr.
    for (uint64_t i = 0; i < weight_length; ++i) {
        int32_t layer_id;
        size_t N, C, H, W;
        value_type *data;
        read_tensor(&in, &layer_id, &N, &C, &H, &W, &data);
        typename std::map<int, tensor_t<value_type> *>::iterator it = weight->find(layer_id);
        if (it != weight->end()) {
            if ((N == it->second->get_N()) &&
                (C == it->second->get_C()) &&
                (H == it->second->get_H()) &&
                (W == it->second->get_W())) {
                it->second->replace_data(data);
//                memcpy(it->second->get_cpu_ptr(), data,
//                       sizeof(value_type) * N * C * H * W);
//                it->second->CPUtoGPU();
            } else {
                fprintf(stderr, "the checkpoint file does not match current network\n");
                exit(-1);
            }
        } else {
            fprintf(stderr, "current network structure does not found layerid %d\n", layer_id);
            exit(-1);
//            weight->insert(std::make_pair(layer_id, t));
        }
    }
    for (uint64_t i = 0; i < bias_length; ++i) {
        int32_t layer_id;
        size_t N, C, H, W;
        value_type *data;
        read_tensor(&in, &layer_id, &N, &C, &H, &W, &data);
        typename std::map<int, tensor_t<value_type> *>::iterator it = bias->find(layer_id);
        if (it != bias->end()) {
            if ((N == it->second->get_N()) &&
                (C == it->second->get_C()) &&
                (H == it->second->get_H()) &&
                (W == it->second->get_W())) {
                it->second->replace_data(data);
//                memcpy(it->second->get_cpu_ptr(), data,
//                       sizeof(value_type) * N * C * H * W);
//                it->second->CPUtoGPU();
            } else {
                fprintf(stderr, "the checkpoint file does not match current network\n");
                exit(-1);
            }
        } else {
            fprintf(stderr, "current network structure does not found layerid %d\n", layer_id);
            exit(-1);
//            bias->insert(std::make_pair(layer_id, t));
        }
    }

#ifdef DEBUG
    printf("in save, size of type: %d, leng1: %lu, leng2: %lu\n",
           size_of_type, weight->size(), bias->size());
#endif

    in.close();

    printf("read from checkpoint file finish\n");
}


static network_saver *saver_instance = NULL;

void install_signal_processor(network_saver *saver) {
    saver_instance = saver;
    std::signal(SIGINT, [](int sig) {

        if (saver_instance == NULL) {
#ifdef DEBUG
            printf("no network saver, exit\n");
#endif
            return;
        }

        saver_instance->save();
        printf("save network success\n");
        exit(sig);
    });
}


INSTANTIATE_CLASS(network_saver_impl);

}