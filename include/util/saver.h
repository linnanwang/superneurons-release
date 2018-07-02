//
// Created by ay27 on 17/4/19.
//

#ifndef SUPERNEURONS_SAVER_H
#define SUPERNEURONS_SAVER_H

#include <registry.h>
#include <cstdlib>
#include <fstream>
#include <csignal>
#include <util/error_util.h>
#include <solver.h>
#include <layer/base_layer.h>

#define SAVER_DEBUG

namespace SuperNeurons {

class network_saver {
public:
    virtual void save()=0;
    virtual void load()=0;
};


template<typename value_type>
class saver_impl {
private:
    std::ofstream file;
    registry_t<value_type> *reg;
    const char* file_path;
    base_solver_t<value_type>* solver;

    void save_meta() {
        uint8_t size_of_type = sizeof(value_type);
        file.write((const char *) &size_of_type, 1);

        uint32_t layer_cnt = (uint32_t)(reg->get_net_layers().size());
        file.write((const char*)&layer_cnt, 4);

#ifdef SAVER_DEBUG
        printf("saver write meta: type byte %d, layer count %d\n", size_of_type, layer_cnt);
#endif
    }

    void save_solver() {
        char* buff = new char[1024*1024*10];
        int len_in_byte;
        solver->gen_description(buff, &len_in_byte);
        file.write(buff, len_in_byte);

        delete[] buff;
    }

    void save_network() {
        char* buff = (char*) (malloc(1024*1024*1024));   // 1GB, i think that's enough
        size_t len_in_byte;
        for (auto layer=reg->get_net_layers().begin(); layer!=reg->get_net_layers().end(); ++layer) {
            int layer_id = layer->first;
            base_layer_t<value_type>* l = (base_layer_t<value_type>*)(layer->second);

            l->gen_description(buff, &len_in_byte);

            file.write(buff, len_in_byte);
        }

        free(buff);
    }

public:
    saver_impl(const char *file_path, registry_t<value_type>* reg, base_solver_t<value_type>* solver)
        : file_path(file_path), reg(reg), solver(solver) {

    }

    void save() {
        file.open(file_path, std::ios::out | std::ios::binary);
        if (file.fail()) {
            fprintf(stderr, "saver_impl: can not open file %s to write\n", file_path);
            exit(-1);
        }

        save_meta();
        save_solver();
        save_network();

        file.close();
    }

};

template<typename value_type>
class network_saver_impl : public network_saver {
private:

    void write_tensor(int32_t layer_id, tensor_t<value_type> *t, std::ofstream *out);
    void read_tensor(std::ifstream *in,
                     int32_t *layer_id,
                     size_t *N, size_t *C, size_t *H, size_t *W,
                     value_type **data);

    registry_t<value_type> *reg;
    const char *file_path;
public:
    network_saver_impl(const char *file_path, registry_t<value_type> *reg)
            : file_path(file_path), reg(reg) {
    }

    void save();
    void load();
};

void install_signal_processor(network_saver *saver);

}

#endif //SUPERNEURONS_SAVER_H
