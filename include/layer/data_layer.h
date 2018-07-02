//
// Created by ay27 on 17/3/31.
//

#ifndef SUPERNEURONS_DATA_LAYER_H
#define SUPERNEURONS_DATA_LAYER_H

#include <tensor.h>
#include <registry.h>
#include <util/parallel_reader.h>
#include <layer/base_network_layer.h>

namespace SuperNeurons {

    template<class value_type>
    class data_layer_t : public base_network_layer_t<value_type> {
    private:
        size_t N, C, H, W;
        const data_mode mode;
        parallel_reader_t<value_type> *reader;

    public:
        data_layer_t(data_mode m,
                     parallel_reader_t<value_type> *_reader) :
                reader(_reader), mode(m), base_network_layer_t<value_type>(DATA_L) {
            this->N = reader->getN();
            this->C = reader->getC();
            this->H = reader->getH();
            this->W = reader->getW();
        }

        ~data_layer_t() {
            delete reader;
        }

        size_t get_batch_size() {
            return this->N;
        }

        void forward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h);

        void backward_setup(registry_t<value_type> *reg, cudnnHandle_t *cudnn_h) {
            // this is the first layer in network, so do nothing here.
        }

        std::vector<value_type> forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg);

        void backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg) {}

        void gen_description(char* buff, size_t* len_in_byte) {
            this->gen_meta_description(buff, len_in_byte);
        }
    };

}

#endif //SUPERNEURONS_DATA_LAYER_H
