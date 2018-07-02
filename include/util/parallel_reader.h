//
// Created by ay27 on 7/20/17.
//

#ifndef SUPERNEURONS_PARALLEL_READER_H
#define SUPERNEURONS_PARALLEL_READER_H

#include <util/common.h>
#include <util/preprocess.h>
#include <tensor.h>
#include <util/thread_routine.h>
#include <util/tensor_queue.h>
#include <util/image_reader.h>

namespace SuperNeurons {


template<class value_type>
class parallel_reader_t : thread_routine_t {
private:
    /** tensor queue */
    tensor_queue_t<value_type> *q;

    tensor_t<value_type> *data=NULL, *label=NULL;
    bool                 is_first_time_to_get_batch = true;

    size_t N, srcC, srcH, srcW, dstC, dstH, dstW;
    preprocessor<value_type> *processor;

    /** lock when outer thread can not fetch gpu tensor */
    std::mutex wait_data_m;
    std::mutex pre_m;

    const char *data_path, *label_path;

    /** the runtime shuffle function does not finish */
    const bool shuffle, in_memory;

    inline void sleep_a_while(size_t ms = 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    void thread_entry(size_t thread_idx, size_t total_threads) override;

public:
    parallel_reader_t(const char *data_path, const char *label_path,
                      size_t thread_cnt, size_t batch_size, size_t dstC, size_t dstH, size_t dstW,
                      preprocessor<value_type> *processor = NULL,
                      size_t _max_cpu_batch_cnt = 4, size_t _max_gpu_batch_cnt = 2,
                      bool in_memory = false,
                      bool shuffle = false)
            : thread_routine_t(thread_cnt), processor(processor),
              data_path(data_path), dstC(dstC), dstH(dstH), dstW(dstW), label_path(label_path),
              in_memory(in_memory), shuffle(shuffle) {

        // we new a base_reader to get the NCHW info
        // the concrete image reader will be created in each thread
        base_reader_t<image_t> *reader = new base_reader_t<image_t>(data_path, batch_size);
        N = reader->get_batch_size();
        srcC = reader->getC();
        srcH = reader->getH();
        srcW = reader->getW();

        if (N*srcC*srcH*srcW != processor->input_size()) {
            fprintf(stderr, "input size do not match! data_reader (%zu), preprocessor (%zu)\n",
                    N*srcC*srcH*srcW, processor->input_size());
            exit(-1);
        }

        if (N*dstC*dstH*dstW != processor->output_size()) {
            fprintf(stderr, "output size do not match! data_reader (%zu), preprocessor (%zu)\n",
                    N*dstC*dstH*dstW, processor->output_size());
            exit(-1);
        }

        delete reader;

        q = new tensor_queue_t<value_type>(N, dstC, dstH, dstW, _max_cpu_batch_cnt, _max_gpu_batch_cnt);

        printf("parallel reader: src(%zu, %zu, %zu, %zu), dst(%zu, %zu, %zu, %zu)\n",
               N, srcC, srcH, srcW, N, dstC, dstH, dstW);

        pre_m.unlock();
        // start threads
        this->start();
    }

    ~parallel_reader_t() {
        this->stop();
        delete q;
    }

    void get_batch(tensor_t<value_type> *_data, tensor_t<value_type> *_label);
//
//    inline std::pair<value_type*, value_type* > get_ptrs() {
//        if (data == NULL || label == NULL) {
//            while ( !q->fetch_gpu_tensor(&data, &label) ) {
//                sleep_a_while();
//            }
//            tensor_lock.lock();
//        }
//        return std::make_pair(data->get_gpu_ptr(), label->get_gpu_ptr());
//    };

    inline size_t getN() const {
        return N;
    }

    inline size_t getC() const {
        return dstC;
    }

    inline size_t getH() const {
        return dstH;
    }

    inline size_t getW() const {
        return dstW;
    }

};


} // namespace SuperNeurons

#endif //SUPERNEURONS_PARALLEL_READER_H
