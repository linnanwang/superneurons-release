//
// Created by ay27 on 7/25/17.
//

#ifndef SUPERNEURONS_ITEM_QUEUE_H
#define SUPERNEURONS_ITEM_QUEUE_H

#include <util/common.h>
#include <tensor.h>
#include <mutex>

//#define DEBUG_TENSOR_QUEUE

namespace SuperNeurons {

typedef enum stage {
    freed = 0,
    cpu = 1,
    cpu_gpu = 2
} stage_t;

template<class value_type>
class tensor_queue_t {
private:
    /** a data_reg and label_reg to track tensor in this queue */
    std::vector<tensor_t<value_type> *> reg;

    std::vector<tensor_t<value_type> *> data_reg;
    std::vector<tensor_t<value_type> *> label_reg;

    /** track every tensor's stage */
    std::vector<stage_t> stage;

    /** mutex for queue */
    std::mutex m;

    /** counter for the tensor store in GPU */
    size_t in_gpu_cnt;

    /** track the next index to process */
    size_t fetch_free_next_idx, fetch_gpu_next_idx, transfer_next_idx;

    const size_t MAX_QUEUE_SIZE, MAX_GPU_CACHE_SIZE;

public:
    tensor_queue_t(size_t N, size_t C, size_t H, size_t W, const size_t max_queue_size, const size_t max_gpu_cache_size)
            : MAX_QUEUE_SIZE(max_queue_size), MAX_GPU_CACHE_SIZE(max_gpu_cache_size), in_gpu_cnt(0) {

        for (size_t i = 0; i < MAX_QUEUE_SIZE; ++i) {
            // the tensor constructor will push itself to the reg vector
            tensor_t<value_type> *tmp1, *tmp2;
            data_reg.push_back(tmp1 = new tensor_t<value_type>(N, C, H, W, &reg, DATA_SOURCE, 0));
            label_reg.push_back(tmp2 = new tensor_t<value_type>(N, 1, 1, 1, &reg, DATA_SOURCE, 0));
            stage.push_back(freed);
        }

        fetch_free_next_idx = 0;
        fetch_gpu_next_idx = 0;
        transfer_next_idx = 0;
    }

    virtual ~tensor_queue_t() {
        m.unlock();
        m.lock();
        for (size_t i = 0; i < reg.size(); ++i) {
            tensor_t<value_type>* t = reg[i];
            t->sync_cpu_to_gpu();
            t->free_gpu_space();
            delete t;
        }
        m.unlock();
    }

    bool fetch_free_tensor(tensor_t<value_type> **data, tensor_t<value_type> **label) {
        bool ret_val;
        m.lock();
        {
            bool res = false;
            size_t idx = fetch_free_next_idx;
            for (size_t i = 0; i < MAX_QUEUE_SIZE; ++i) {
                if (stage[idx] == freed) {
                    stage[idx] = cpu;
                    *data = data_reg[idx];
                    *label = label_reg[idx];
                    res = true;

                    // move to next item
                    idx = (idx + 1) % MAX_QUEUE_SIZE;
                    break;
                }
                idx = (idx + 1) % MAX_QUEUE_SIZE;
            }
            fetch_free_next_idx = idx;
#ifdef DEBUG_TENSOR_QUEUE
            if (res) {
                printf("T-- finish fetch_free_next_idx = %zu, stage=%d %d %d %d\n", idx, stage[0], stage[1], stage[2],
                       stage[3]);
            }
#endif
            ret_val = res;
        }
        m.unlock();
        return ret_val;
    }

    bool fetch_gpu_tensor(tensor_t<value_type> **data, tensor_t<value_type> **label) {
        bool ret_val;
        m.lock();
        {
            bool res = false;
            size_t idx = fetch_gpu_next_idx;
//#ifdef DEBUG_TENSOR_QUEUE
//            printf("fetch_gpu_next_idx = %zu, stage=%d %d %d %d\n", idx, stage[0], stage[1], stage[2], stage[3]);
//#endif
            for (size_t i = 0; i < MAX_QUEUE_SIZE; ++i) {
                if (stage[idx] == cpu_gpu) {
                    data_reg[idx]->sync_cpu_to_gpu();
                    label_reg[idx]->sync_cpu_to_gpu();
                    *data = data_reg[idx];
                    *label = label_reg[idx];
                    res = true;

                    // move to next item
                    idx = (idx + 1) % MAX_QUEUE_SIZE;
                    break;
                }
                idx = (idx + 1) % MAX_QUEUE_SIZE;
            }
            fetch_gpu_next_idx = idx;
#ifdef DEBUG_TENSOR_QUEUE
            if (res) {
                printf("T-- finish fetch_gpu_next_idx = %zu, stage=%d %d %d %d\n", idx, stage[0], stage[1], stage[2],
                       stage[3]);
            }
#endif
            ret_val = res;
        }
        m.unlock();
        return ret_val;
    }

    bool transfer_cpu_to_gpu() {
        bool ret_val;
        m.lock();
        {
            bool res = false;
            if (in_gpu_cnt >= MAX_GPU_CACHE_SIZE) {
                res = false;
            } else {
                // has free gpu cache
                size_t idx = transfer_next_idx;
                for (size_t i = 0; i < MAX_QUEUE_SIZE; ++i) {
                    if (in_gpu_cnt >= MAX_GPU_CACHE_SIZE) {
                        break;
                    }

                    if (stage[idx] == cpu) {

                        data_reg[idx]->async_cpu_to_gpu();
                        label_reg[idx]->async_cpu_to_gpu();

                        stage[idx] = cpu_gpu;
                        in_gpu_cnt++;

                        res = true;
                    }
                    idx = (idx + 1) % MAX_QUEUE_SIZE;
                }
                transfer_next_idx = idx;
#ifdef DEBUG_TENSOR_QUEUE
                if (res) {
                    printf("T-- finish transfer_next_idx = %zu, stage=%d %d %d %d\n", idx, stage[0], stage[1], stage[2],
                           stage[3]);
                }
#endif
            }
            ret_val = res;
        }
        m.unlock();
        return ret_val;
    }

    bool free_gpu_tensor(tensor_t<value_type> *data, tensor_t<value_type> *label) {
        bool ret_val;
        m.lock();
        {
            bool res = false;
            size_t idx = MAX_QUEUE_SIZE;
            for (size_t i = 0; i < MAX_QUEUE_SIZE; ++i) {
                if (data == data_reg[i] && label == label_reg[i]) {
                    idx = i;
                    break;
                }
            }
            if (idx < MAX_QUEUE_SIZE && stage[idx] == cpu_gpu) {
                // 1. free gpu space
                data_reg[idx]->free_gpu_space();
                label_reg[idx]->free_gpu_space();

                // 2. change stage
                stage[idx] = freed;
                // 3. decrease in_gpu_cnt counter
                in_gpu_cnt--;

                // 4. set return value
                res = true;

#ifdef DEBUG_TENSOR_QUEUE
                printf("T-- free index = %zu\n", idx);
#endif
            }

            ret_val = res;
        }
        m.unlock();
        return ret_val;
    }


};

} // namespace SuperNeurons

#endif //SUPERNEURONS_ITEM_QUEUE_H
