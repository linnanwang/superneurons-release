//
// Created by ay27 on 7/24/17.
//

#ifndef SUPERNEURONS_THREAD_ROUTINE_H
#define SUPERNEURONS_THREAD_ROUTINE_H

#include <util/common.h>
#include <atomic>
#include <thread>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#include <thread>
#include <pthread.h>
#endif


inline void set_cpu_affinity(pthread_t thread, int cpu_id) {
#ifdef __linux__
    long numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    unsigned int nthreads = std::thread::hardware_concurrency();
    numCPU = numCPU > nthreads ? numCPU : nthreads;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // set affinity to the last core
    if (cpu_id < 0) {
        CPU_SET(numCPU + cpu_id, &cpuset);
    }
    else {
        CPU_SET(cpu_id, &cpuset);
    }
    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np with rc = " << rc << std::endl;
    }
//    } else {
//        // must wait a little to set affinity
//        std::this_thread::sleep_for(std::chrono::milliseconds(20));
//        std::cout << "thread run on CPU " << sched_getcpu() << "\n";
//    }
#endif
}

inline void set_main_thread_cpu_affinity(int cpu_id) {
#ifdef __linux__

    if (cpu_id > -1)
    {
        cpu_set_t mask;
        int status;

        CPU_ZERO(&mask);
        CPU_SET(cpu_id, &mask);
        status = sched_setaffinity(0, sizeof(mask), &mask);
        if (status != 0)
        {
            fprintf(stderr, "set affinity error %d\n", status);
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            std::cout << "Main thread run on CPU " << sched_getcpu() << "\n";
        }
    }
#endif
}

namespace SuperNeurons {

class thread_routine_t {
private:
    const size_t max_thread_num;
    const int cpu_id;
    std::atomic_bool _should_stop;
    std::vector<std::shared_ptr<std::thread>> threads;

public:
    thread_routine_t(size_t num_threads, int cpu_id_=-1) : max_thread_num(num_threads), _should_stop(false), cpu_id(cpu_id_) { }

    void start() {
        for (size_t i = 0; i < max_thread_num; ++i) {
            threads.push_back(std::make_shared<std::thread>([&](thread_routine_t* this_ptr, size_t thread_idx) {
#ifdef __linux__
                // must wait a little to set affinity
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                std::cout << "thread run on CPU " << sched_getcpu() << "\n";
#endif
                this_ptr->thread_entry(thread_idx, max_thread_num);
            }, this, i));

            set_cpu_affinity(threads[i]->native_handle(), cpu_id);
        }
    }

    void stop() {
        _should_stop = true;
        for (size_t i = 0; i < max_thread_num; ++i) {
            threads[i].get()->join();
        }
    }

    bool should_stop() {
        return _should_stop.load();
    }

    virtual void thread_entry(size_t thread_idx, size_t total_threads) = 0;
};

} // namespace SuperNeurons

#endif //SUPERNEURONS_THREAD_ROUTINE_H
