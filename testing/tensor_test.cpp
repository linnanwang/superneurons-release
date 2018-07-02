//
// Created by ay27 on 7/26/17.
//

#include <tensor.h>
#include <util/common.h>
#include <stdio.h>

using namespace std;
using namespace SuperNeurons;

int main() {
    size_t N = 2000,
            C = 3,
            H = 227,
            W = 227;

    vector<tensor_t<float> *> reg;

    tensor_t<float> t1(N, C, H, W, &reg, DATA_SOURCE, 0);
    tensor_t<float> t2(N, C, H, W, &reg, DATA_SOURCE, 0);
    tensor_t<float> t3(N, C, H, W, &reg, DATA_SOURCE, 0);
    tensor_t<float> t4(N, C, H, W, &reg, DATA_SOURCE, 0);
    tensor_t<float> t5(N, C, H, W, &reg, DATA_SOURCE, 0);
    tensor_t<float> t6(N, C, H, W, &reg, DATA_SOURCE, 0);

    bool is_ready = t1.is_gpu_to_cpu_ready();
    printf("INIT STAGE, to check if t1 is ready:%d\n", is_ready);

    // init tensor
    for (size_t i = 0; i < N * C * H * W; ++i) {
        t1.get_cpu_ptr()[i] = i;
        t2.get_cpu_ptr()[i] = i;
        t3.get_cpu_ptr()[i] = i;
        t4.get_cpu_ptr()[i] = i;
        t5.get_cpu_ptr()[i] = i;
        t6.get_cpu_ptr()[i] = i;
    }

    // sync
    double ts = get_cur_time();
    t1.CPUtoGPU();
    t2.CPUtoGPU();
    t3.CPUtoGPU();
    t4.CPUtoGPU();
    printf("cpu to gpu sync time = %f\n", get_cur_time() - ts);

    for (size_t i = 0; i < N * C * H * W; ++i) {
        t1.get_cpu_ptr()[i] = 0;
        t2.get_cpu_ptr()[i] = 0;
        t3.get_cpu_ptr()[i] = 0;
        t4.get_cpu_ptr()[i] = 0;
    }
    t5.CPUtoGPU();
    t6.CPUtoGPU();

    ts = get_cur_time();
    t1.GPUtoCPU();
    t2.GPUtoCPU();
    t3.GPUtoCPU();
    t4.GPUtoCPU();
    printf("gpu to cpu sync time = %f\n", get_cur_time() - ts);

    for (size_t i = 0; i < N * C * H * W; ++i) {
        if (t1.get_cpu_ptr()[i] != i || t2.get_cpu_ptr()[i] != i || t3.get_cpu_ptr()[i] != i || t4.get_cpu_ptr()[i] != i) {
            fprintf(stderr, "error in %zu\n", i);
        }
    }

    // async
    ts = get_cur_time();
    t1.async_cpu_to_gpu();
    t2.async_cpu_to_gpu();
    t3.async_cpu_to_gpu();
    t4.async_cpu_to_gpu();

    cublasHandle_t handler;
    cublasCreate_v2(&handler);
    float one=1.0;
    cublasSaxpy_v2(handler, N*C*H*W, &one, t5.get_gpu_ptr(), 1, t6.get_gpu_ptr(), 1);


    while (!(t1.is_cpu_to_gpu_ready() && t2.is_cpu_to_gpu_ready() && t3.is_cpu_to_gpu_ready() &&
             t4.is_cpu_to_gpu_ready())) {}
    printf("cpu to gpu Async time = %f\n", get_cur_time() - ts);

    for (size_t i = 0; i < N * C * H * W; ++i) {
        t1.get_cpu_ptr()[i] = 0;
        t2.get_cpu_ptr()[i] = 0;
        t3.get_cpu_ptr()[i] = 0;
        t4.get_cpu_ptr()[i] = 0;
    }

    ts = get_cur_time();
    t1.async_gpu_to_cpu();
    t2.async_gpu_to_cpu();
    t3.async_gpu_to_cpu();
    t4.async_gpu_to_cpu();
    while (!(t1.is_gpu_to_cpu_ready() && t2.is_gpu_to_cpu_ready() && t3.is_gpu_to_cpu_ready() &&
             t4.is_gpu_to_cpu_ready())) {}
    printf("gpu to cpu Async time = %f\n", get_cur_time() - ts);

    for (size_t i = 0; i < N * C * H * W; ++i) {
        if (t1.get_cpu_ptr()[i] != i || t2.get_cpu_ptr()[i] != i || t3.get_cpu_ptr()[i] != i || t4.get_cpu_ptr()[i] != i) {
            fprintf(stderr, "error in %zu\n", i);
        }
    }
    
    
}
