#include <solver.h>
#include <math_functions.h>
#include <stream_singleton.h>

namespace SuperNeurons {


template<typename value_type>
__global__ void AdaGradUpdate(int N, value_type *grad, value_type *history, value_type eps,
                              value_type local_rate) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float gi = grad[i];
        float hi = history[i] = history[i] + gi * gi;
        grad[i] = local_rate * gi / (sqrt(hi) + eps);
    }
    __syncthreads();
}

template<typename value_type>
void adagrad_update(int N, value_type *g, value_type *h, value_type delta,
                    value_type local_rate) {
    AdaGradUpdate<value_type> << < (N + 255) / 256, 256, 0, stream_singleton::get_compute_stream() >> > (N, g, h, delta, local_rate);
}


template<typename value_type>
__global__ void RMSPropUpdate(int N, value_type *grad, value_type *history,
                              value_type rms_decay, value_type delta, value_type local_rate) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float gi = grad[i];
        float hi = history[i] = rms_decay * history[i] + (1 - rms_decay) * gi * gi;
        grad[i] = local_rate * grad[i] / (sqrt(hi) + delta);
    }
    __syncthreads();
}

template<typename value_type>
void rmsprop_update(int N, value_type *g, value_type *h, value_type rms_decay,
                    value_type delta, value_type local_rate) {
    RMSPropUpdate<value_type> << < (N + 255) / 256, 256, 0, stream_singleton::get_compute_stream() >> > (N, g, h, rms_decay, delta, local_rate);
}

template <typename Dtype>
__global__ void MomentumUpdate(int N, Dtype* g, Dtype* h, Dtype momentum, Dtype local_rate) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        g[i] = h[i] = momentum*h[i] + local_rate*g[i];
    }
    __syncthreads();
}

template<typename value_type>
void momentum_update(int N, value_type* g, value_type* h, value_type momentum, value_type local_rate) {
    MomentumUpdate<value_type> << < (N + 255) / 256, 256, 0, stream_singleton::get_compute_stream() >> > (N, g, h, momentum, local_rate);
}

template void momentum_update(int N, float* g, float* h, float momentum, float local_rate);

template void momentum_update(int N, double* g, double* h, double momentum, double local_rate);

template void adagrad_update<float>(int, float *, float *, float, float);

template void adagrad_update<double>(int, double *, double *, double, double);

template void rmsprop_update<float>(int, float *, float *, float, float, float);

template void rmsprop_update<double>(int, double *, double *, double, double, double);

}  // namespace SuperNeurons
