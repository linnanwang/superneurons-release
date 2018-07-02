#include <layer/padding_layer.h>
#include <device_atomic_functions.h>
#include <math_functions.h>
#include <stream_singleton.h>


namespace SuperNeurons {

template<class value_type>
__global__ void padding_fkernel(size_t N, size_t C, size_t H, size_t W, size_t padC, size_t padH, size_t padW,
                                const value_type *src, value_type *dst) {

    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        size_t CC = C + 2 * padC, HH = H + 2 * padH, WW = W + 2 * padW;

        for (size_t c = 0; c < padC; ++c) {
            for (size_t hw = 0; hw < HH * WW; ++hw) {
                dst[(n * CC + c) * HH * WW + hw] = 0.0;
            }
        }

        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < padH; ++h) {
                for (size_t w = 0; w < WW; ++w) {
                    dst[index(n, c + padC, h, w, CC, HH, WW)] = 0.0;
                }
            }

            for (size_t h = 0; h < H; ++h) {
                // pad before w
                for (size_t w = 0; w < padW; ++w) {
                    dst[index(n, c + padC, h + padH, w, CC, HH, WW)] = 0;
                }
                // copy it
                for (size_t w = 0; w < W; ++w) {
                    dst[index(n, c + padC, h + padH, w + padW, CC, HH, WW)] = src[index(n, c, h, w, C, H, W)];
                }
                // pad after
                for (size_t w = 0; w < padW; ++w) {
                    dst[index(n, c + padC, h + padH, w + padW + W, CC, HH, WW)] = 0;
                }
            }
            // pad after
            for (size_t h = 0; h < padH; ++h) {
                for (size_t w = 0; w < WW; ++w) {
                    dst[index(n, c + padC, h + padH + H, w, CC, HH, WW)] = 0.0;
                }
            }
        }

        for (size_t c = 0; c < padC; ++c) {
            for (size_t hw = 0; hw < HH * WW; ++hw) {
                dst[(n * CC + c + padC + C) * HH * WW + hw] = 0.0;
            }
        }
    }

    __syncthreads();
}


template<class value_type>
__global__ void padding_bkernel(size_t N, size_t C, size_t H, size_t W, size_t padC, size_t padH, size_t padW,
                                const value_type *src, value_type *dst) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        size_t CC = C + 2 * padC, HH = H + 2 * padH, WW = W + 2 * padW;
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    dst[index(n, c, h, w, C, H, W)] = src[index(n, c + padC, h + padH, w + padW, CC, HH,
                                                                         WW)];
                }
            }
        }
        __syncthreads();
    }
}


template<class value_type>
void padding_forward(size_t N, size_t C, size_t H, size_t W, size_t padC, size_t padH, size_t padW,
                     const value_type *src, value_type *dst) {
    padding_fkernel<value_type> << < (N + 255) / 256, 256, 0, stream_singleton::get_compute_stream() >> > (N, C, H, W, padC, padH, padW, src, dst);
}

template<class value_type>
void padding_backward(size_t N, size_t C, size_t H, size_t W, size_t padC, size_t padH, size_t padW,
                     const value_type *src, value_type *dst) {
    padding_bkernel<value_type> << < (N + 255) / 256, 256, 0, stream_singleton::get_compute_stream() >> > (N, C, H, W, padC, padH, padW, src, dst);
}

template void padding_forward<float>(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const float*, float*);

template void padding_forward<double>(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const double *, double*);

template void padding_backward<float>(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const float*, float*);

template void padding_backward<double>(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const double *, double*);



} // namespace SuperNeurons