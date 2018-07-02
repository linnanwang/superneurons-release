#include <tensor.h>
#include <math_functions.h>
#include <stream_singleton.h>

namespace SuperNeurons {

__global__ void tensor_sum_fkernel(float* ptr1, float* ptr2, int size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < size) {
        ptr1[i] = ptr1[i] + ptr2[i];
    }
    __syncthreads();
}

__global__ void tensor_sum_dkernel(double* ptr1, double* ptr2, int size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < size) {
        ptr1[i] = ptr1[i] + ptr2[i];
    }
    __syncthreads();
}

template <class value_type>
void tensor_sum(value_type* ptr1, value_type* ptr2, int size)
{
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "HALF precision not supported so far!@softmax_layer.cu:line27\n");
        exit(1);
    } else if(sizeof(value_type) == 4) {
        //single precision
        tensor_sum_fkernel <<<(size+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((float*) ptr1, (float*) ptr2, size);
    } else {
        //double precision
        tensor_sum_dkernel <<<(size+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((double*) ptr1, (double*) ptr2, size);
    }
}

__global__ void tensor_copy_fkernel(float* ptr1, float* ptr2, int size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < size) {
        ptr2[i] = ptr1[i];
    }
    __syncthreads();
}

__global__ void tensor_copy_dkernel(double* ptr1, double* ptr2, int size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < size) {
        ptr2[i] = ptr1[i];
    }
    __syncthreads();
}

template <class value_type>
void tensor_copy(value_type* ptr1, value_type* ptr2, int size)
{
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "HALF precision not supported so far!@softmax_layer.cu:line27\n");
        exit(1);
    } else if(sizeof(value_type) == 4) {
        //single precision
        tensor_copy_fkernel <<<(size+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((float*) ptr1,  (float*)  ptr2, size);
    } else {
        //double precision
        tensor_copy_dkernel <<<(size+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((double*) ptr1, (double*) ptr2, size);
    }
}

__global__ void tensor_scale_fkernel(float* ptr1, float s, int size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < size) {
        ptr1[i] = ptr1[i] * s;
    }
    __syncthreads();
}

__global__ void tensor_scale_dkernel(double* ptr1, double s, int size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < size) {
        ptr1[i] = ptr1[i] * s;
    }
    __syncthreads();
}

template <class value_type>
void tensor_scale(value_type* ptr1, value_type s, int size)
{
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "HALF precision not supported so far!@softmax_layer.cu:line27\n");
        exit(1);
    } else if(sizeof(value_type) == 4) {
        //single precision
        tensor_scale_fkernel <<<(size+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((float*) ptr1,  (float)  s, size);
    } else {
        //double precision
        tensor_scale_dkernel <<<(size+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((double*) ptr1, (double) s, size);
    }
}

template void tensor_scale<float>  (float*  ptr1, float  s, int size);
template void tensor_scale<double> (double* ptr1, double s, int size);
template void tensor_sum<float>    (float*  ptr1,  float* ptr2,  int size);
template void tensor_sum<double>   (double* ptr1, double* ptr2,  int size);
template void tensor_copy<float>   (float*  ptr1,  float* ptr2,  int size);
template void tensor_copy<double>  (double* ptr1, double* ptr2,  int size);


} //SuperNeurons
