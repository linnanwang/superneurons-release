#include <layer/softmax_layer.h>
#include <device_atomic_functions.h>
#include <math_functions.h>
#include <stream_singleton.h>

namespace SuperNeurons {

//TO DO Half-Precision
//
__global__ void softmax_loss_fkernel(float* pred_gpu_ptr, float* label_gpu_ptr, int N, int C, int H, int W, float* loss)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        int corr   = (int) label_gpu_ptr[i];
        int idx    = i*C*H*W + corr;
        float tmp  = (float) (logf(pred_gpu_ptr[idx]) * -1.0/N);
        atomicAdd(loss, tmp); //atomicAdd only takes float
    }
    __syncthreads();
}

__global__ void softmax_loss_dkernel(double* pred_gpu_ptr, double* label_gpu_ptr, int N, int C, int H, int W, float* loss) {

}


template <class value_type>
float softmax_loss(value_type* pred_gpu_ptr, value_type* label_gpu_ptr, int N, int C, int H, int W)
{
    float* loss = NULL;
    cudaMalloc(&(loss), sizeof(float));
    cudaMemset(loss, 0, sizeof(float));
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "HALF precision not supported so far!@softmax_layer.cu:line27\n");
        exit(1);
    } else if(sizeof(value_type) == 4) {
        //single precision
        softmax_loss_fkernel <<<(N+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((float*) pred_gpu_ptr, (float*) label_gpu_ptr, N, C, H, W,  loss);
    } else {
        //double precision
        //TODO
    }
    float result = 0;
    cudaMemcpy((void*) &result, (void*) loss, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

__global__ void softmax_grad_fkernel(float* pred_gpu_ptr, float* label_gpu_ptr, int N, int C, int H, int W)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        int corr   = (int) label_gpu_ptr[i];
        int idx    = i*C*H*W + corr;
        pred_gpu_ptr[idx] -= 1;
    }
    __syncthreads();
}

__global__ void softmax_grad_dkernel(double* pred_gpu_ptr, double* label_gpu_ptr, int N, int C, int H, int W)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        int corr   = (int) label_gpu_ptr[i];
        int idx    = i*C*H*W + corr;
        pred_gpu_ptr[idx] -= 1;
    }
    __syncthreads();
}


template <class value_type>
void softmax_grad(value_type* pred_gpu_ptr, value_type* label_gpu_ptr, int N, int C, int H, int W)
{
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "HALF precision not supported so far!@softmax_layer.cu:line27\n");
        exit(1);
    } else if(sizeof(value_type) == 4) {
        //single precision
        softmax_grad_fkernel <<<(N+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((float*) pred_gpu_ptr, (float*) label_gpu_ptr, N, C, H, W);
    } else {
        //double precision
        softmax_grad_dkernel <<<(N+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((double*) pred_gpu_ptr, (double*) label_gpu_ptr, N, C, H, W);
    }
}

__global__ void softmax_top1_accuracy_fkernel(float* label, float* predict, int N, int C, int H, int W, int* corr_counter)
{
    //atomic add takes int argument
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        float max   = 0;
        int pred    = 0;
        float corr  = label[i];

        for(int j = 0; j < C; j++) {
            float curt = predict[ i*C*H*W+j ];
            if( max < curt ) {
                max   = curt;
                pred  = j;
            }
        }
        if (corr == pred) {
            atomicAdd( corr_counter, 1 );
        }
    }
    __syncthreads();
}

__global__ void softmax_top1_accuracy_dkernel(double* label, double* predict, int N, int C, int H, int W, int* corr_counter)
{
    //atomic add takes int argument
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        double max   = 0;
        int pred    = 0;
        double corr  = label[i];

        for(int j = 0; j < C; j++) {
            double curt = predict[ i*C*H*W+j ];
            if( max < curt ) {
                max   = curt;
                pred  = j;
            }
        }
        if (corr == pred) {
            atomicAdd( corr_counter, 1 );
        }
    }
    __syncthreads();
}


template <class value_type>
value_type softmax_top1_accuracy(value_type* label, value_type* predict, int N, int C, int H, int W) {
    int* corr_counter = NULL;
    cudaMalloc(&(corr_counter), sizeof(int));
    cudaMemset(corr_counter, 0, sizeof(int));

    if (sizeof(value_type) == 2) {
        fprintf(stderr, "HALF precision not supported so far!@softmax_layer.cu:line27\n");
        exit(1);
    } else if(sizeof(value_type) == 4) {
        //single precision
        softmax_top1_accuracy_fkernel <<<(N+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((float*) label, (float*) predict, N, C, H, W, corr_counter );
    } else {
        //double precision
        softmax_top1_accuracy_dkernel <<<(N+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((double*) label, (double*) predict, N, C, H, W, corr_counter );

    }
    int result = 0;
    cudaMemcpy((void*) &result, (void*) corr_counter, sizeof(int), cudaMemcpyDeviceToHost);
    value_type accuracy = (value_type) result / (value_type) N;
    return accuracy;
}

__global__ void softmax_top5_accuracy_fkernel(float* label, float* predict, int N, int C, int H, int W, int* corr_counter)
{
    //atomic add takes int argument
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
            int idx_top5[5] = {-1, -1, -1, -1, -1};
            int corr = (int) label[i];
            for(int j = 0; j < 5; j++) 
            {
                float max = 0;
                int pred = 0;
                for( int k = 0; k < C; k++ )
                {
                    bool is_skip = false;
                	for(int m = 0; m < 5; m++) {
                	    if(idx_top5[m] == k) is_skip = true;
                	}
                    if( is_skip ) continue;
                    float curt = predict[i*C*H*W+k];
                    if( max < curt )
                    {
                        max  = curt;
                        pred = k;
                    }
                }
                idx_top5[j] = pred;
            }
            bool is_corr = false;
            for(int m = 0; m < 5; m++) {
                if(idx_top5[m] == corr) is_corr = true;
            }
            if ( is_corr ) {
            	atomicAdd( corr_counter, 1 );
            }
    }
    __syncthreads();
}

template <class value_type>
value_type softmax_top5_accuracy(value_type* label, value_type* predict, int N, int C, int H, int W) {
    int* corr_counter = NULL;
    cudaMalloc(&(corr_counter), sizeof(int));
    cudaMemset(corr_counter, 0, sizeof(int));

    if (sizeof(value_type) == 2) {
        fprintf(stderr, "HALF precision not supported so far!@softmax_layer.cu:line27\n");
        exit(1);
    } else if(sizeof(value_type) == 4) {
        //single precision
        softmax_top5_accuracy_fkernel <<<(N+255)/256, 256, 0, stream_singleton::get_compute_stream()>>> ((float*) label, (float*) predict, N, C, H, W, corr_counter );
    } else {
        //double precision
        //softmax_top5_accuracy_dkernel <<<(N+255)/256, 256>>> ((float*) label, (float*) predict, N, C, H, W, corr_counter );
    }
    int result = 0;
    cudaMemcpy((void*) &result, (void*) corr_counter, sizeof(int), cudaMemcpyDeviceToHost);
    value_type accuracy = (value_type) result / (value_type) N;
    return accuracy;
}

template float softmax_top5_accuracy<float>(float* label, float* predict, int N, int C, int H, int W);
template double softmax_top5_accuracy<double>(double* label, double* predict, int N, int C, int H, int W);

template float softmax_top1_accuracy<float>(float* label, float* predict, int N, int C, int H, int W);
template double softmax_top1_accuracy<double>(double* label, double* predict, int N, int C, int H, int W);

template float softmax_loss<float>(float* pred_gpu_ptr, float* label_gpu_ptr, int N, int C, int H, int W);
template float softmax_loss<double>(double* pred_gpu_ptr, double* label_gpu_ptr, int N, int C, int H, int W);

template void softmax_grad<float>(float* pred_gpu_ptr, float* label_gpu_ptr, int N, int C, int H, int W);
template void softmax_grad<double>(double* pred_gpu_ptr, double* label_gpu_ptr, int N, int C, int H, int W);

} //SuperNeurons
