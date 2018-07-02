#if !defined(_TENSOR_H_)
#define _TENSOR_H_
#include <vector>
#include <cassert>
#include <cudnn.h>
#include <stdio.h>
#include <switch.h>
#include <random>
#include <sys/time.h>
#include <chrono>
#include <math.h>
#include <atomic>
#include <util/error_util.h>
#include <util/common.h>
#include <initializer.h>
#include <stream_singleton.h>
#include <gpu_malloc.h>
#include <util/lru.h>
#include <cufft.h>


//#define BLASX_MALLOC

namespace SuperNeurons{
    
typedef enum TENSOR_TYPE {
    DATA        = 0,
    GRAD        = 1,
    PARAM       = 2,
    AUX         = 3,
    BN_MEAN_VAR = 4,
    CONV_BUFF   = 5,
    DATA_SOURCE = 6
}TENSOR_TYPE;

    
template <class value_type>
class tensor_t {
private:
    std::atomic<int> state;
    
    TENSOR_TYPE    data_t;
    value_type*    gpu_ptr  = NULL;                  //gpu and cpu data are mutually exclusive
    value_type*    cpu_ptr  = NULL;
    cufftComplex*  freq_ptr = NULL;
    
    size_t GPU_id;                               //this identifies the GPU RAM
    int layer_id;                                //this identifies the affilited layer
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    /*--cufft-handle--*/
    //this has to for each tensor
    cufftHandle fft_plan_f;
    cufftHandle fft_plan_b;
    
    /*---tensor_id----*/
    static size_t tensor_counter;
    const  size_t tensor_id = 0;
    
#ifdef LRU_ON
    lru_list_t* lru = lru_singleton::get_lru();
#endif
    
    //CUDNN configuration
    cudnnDataType_t cudnn_data_type;
    cudnnTensorFormat_t cudnn_tensor_format;    //default with NCHW
    cudnnTensorDescriptor_t cudnn_tensor_desc;
    
    cudaEvent_t  cpu2gpu_event, gpu2cpu_event;
    std::atomic_bool cpu2gpu_event_not_happen, gpu2cpu_event_not_happen;

    blasx_gpu_malloc_t* gpu_malloc = NULL;

    void acquireSpaceCPU(long total) {
        assert( cpu_ptr == NULL );
        assert( total > 0 );
        checkCudaErrors( cudaMallocHost(&(this->cpu_ptr), total*sizeof(value_type) ) );
    }

    void freeSpaceCPU() {
        if (cpu_ptr == NULL) {
            return;
        }
        checkCudaErrors(cudaFreeHost(this->cpu_ptr));
        this->cpu_ptr = NULL;
    }

    void acquireSpaceGPU(long total);
    void freeSpaceGPU(mem_mode target=CPU);

    // make it private
    void atomic_set_state(int m);
    inline void check_state(mem_mode target);
    
public:

    int hit_cnt = 0, miss_cnt = 0, into_cnt = 0;

    tensor_t(size_t n, size_t c, size_t h, size_t w, std::vector<tensor_t<value_type>* >* reg, TENSOR_TYPE dtype, int layer_id):tensor_id(tensor_counter) {
        assert(n >= 1);
        assert(c >= 1);
        assert(h >= 1);
        assert(w >= 1);
        
        // TODO : set GPU affinity
        GPU_id = 0;
#ifdef BLASX_MALLOC
        gpu_malloc = blasx_gpu_singleton::get_blasx_gpu_malloc_t(GPU_id);
#endif
        
        this->state    = VOID;
        this->data_t   = dtype;
        this->layer_id = layer_id;

        
        switch (sizeof(value_type))
        {
            case 2 : cudnn_data_type = CUDNN_DATA_HALF; break;
            case 4 : cudnn_data_type = CUDNN_DATA_FLOAT; break;
            case 8 : cudnn_data_type = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
    
        cudnn_tensor_format = CUDNN_TENSOR_NCHW;
        checkCUDNN( cudnnCreateTensorDescriptor(&cudnn_tensor_desc) );
        checkCUDNN( cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc,
                                               this->cudnn_tensor_format,
                                               this->cudnn_data_type,
                                               n, c, h, w) );
        const size_t total_size = n * c * h * w;
        
        // GRAD init CuComplex array
        // specialized initialization
        // rigth now we only consder 1d case
        if (this->data_t == GRAD) {
            //cuFFT only gives half of the freq info
            const size_t freq_size = total_size / 2 + 1;
            CHECK_EQ( cudaMalloc((void**)&(this->freq_ptr), sizeof(cufftComplex)*total_size), cudaSuccess);
            CHECK_NOTNULL(this->freq_ptr);
            CHECK_EQ( cufftPlan1d(&fft_plan_f, total_size, CUFFT_R2C, 1), CUFFT_SUCCESS );
            CHECK_EQ( cufftPlan1d(&fft_plan_b, total_size, CUFFT_C2R, 1), CUFFT_SUCCESS );
        }
        
#ifdef LIVENESS
        if(this->data_t != CONV_BUFF && this->data_t != DATA ) {
            acquireSpaceGPU(n*c*h*w);
        }
        if( this->data_t != CONV_BUFF ) acquireSpaceCPU(n*c*h*w);
#else
        acquireSpaceCPU(n*c*h*w);
        acquireSpaceGPU(n*c*h*w);
        this->atomic_set_state(GPU_FUL);
#endif
        
        this->N = n;
        this->C = c;
        this->H = h;
        this->W = w;
        
        reg->push_back(this);
        /*---init-event-asyn-comm--*/
        checkCudaErrors(cudaEventCreate(&cpu2gpu_event));
        checkCudaErrors(cudaEventCreate(&gpu2cpu_event));
        cpu2gpu_event_not_happen = true;
        gpu2cpu_event_not_happen = true;
        /*---init-counter---*/
        tensor_counter++;
        
#ifdef DEBUG
        const size_t total_size_bytes = sizeof(value_type)*n*c*h*w;
        
        if(this->data_t == DATA) {
            printf("create tensor:%p DATA gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == PARAM) {
            printf("create tensor:%p PARAM gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == GRAD) {
            printf("create tensor:%p GRAD gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == AUX) {
            printf("create tensor:%p AUX gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == BN_MEAN_VAR) {
            printf("create tensor:%p BN_MEAN_VAR gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == CONV_BUFF) {
            printf("create tensor:%p CONV_BUFF gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else if(this->data_t == DATA_SOURCE) {
            printf("create tensor:%p DATA_SOURCE gpu_ptr:%p size: %zu byte\n", this, gpu_ptr, total_size_bytes);
        } else {
            printf("unsupported type@%d tensor.h line 86\n", this->data_t);
            exit(1);
        }
#endif
    }
    
    ~tensor_t() {
        if(cpu_ptr != NULL) cudaFreeHost(cpu_ptr);
        if(gpu_ptr != NULL) gpu_ptr = NULL;
        checkCUDNN( cudnnDestroyTensorDescriptor(cudnn_tensor_desc) );
        cufftDestroy(fft_plan_f);
        cufftDestroy(fft_plan_b);
        checkCudaErrors(cudaEventDestroy(cpu2gpu_event));
        checkCudaErrors(cudaEventDestroy(gpu2cpu_event));
    }
    
    /*----utility functions----*/

    /**
     * NCHW, layer_id, data_type, data
     */
    void gen_description(char* buff, size_t* len_in_byte) {
        value_type _n = N, _c = C, _h = H, _w = W;
        value_type _layer_id = layer_id, _type = data_t;

        size_t SIZE = sizeof(value_type);
        memcpy(buff, &_n, SIZE);
        memcpy(buff+1*SIZE, &_c, SIZE);
        memcpy(buff+2*SIZE, &_h, SIZE);
        memcpy(buff+3*SIZE, &_w, SIZE);
        memcpy(buff+4*SIZE, &_layer_id, SIZE);
        memcpy(buff+5*SIZE, &_type, SIZE);

        this->GPUtoCPU();

        memcpy(buff+6*SIZE, this->cpu_ptr, N*C*H*W*SIZE);

        *len_in_byte = (6+N*C*H*W)*SIZE;
    }

    //those are for mem_controller
    void stash_gpu_space() {
        long total = this->N * this->C * this->H * this->W;
        acquireSpaceGPU(total);
    }
    
    inline void free_gpu_space(mem_mode target=CPU) {
        freeSpaceGPU(target);
    }
    
    inline size_t get_N() { return this->N; }
    
    inline size_t get_C() { return this->C; }
    
    inline size_t get_H() { return this->H; }
    
    inline size_t get_W() { return this->W; }
    
    inline size_t get_scalar_count() {
        return this->get_N()*this->get_C()*this->get_H()*this->get_W();
    }
    
    inline size_t get_mem_size() {
        const size_t total_size_bytes = sizeof(value_type)*this->N*this->C*this->H*this->W;
        return total_size_bytes;
    }
    
    void reshape(size_t n, size_t c, size_t h, size_t w) {
        assert(N*C*H*W == n*c*h*w);
        this->N = n;
        this->C = c;
        this->H = h;
        this->W = w;
        checkCUDNN( cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc,
                                               this->cudnn_tensor_format,
                                               this->cudnn_data_type,
                                               n, c, h, w) );
    }

    // add it to support data reader
    void replace_data(value_type *new_cpu_ptr, value_type* new_gpu_ptr=NULL);

    void replace_gpu_ptr_without_free(value_type* new_gpu_ptr) {
        // NOTE: this is a danger action!!!! now it just used in parallel_reader!!!!
        this->gpu_ptr = new_gpu_ptr;
        this->atomic_set_state(GPU_FUL);
    }
    
    inline int get_layer_id() {
        return this->layer_id;
    }

    inline TENSOR_TYPE get_type() {
        return this->data_t;
    }

    inline value_type* get_cpu_ptr() {
        return this->cpu_ptr;
    }
    
    inline value_type* get_gpu_ptr() {
        return this->gpu_ptr;
    }
    
    inline cudnnTensorDescriptor_t get_tensor_desc() {
        return this->cudnn_tensor_desc;
    }
    
    inline cudnnTensorFormat_t get_tensor_format() {
        return this->cudnn_tensor_format;
    }
    
    void GPUtoCPU();
    
    void CPUtoGPU();
    
    void async_cpu_to_gpu();
    
    void async_gpu_to_cpu();
    
    inline bool is_cpu_to_gpu_ready();
    
    inline bool is_gpu_to_cpu_ready();
    
    void sync_cpu_to_gpu();
    
    void sync_gpu_to_cpu();
    
    void init(initializer_t<value_type> *initializer);
    
    void printTensorNoDebug(const char* str);
    
    void printTensor(const char* str);
    
    void printTensorFirst(const char* str);
    
    void writeToFile(const char* str);
    
    void hostRegister();
    
    void resizeTensor(size_t n, size_t c, size_t h, size_t w);
    
    void copy(tensor_t<value_type>* t,
              int src_start_idx=-1, int src_end_idx=-1, int dst_start_idx=-1, int dst_end_idx=-1);
    
    value_type get_scalar(const size_t n, const size_t c, const size_t h, const size_t w);
    
    void set_scalar( const size_t n, const size_t c, const size_t h, const size_t w, const value_type t );
    
    mem_mode get_state();

    /*---math functions-------*/
    void scale(value_type s);
    
    void sum(tensor_t<value_type>* t);
    
    value_type squared_sum(cublasHandle_t *handle);

    void forward_fft();
    
    void backward_fft();

};


    
} // SuperNeurons namespace

#endif // _TENSOR_H_
