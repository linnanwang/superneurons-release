#include <util/common.h>
#include <tensor.h>
#include <cublas_alias.h>
#include <util/mem_util.h>
#include <thread>


namespace SuperNeurons{

//PRIVATE METHODS

template <class value_type>
inline void tensor_t<value_type>::check_state(mem_mode target) {
#ifdef DEBUG
    mem_mode curt = this->get_state();
    if (curt != target) {
        printf("err state: tensor %p current state is : %d, target state is : %d\n", this, curt, target);
    }
#endif
}

template <class value_type>
inline void tensor_t<value_type>::atomic_set_state(int new_val) {
    int old_val = this->state.load();
#ifdef DEBUG
    printf("^^change state: layer %d tensor %p, %d -> %d\n", layer_id, this, old_val, new_val);
#endif
    while( !(this->state.compare_exchange_strong( old_val, new_val ) ) ) {
        old_val = this->state.load();
    };
}

template <class value_type>
inline mem_mode tensor_t<value_type>::get_state() {
    return (mem_mode) this->state.load();
}

//PUBLIC METHODS

template <class value_type>
void tensor_t<value_type>::sync_cpu_to_gpu() {
    /**
     * sync the async data transfer
     * state: CPU2GPU -> GPU_FUL
     */

    if (is_cpu_to_gpu_ready()) {
        this->atomic_set_state(GPU_FUL);
        return;
    }

    check_state(CPU2GPU);

    checkCudaErrors( cudaEventSynchronize(this->cpu2gpu_event) );
    while (!is_cpu_to_gpu_ready()) { }
    this->atomic_set_state(GPU_FUL);

#ifdef LRU_ON
    if (this->get_type() == DATA) {
        lru->update(this);
    }
#endif
}

template <class value_type>
void tensor_t<value_type>::sync_gpu_to_cpu() {
    /**
     * sync the async data transfer
     * state: GPU2CPU -> GPU_FUL
     */

    if (is_gpu_to_cpu_ready()) {
        return;
    }

    check_state(GPU2CPU);

    checkCudaErrors( cudaEventSynchronize(this->gpu2cpu_event) );
    while (!is_gpu_to_cpu_ready()) { }
    this->atomic_set_state(GPU_FUL);
}

template <class value_type>
void tensor_t<value_type>::GPUtoCPU() {
    /**
     * Sync GPU to CPU
     * state : GPU_FUL
     */
    check_state(GPU_FUL);

    assert(this->cpu_ptr != NULL);
    assert(this->gpu_ptr != NULL);
    long total = this->N*this->C*this->H*this->W;
    checkCudaErrors( cudaMemcpy((void*) this->cpu_ptr, (void*) this->gpu_ptr, total*sizeof(value_type), cudaMemcpyDeviceToHost) );
#ifdef DEBUG
    printf("GPUtoCPU : %p layer %d type %d\n", this, this->get_layer_id(), this->get_type());
#endif
}

template <class value_type>
void tensor_t<value_type>::CPUtoGPU() {
    /**
     * Sync CPU to GPU
     * state : VOID, CPU, GPU_NIL, RECOMPUTE -> GPU_FUL
     */
    assert(this->cpu_ptr != NULL);

    if (data_t == DATA) {
        into_cnt += 1;
    }

    if (this->get_state() == GPU_FUL) {
        if (data_t == DATA) {
            hit_cnt += 1;
        }
        // we do nothing because GPU has valid data
        return;
    }

    bool is_void_pre = this->get_state() == VOID;
    bool is_recompute_pre = this->get_state() == RECOMPUTE;

    if (this->gpu_ptr == NULL) {
        stash_gpu_space();
    }

    if (is_void_pre || is_recompute_pre) {
        this->atomic_set_state(GPU_FUL);
        if (data_t == DATA) {
            hit_cnt += 1;
        }

#ifdef LRU_ON
        if (this->get_type() == DATA) {
            lru->update(this);
        }
#endif

        return;
    }

    check_state(GPU_NIL);

    if (data_t == DATA) {
        miss_cnt += 1;
    }

    long total = this->N*this->C*this->H*this->W;
    checkCudaErrors( cudaMemcpy((void*) this->gpu_ptr, (void*) this->cpu_ptr, total*sizeof(value_type), cudaMemcpyHostToDevice) );

#ifdef DEBUG
    printf("CPUtoGPU : %p layer %d type %d\n", this, this->get_layer_id(), this->get_type());
#endif

    this->atomic_set_state(GPU_FUL);

#ifdef LRU_ON
    if (this->get_type() == DATA) {
        lru->update(this);
    }
#endif
}

template <class value_type>
void tensor_t<value_type>::async_cpu_to_gpu() {
    /**
     * Async CPU to GPU
     * state : GPU_NIL -> CPU2GPU
     */
    assert(this->cpu_ptr != NULL);

    if (this->get_state() == GPU_FUL) {
        // we do nothing because GPU has valid data
        return;
    }

    if (this->gpu_ptr == NULL) {
        stash_gpu_space();
    }

    check_state(GPU_NIL);

    this->atomic_set_state(CPU2GPU);
    long total = this->N*this->C*this->H*this->W;
    checkCudaErrors(cudaMemcpyAsync((void*) this->gpu_ptr, (void*)this->cpu_ptr,
                                    total* sizeof(value_type), cudaMemcpyHostToDevice, stream_singleton::get_cpu2gpu_stream()));
    checkCudaErrors(cudaEventRecord(this->cpu2gpu_event, stream_singleton::get_cpu2gpu_stream()));
}

template <class value_type>
void tensor_t<value_type>::async_gpu_to_cpu() {
    /**
     * Async CPU to GPU
     * state : GPU_FUL -> GPU2CPU
     */
    assert(this->cpu_ptr != NULL);
    assert(this->gpu_ptr != NULL);

    check_state(GPU_FUL);

    this->atomic_set_state(GPU2CPU);
    long total = this->N*this->C*this->H*this->W;
    checkCudaErrors(cudaMemcpyAsync((void*) this->cpu_ptr, (void*)this->gpu_ptr,
                                    total* sizeof(value_type), cudaMemcpyDeviceToHost, stream_singleton::get_gpu2cpu_stream()));
    checkCudaErrors(cudaEventRecord(this->gpu2cpu_event, stream_singleton::get_gpu2cpu_stream()));
}

template <class value_type>
inline bool tensor_t<value_type>::is_cpu_to_gpu_ready() {
    /**
     * check if the async cpu 2 gpu finish.
     * state : CPU2GPU -> GPU
     */
    if (cpu2gpu_event_not_happen.load()) {
        return true;
    }

    check_state(CPU2GPU);

    cudaError_t r = cudaEventQuery(this->cpu2gpu_event);
    if (r == cudaSuccess) {
        cpu2gpu_event_not_happen = true;

        this->atomic_set_state(GPU_FUL);

#ifdef LRU_ON
        if (this->get_type() == DATA) {
            lru->update(this);
        }
#endif

        return true;
    } else if (r == cudaErrorNotReady) {
        return false;
    } else {
        fprintf(stderr, "error when checking cpu2gpu_event, error message : %s\n", cudaGetErrorString(r));
        return false;
    }
}

template <class value_type>
inline bool tensor_t<value_type>::is_gpu_to_cpu_ready() {
    /**
     * check if async gpu 2 cpu finish.
     * state : GPU2CPU -> GPU
     */
    if (gpu2cpu_event_not_happen.load()) {
        return true;
    }

    check_state(GPU2CPU);

    cudaError_t r = cudaEventQuery(this->gpu2cpu_event);
    if (r == cudaSuccess) {
        gpu2cpu_event_not_happen = true;

        this->atomic_set_state(GPU_FUL);

        return true;
    } else if (r == cudaErrorNotReady) {
        return false;
    } else {
        fprintf(stderr, "error when checking cpu2gpu_event, error message : %s\n", cudaGetErrorString(r));
        return false;
    }
}

//------GPU functions-----//
template <class value_type> //ptr1 = ptr1 + ptr2
void tensor_sum(value_type* ptr1, value_type* ptr2, int size);

template <class value_type> //copy ptr1 to ptr2
void tensor_copy(value_type* ptr1, value_type* ptr2, int size);

template <class value_type> //ptr1 = ptr1 * s
void tensor_scale(value_type* ptr1, value_type s, int size);
//-----------------------//

template <class value_type>
void tensor_t<value_type>::sum(tensor_t<value_type>* t) {
    size_t len = this->N*this->C*this->H*this->W;
    value_type one = 1.0;
    tensor_sum(this->get_gpu_ptr(), t->get_gpu_ptr(), len);
}

template <class value_type>
value_type tensor_t<value_type>::squared_sum(cublasHandle_t *handle) {
    size_t len = this->N*this->C*this->H*this->W;
    value_type squared_sum = 0;
    value_type result = 0;
    cublas_dot(handle, this->get_scalar_count(), this->get_gpu_ptr(), 1, this->get_gpu_ptr(), 1, &result);
    return result;
}

template <class value_type>
void tensor_t<value_type>::copy(tensor_t<value_type>* t,
                                int src_start_idx, int src_end_idx,
                                int dst_start_idx, int dst_end_idx) {
    size_t len = 0, offset_dst = 0, offset_src = 0;
    if ((src_start_idx == -1) && (src_end_idx == -1) && ( dst_start_idx == -1) && (dst_end_idx == -1)) {
        len = this->N * this->C * this->H * this->W;
    }
    if ((src_start_idx >= 0) && (src_end_idx >= 0)) {
        len = (size_t) (src_end_idx - src_start_idx);
        offset_src = (size_t) src_start_idx;
    }
    if ((dst_start_idx >= 0) && (dst_end_idx >= 0)) {
        if (len != 0) {
            if (len != (size_t)(dst_end_idx - dst_start_idx)) {
                fprintf(stderr, "tensor copy size does not match, src len: %zu, dst len: %d\n", len, dst_end_idx - dst_start_idx);
            }
        } else {
            len = (size_t) (dst_end_idx - dst_start_idx);
        }
        offset_dst = (size_t) dst_start_idx;
    }
    // TODO : this memcpy is with error in loss decrease
//    cudaMemcpy(this->get_gpu_ptr()+offset_dst, t->get_gpu_ptr()+offset_src, len, cudaMemcpyDeviceToDevice);
    tensor_copy(t->get_gpu_ptr()+offset_src, this->get_gpu_ptr()+offset_dst, len);
}

template <class value_type>
void tensor_t<value_type>::scale(value_type s) {
    size_t len = this->N*this->C*this->H*this->W;
    tensor_scale(this->get_gpu_ptr(), s, len);
}



template <class value_type>
void tensor_t<value_type>::hostRegister() {
    if (this->gpu_ptr != NULL) {
        long total = this->N * this->C * this->H * this->W;
        checkCudaErrors( cudaHostRegister(this->cpu_ptr, total*sizeof(value_type), cudaHostRegisterPortable) );
    }
}

#define PRINT_TENSOR
template <class value_type>
void tensor_t<value_type>::printTensor(const char* str) {
#ifdef PRINT_TENSOR
    printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("PRINT OUT TENSOR %p N:%zu C%zu H:%zu W:%zu@:%s\n", this, this->N, this->C, this->H, this->W, str);
    GPUtoCPU();
    for(size_t n = 0; n < this->N; n++) {
        printf("#################### CPU n:%zu ####################\n", n);
        for (size_t c = 0; c < this->C; c++) {
            printf("--------c:%zu--------\n", c);
            for (size_t h = 0; h < this->H; h++) {
                for (size_t w = 0; w < this->W; w++) {
                    //float and double
                    printf(" %3.3f, ", this->cpu_ptr[((n*C+c)*H+h)*W+w]);
                }
                printf("\n");
            }
        }
    }
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
#endif
}

template <class value_type>
void tensor_t<value_type>::printTensorNoDebug(const char* str) {
    printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N, this->C, this->H, this->W, str);
    GPUtoCPU();
    for(size_t n = 0; n < this->N; n++) {
        printf("#################### CPU n:%zu ####################\n", n);
        for (size_t c = 0; c < this->C; c++) {
            printf("--------c:%zu--------\n", c);
            for (size_t h = 0; h < this->H; h++) {
                for (size_t w = 0; w < this->W; w++) {
                    //float and double
                    printf(" %3.5f ", this->cpu_ptr[((n*C+c)*H+h)*W+w]);
                }
                printf("\n");
            }
        }
    }
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
}
    
template <class value_type>
void tensor_t<value_type>::writeToFile(const char* str) {
    printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N, this->C, this->H, this->W, str);
    FILE *fp;
    fp = fopen(str, "a");
    GPUtoCPU();
    for(size_t n = 0; n < this->N; n++) {
        //fprintf(fp, "#################### CPU n:%zu ####################\n", n);
        for (size_t c = 0; c < this->C; c++) {
            //fprintf(fp, "--------c:%zu--------\n", c);
            for (size_t h = 0; h < this->H; h++) {
                for (size_t w = 0; w < this->W; w++) {
                    //float and double
                    fprintf(fp, "%f ", this->cpu_ptr[((n*C+c)*H+h)*W+w]);
                }
                //fprintf(fp, "\n");
            }
        }
    }
    fclose(fp);
}


template <class value_type>
void tensor_t<value_type>::printTensorFirst(const char* str) {
    printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N, this->C, this->H, this->W, str);
//        size_t total = this->N*this->C*this->H*this->W;
        GPUtoCPU();
        for(size_t n = 0; n < 1; n++) {
            printf("#################### CPU n:%zu ####################\n", n);
            for (size_t c = 0; c < this->C; c++) {
                printf("--------c:%zu--------\n", c);
                for (size_t h = 0; h < this->H; h++) {
                    for (size_t w = 0; w < this->W; w++) {
                        //float and double
                        printf(" %2.0f ", this->cpu_ptr[((n*C+c)*H+h)*W+w]);
                    }
                    printf("\n");
                }
            }
        }
        printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
}


template <class value_type>
void tensor_t<value_type>::resizeTensor(size_t n, size_t c, size_t h, size_t w) {
    /**
     * state : not change
     */
    assert(n >= 1);
    assert(c >= 1);
    assert(h >= 1);
    assert(w >= 1);

//    bool flag = this->gpu_ptr != NULL;
    freeSpaceGPU();

//    if (flag) {
        acquireSpaceGPU(n * c * h * w);
//    }

    freeSpaceCPU();

#ifdef LIVENESS
    if (this->data_t != CONV_BUFF) {
        acquireSpaceCPU(n * c * h * w);
    }
#else
    acquireSpaceCPU(n * c * h * w);
#endif

    this->N = n;
    this->C = c;
    this->H = h;
    this->W = w;
    
    CHECK_GT( (int) n, 0);
    CHECK_GT( (int) c, 0);
    CHECK_GT( (int) h, 0);
    CHECK_GT( (int) w, 0);
    
    checkCUDNN( cudnnDestroyTensorDescriptor(cudnn_tensor_desc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&cudnn_tensor_desc) );
    checkCUDNN( cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc,
                                           this->cudnn_tensor_format,
                                           this->cudnn_data_type,
                                           n, c, h, w) );
}

template <class value_type>
value_type tensor_t<value_type>::get_scalar(const size_t n, const size_t c, const size_t h, const size_t w)
{
    assert( n < N );
    assert( c < C );
    assert( h < H );
    assert( w < W );
    GPUtoCPU();
    return (this->cpu_ptr[((n*C+c)*H+h)*W+w]);
}

template <class value_type>
void tensor_t<value_type>::set_scalar(const size_t n, const size_t c, const size_t h, const size_t w, value_type t)
{
    assert( n < N );
    assert( c < C );
    assert( h < H );
    assert( w < W );
    GPUtoCPU();
    this->cpu_ptr[((n*C+c)*H+h)*W+w] = t;
    CPUtoGPU();
}

template <class value_type>
void tensor_t<value_type>::init(initializer_t<value_type> *initializer) {
    initializer->call(this->cpu_ptr, this->N, this->C, this->H, this->W);
    CPUtoGPU();

    // TODO : the initializer should be used only once !!!
//#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
//    delete initializer;
}

template <class value_type>
void tensor_t<value_type>::acquireSpaceGPU(long total) {
    if ( gpu_ptr != NULL) {
        return;
    }
    assert( total > 0 );

//    printf("before malloc %zu byte\n", query_free_mem());
    gmalloc(gpu_malloc, &(this->gpu_ptr), sizeof(value_type)*total);
//    printf("after malloc %zu byte\n", query_free_mem());

    if (this->gpu_ptr != NULL) {
        this->atomic_set_state(GPU_NIL);
    }

    if (data_t != DATA && data_t != CONV_BUFF) {
        return;
    }

//    if (data_t != CONV_BUFF) {
//        into_cnt += 1;
//    }
//
//    if (data_t != CONV_BUFF && this->gpu_ptr != NULL) {
//        hit_cnt += 1;
//    }

#ifdef LRU_ON
//    if (this->gpu_ptr == NULL && data_t != CONV_BUFF) {
//        miss_cnt += 1;
//    }

    while (this->gpu_ptr == NULL) {
        printf("LRU start!! tensor %p, current free memory %zu\n", this, query_free_mem());
        lru->print_list();
        int x = 0;
        while (lru->get_item(x) != NULL) {
            tensor_t<value_type>* t = (tensor_t<value_type>*)lru->get_item(x)->item;
            printf("tensor %p layer %d\n", t, t->get_layer_id());
            x += 1;
        }

        // kick out some tensors in LRU
        tensor_t<value_type> *t = (tensor_t<value_type> *) (lru->remove_oldest());
        if (t == NULL) {
            lru->print_list();
            fprintf(stderr, "LRU NULL !!! Can not alloc GPU memory !!!! tensor %p need %zu free %zu\n", this, total, query_free_mem());
            exit(-1);
        }
        printf("kick out tensor %p layer %d\n", t, t->get_layer_id());
        if (t->get_state() == GPU_FUL) {
            t->GPUtoCPU();
            t->free_gpu_space(CPU);
        } else {
            fprintf(stderr, "Found a not GPU tensor in LRU %p\n", t);
            fprintf(stderr, "tensor: %p, layer %d type: %d, state: %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            lru->print_list();
            exit(-1);
        }

//#ifdef DEBUG
        printf("kick out oldest? \n");
        lru->print_list();
//#endif

        gmalloc(gpu_malloc, &(this->gpu_ptr), sizeof(value_type) * total);
    }

    if (this->gpu_ptr != NULL) {
        if (this->get_state() != GPU_NIL) {
            this->atomic_set_state(GPU_NIL);
        }
        if (this->get_type() == DATA) {
            lru->update(this);
        }
    }

#endif

}

template <class value_type>
void tensor_t<value_type>::freeSpaceGPU(mem_mode target) {
    if (this->cpu_ptr == NULL) {
        this->atomic_set_state(VOID);
    } else {
        this->atomic_set_state(target);
    }

    if (gpu_ptr == NULL) {
        return;
    }
#ifdef DEBUG
    printf("free tensor %p layer %d gpu %p  curt: %d target: %d\n", this, this->get_layer_id(), gpu_ptr, get_state(), target);
#endif

    gfree(gpu_malloc, this->gpu_ptr);
    this->gpu_ptr = NULL;

#ifdef LRU_ON
    if (this->get_type() == DATA) {
        lru->remove_item(lru->find(this));
    }
#endif
}

template <class value_type>
void tensor_t<value_type>::replace_data(value_type *new_cpu_ptr, value_type *new_gpu_ptr) {

    if (new_cpu_ptr != NULL) {
        value_type *old_cpu_ptr = this->cpu_ptr;
        this->cpu_ptr = new_cpu_ptr;
        checkCudaErrors(cudaFreeHost(old_cpu_ptr));

        if (new_gpu_ptr == NULL) {
            CPUtoGPU();
        }
    }

    if (new_gpu_ptr != NULL) {
        value_type *old_gpu_ptr = this->gpu_ptr;
        this->gpu_ptr = new_gpu_ptr;

        // remember to free the old ptr
        checkCudaErrors(cudaFree(old_gpu_ptr));
    }
}
    
/*---math functions-------*/
template <class value_type>
void tensor_t<value_type>::forward_fft() {
    CHECK_EQ(this->data_t, GRAD);
    CHECK_EQ( cufftExecR2C(fft_plan_f, (cufftReal*) this->gpu_ptr, (cufftComplex*) this->freq_ptr ), CUFFT_SUCCESS );
    const size_t total_size = this->get_scalar_count();
}
    
template <class value_type>
void tensor_t<value_type>::backward_fft() {
    CHECK_EQ(this->data_t, GRAD);
    CHECK_EQ( cufftExecC2R(fft_plan_b, (cufftComplex*) this->freq_ptr, (cufftReal*) this->gpu_ptr), CUFFT_SUCCESS );
    const value_type rescale_factor = 1.0f / (value_type) this->get_scalar_count();
    this->scale( rescale_factor );
}

    
template <class value_type>
size_t tensor_t<value_type>::tensor_counter = 0;

INSTANTIATE_CLASS(tensor_t);

} // SuperNeurons namespace
