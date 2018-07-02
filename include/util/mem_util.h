//
// Created by ay27 on 9/5/17.
//

#ifndef SUPERNEURONS_MEM_UTIL_H
#define SUPERNEURONS_MEM_UTIL_H

#include <util/common.h>
#include <gpu_malloc.h>
#include <util/error_util.h>

namespace SuperNeurons {

#define BYTE_TO_MB(_size_in_byte) (((double)(_size_in_byte)) / 1024.0 / 1024.0)

inline size_t query_free_mem() {
#ifdef BLASX_MALLOC
    size_t mem_free = blasx_gpu_singleton::get_blasx_gpu_malloc_t(0)->free_size;
    return mem_free;
#else
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, &mem_tot_0);
    return mem_free_0;
#endif
}

inline size_t query_used_mem() {
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, &mem_tot_0);
    return mem_tot_0 - mem_free_0;
}

inline void _gmalloc(blasx_gpu_malloc_t* blasx, void** ptr, size_t size_in_byte) {
#ifdef BLASX_MALLOC
    *ptr = blasx_gpu_malloc(blasx, size_in_byte);
#else
    cudaMalloc(ptr, size_in_byte);

    // we use this function to clear the error state in cuda runtime.
    cudaGetLastError();
#endif
}

template <class T>
inline void gmalloc(blasx_gpu_malloc_t* blasx, T** ptr, size_t size_in_byte) {
    _gmalloc(blasx, (void**)ptr, size_in_byte);
}


inline void _gfree(blasx_gpu_malloc_t* blasx, void* ptr) {
#ifdef BLASX_MALLOC
    blasx_gpu_free(blasx, ptr);
#else
    checkCudaErrors( cudaFree(ptr) );
#endif
}

template <class T>
inline void gfree(blasx_gpu_malloc_t* blasx, T* ptr) {
    _gfree(blasx, (void*)ptr);
}

} // namespace SuperNeurons

#endif //SUPERNEURONS_MEM_UTIL_H
