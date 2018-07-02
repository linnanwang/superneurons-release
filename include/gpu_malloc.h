#ifndef GPU_MALLOC_H
#define GPU_MALLOC_H
#include <stdio.h>
#include <stdlib.h>
//#include <blasx_common.h>
#include <util/common.h>
#include <map>

#define BLASX_GPU_MEM_MAX_SEGMENT    200
#define BLASX_GPU_INIT_MEM 1000000*50

/*---gpu_malloc---*/
typedef struct segment {
    void  *addr;       /* Address of the first byte of this segment */
    size_t mem_size;   /* Size of memory occupied by this segment */
    size_t mem_free;   /* Size of memory free after this segment */
    struct segment *next;
    struct segment *prev;
} blasx_gpu_segment_t;

typedef struct gpu_malloc_s {
    void                *base;                 /* Base pointer              */
    blasx_gpu_segment_t *allocated_segments;   /* List of allocated segment */
    blasx_gpu_segment_t *free_segments;        /* List of available segment */
    size_t               total_size;           /* total memory size ocupied */
    size_t               free_size;            /* total free memory size    */
    int                  max_segment;          /* Maximum number of segment */
} blasx_gpu_malloc_t;

blasx_gpu_malloc_t* blasx_gpu_malloc_init(int GPU_id);
void   blasx_gpu_malloc_fini(blasx_gpu_malloc_t* gdata, int GPU_id);
void*  blasx_gpu_malloc(blasx_gpu_malloc_t *gdata, size_t nbytes);
void   blasx_gpu_free(blasx_gpu_malloc_t *gdata, void *addr);



class blasx_gpu_singleton {
private:
    std::map<int, blasx_gpu_malloc_t*> malloc_gpus;

    blasx_gpu_singleton() {

    }

    ~blasx_gpu_singleton() {
        for (auto it=malloc_gpus.begin(); it!=malloc_gpus.end(); ++it) {
            blasx_gpu_malloc_fini(it->second, it->first);
        }
    }

    blasx_gpu_malloc_t* _get_blasx(int GPU_id) {
        auto res = malloc_gpus.find(GPU_id);
        if (res != malloc_gpus.end()) {
            return res->second;
        } else {
            blasx_gpu_malloc_t* tmp = blasx_gpu_malloc_init(GPU_id);
            malloc_gpus.insert(std::make_pair(GPU_id, tmp));
            return tmp;
        }
    }

    static blasx_gpu_singleton* instance;

public:

    static blasx_gpu_malloc_t* get_blasx_gpu_malloc_t(int GPU_id) {
        if (instance == NULL) {
            instance = new blasx_gpu_singleton();
        }
        return instance->_get_blasx(GPU_id);
    }

    static void destroy_all_instance() {
        delete instance;
    }
};

#endif /* GPU_MALLOC_H */

