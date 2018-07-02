#if !defined(_COMMON_H_)
#define _COMMON_H_

#include <map>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <switch.h>
#include <utility>
#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include <assert.h>
#include <cublas_v2.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <util/superneurons_math.h>

#define LIVENESS
//#define RECOMPUTE_ON
//#define LARGER
//#define LRU_ON
//#define BLASX_MALLOC
//#define BENCHMARK

#define BLASX_GPU_MEM_SIZE  (1024L*1024L*1000L*3L)


typedef std::pair<int, int> d_key_t;

typedef enum net_comp {
    FORWARD  = 0,
    BACKWARD = 1
} net_comp;

typedef enum network_stage {
    NET_TRAIN   = 0,
    NET_INFER   = 1
} network_stage;

typedef enum data_mode {
    DATA_TRAIN   = 0,
    DATA_TEST    = 1
} data_mode;

typedef enum join_mode {
    ELEWISE_SUM  = 0,
    ELEWISE_MAX  = 1
} join_mode;

typedef enum structure_type {
    FORK = 0,
    JOIN = 1,
} structure_type;

typedef enum mem_mode {
    VOID      = 0,
    GPU_NIL   = 1,      // gpu with invalid data
    GPU_FUL   = 2,      // gpu with valid data
    CPU       = 3,
    CPU2GPU   = 4,
    GPU2CPU   = 5,
    RECOMPUTE = 6
} mem_mode;

typedef enum LAYER {
    /*---network layers---*/
    CONV    = 0,
    POOL    = 1,
    ACT     = 2,
    BN      = 3,
    FC      = 4,
    LRN     = 5,
    PADDING = 6,
    DATA_L  = 7,
    DROPOUT = 8,
    SOFTMAX = 9,
    /*--structure layers--*/
    CONCAT  = 10,
    FORK_L  = 11,
    JOIN_L  = 12
} LAYER;


#define INSTANTIATE_CLASS(classname) \
char gInstantiationGuard##classname; \
template class classname<float>; \
template class classname<double>

#endif // _COMMON_H_
