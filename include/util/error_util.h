/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#if !defined(_ERROR_UTIL_H_)
#define _ERROR_UTIL_H_

#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#define TOSTR_(s)   #s
#define TOSTR(s)    TOSTR_(s)
#if defined(__GNUC__)
#define COMPILER_NAME "GCC"
#define COMPILER_VER  TOSTR(__GNUC__) "." TOSTR(__GNUC_MINOR__) "." TOSTR(__GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#if _MSC_VER < 1500
#define COMPILER_NAME "MSVC_2005"
#elif _MSC_VER < 1600
#define COMPILER_NAME "MSVC_2008"
#elif _MSC_VER < 1700
#define COMPILER_NAME "MSVC_2010"
#elif _MSC_VER < 1800
#define COMPILER_NAME "MSVC_2012"
#elif _MSC_VER < 1900
#define COMPILER_NAME "MSVC_2013"
#elif _MSC_VER < 2000
#define COMPILER_NAME "MSVC_2014"
#else
#define COMPILER_NAME "MSVC"
#endif
#define COMPILER_VER  TOSTR(_MSC_FULL_VER) "." TOSTR(_MSC_BUILD)
#elif defined(__clang_major__)
#define COMPILER_NAME "CLANG"
#define COMPILER_VER  TOSTR(__clang_major__ ) "." TOSTR(__clang_minor__) "." TOSTR(__clang_patchlevel__)
#elif defined(__INTEL_COMPILER)
#define COMPILER_NAME "ICC"
#define COMPILER_VER TOSTR(__INTEL_COMPILER) "." TOSTR(__INTEL_COMPILER_BUILD_DATE)
#else
#define COMPILER_NAME "unknown"
#define COMPILER_VER  "???"
#endif

#define CUDNN_VERSION_STR  TOSTR(CUDNN_MAJOR) "." TOSTR (CUDNN_MINOR) "." TOSTR(CUDNN_PATCHLEVEL)

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
      std::stringstream _where, _message;                                \
      _where << __FILE__ << ':' << __LINE__;                             \
      _message << _error.str() + "\n" << __FILE__ << ':' << __LINE__;\
      std::cerr << _message.str() << "\nAborting...\n";                  \
      cudaDeviceReset();                                                 \
      exit(EXIT_FAILURE);                                                \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
      std::stringstream _where, _message;                                \
      _where << __FILE__ << ':' << __LINE__;                             \
      _message << _error.str() + "\n" << __FILE__ << ':' << __LINE__;\
      std::cerr << _message.str() << "\nAborting...\n";                  \
      cudaDeviceReset();                                                 \
      exit(EXIT_FAILURE);                                                \
    }                                                                  \
}

#define checkCublasErrors(status) {                                    \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cublas failure\nError code " << status;        \
      std::stringstream _where, _message;                                \
      _where << __FILE__ << ':' << __LINE__;                             \
      _message << _error.str() + "\n" << __FILE__ << ':' << __LINE__;\
      std::cerr << _message.str() << "\nAborting...\n";                  \
      cudaDeviceReset();                                                 \
      exit(EXIT_FAILURE);                                                \
    }                                                                  \
}

// CUDA Utility Helper Functions

static void  showDevices( void )
{
    int totalDevices;
    checkCudaErrors(cudaGetDeviceCount( &totalDevices ));
    printf("\nThere are %d CUDA capable devices on your machine :\n", totalDevices);
    for (int i=0; i< totalDevices; i++) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, i ));
        printf( "device %d : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",
                    i, prop.multiProcessorCount, prop.major, prop.minor,
                    (float)prop.clockRate*1e-3,
                    (int)(prop.totalGlobalMem/(1024*1024)),
                    (float)prop.memoryClockRate*1e-3,
                    prop.ECCEnabled,
                    prop.multiGpuBoardGroupID);
    }
}

static double get_cur_time() {
    struct timeval   tv;
    struct timezone  tz;
    double cur_time;
    gettimeofday(&tv, &tz);
    cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
    return cur_time;
}


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#else // Linux Includes
#include <string.h>
#include <strings.h>
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#endif
inline int stringRemoveDelimiter(char delimiter, const char *string)
{
    int string_start = 0;

    while (string[string_start] == delimiter)
    {
        string_start++;
    }

    if (string_start >= (int)strlen(string)-1)
    {  
        return 0;
    }

    return string_start;
}

#endif // _ERROR_UTIL_H_
