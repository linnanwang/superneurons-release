##########################################################
# find and set cuda and cudnn

# find and include cuda
find_package(CUDA)
if (NOT CUDA_FOUND)
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT_DIR})
    set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    if (APPLE)
        set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcudart.dylib)
        set(CUDA_CUBLAS_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcublas.dylib)
		set(CUDA_CUFFT_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcufft.dylib)
    else ()
        set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)
        set(CUDA_CUBLAS_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so)
		set(CUDA_CUFFT_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft.so)
    endif (APPLE)
endif (NOT CUDA_FOUND)
include_directories("${CUDA_INCLUDE_DIRS}")

# find and include cudnn
if (APPLE)
    message("Platform is Darwin")
    include_directories("${CUDNN_ROOT_DIR}/include")
    set(CUDNN_LIBRARIES "${CUDNN_ROOT_DIR}/lib/libcudnn.dylib")
    set(CMAKE_MACOSX_RPATH 1)   # fix macosx rpath warning
else ()
    message("Platform is Linux")
    include_directories("${CUDNN_ROOT_DIR}/include")
    set(CUDNN_LIBRARIES "${CUDNN_ROOT_DIR}/lib64/libcudnn.so")
endif (APPLE)

# end of finding cuda and cudnn
##########################################################

# detch architectures, set arch and code
set(arch_detect_file ${PROJECT_ROOT_DIR}/cmake/arch_detect.cu)

# detect_out is a list with format code1;code2;code3;...
execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${arch_detect_file}"
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
        RESULT_VARIABLE detect_res OUTPUT_VARIABLE detect_out
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)


SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS -std=c++11)
message("CUDA NVCC FLAGS : ${CUDA_NVCC_FLAGS}")

if ((${detect_res} EQUAL -1) OR (${detect_res} EQUAL 255))
    message("detect architectures failed, using default setting of nvcc")
else()
    message("detect architectures ${detect_out}")
    set(__flags "")
    set(ARCH_CODE "")
    foreach(__arch_code ${detect_out})
        list(APPEND __flags "-gencode arch=compute_${__arch_code},code=sm_${__arch_code} ")
        list(APPEND ARCH_CODE "${__arch_code} ")
    endforeach()
    list(APPEND CUDA_NVCC_FLAGS "${__flags} ")
endif ()

list(APPEND THIRD_LIBS ${CUDA_LIBRARIES})
list(APPEND THIRD_LIBS ${CUDA_CUBLAS_LIBRARIES})
list(APPEND THIRD_LIBS ${CUDNN_LIBRARIES})
list(APPEND THIRD_LIBS ${CUDA_CUFFT_LIBRARIES})
