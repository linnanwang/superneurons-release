#
# Created by ay27 on 17/3/7.
#

CUDA_ADD_EXECUTABLE(alexnet alexnet.cpp)
target_link_libraries(alexnet superneurons)

CUDA_ADD_EXECUTABLE(lenet lenet.cpp)
target_link_libraries(lenet superneurons)

#CUDA_ADD_EXECUTABLE(datareader_test datareader_test.cpp
#        ../include/util/image_reader.h
#        ../include/util/base_reader.h
#        ../include/util/parallel_reader.h
#        ../include/util/binary_dumper.h
#        ../include/util/thread_routine.h
#        ../include/util/tensor_queue.h
#        ../include/util/mem_util.h
#        ../include/liveness.h
#        ../include/recompute.h
#        ../include/mem_control.h
#        ../include/network.h
#        )
#target_link_libraries(datareader_test superneurons)

#CUDA_ADD_EXECUTABLE(saver_test
#        ../include/util/saver.h
#        ../include/solver.h
#        ../src/solver.cpp
#        ../include/layer/batch_normalization_layer.h
#        ../include/layer/concat_layer.h
#        ../include/layer/cudnn_activation_layer.h
#        ../include/layer/cudnn_convolution_layer.h
#        ../include/layer/cudnn_pooling_layer.h
#        ../include/layer/data_layer.h
#        ../include/layer/dropout_layer.h
#        ../include/layer/fork_layer.h
#        ../include/layer/fully_connected_layer.h
#        ../include/layer/join_layer.h
#        ../include/layer/local_response_norm_layer.h
#        ../include/layer/padding_layer.h
#        ../include/layer/softmax_layer.h
#        ../include/layer/base_layer.h)
#target_link_libraries(saver_test superneurons)

#CUDA_ADD_EXECUTABLE(test_lru test_lru.cpp ../include/util/lru.h)
#target_link_libraries(test_lru superneurons)

CUDA_ADD_EXECUTABLE(residual_32_torch residual_32_torch.cpp)
target_link_libraries(residual_32_torch superneurons)


#CUDA_ADD_EXECUTABLE(residual_32_nobn residual_32_nobn.cpp)
#target_link_libraries(residual_32_nobn superneurons)

#CUDA_ADD_EXECUTABLE(fork_join_test fork_join_test.cpp)
#target_link_libraries(fork_join_test superneurons)

#CUDA_ADD_EXECUTABLE(tensor_test tensor_test.cpp ../include/tensor.h)
#target_link_libraries(tensor_test superneurons)


CUDA_ADD_EXECUTABLE(cifar10 cifar10.cpp)
target_link_libraries(cifar10 superneurons)

CUDA_ADD_EXECUTABLE(cifar cifar.cpp)
target_link_libraries(cifar superneurons)

CUDA_ADD_EXECUTABLE(residual_32 residual_32.cpp)
target_link_libraries(residual_32 superneurons)

CUDA_ADD_EXECUTABLE(residual_imagenet residual_imagenet.cpp)
target_link_libraries(residual_imagenet superneurons)

CUDA_ADD_EXECUTABLE(inception_v4 inception_v4.cpp)
target_link_libraries(inception_v4 superneurons)

CUDA_ADD_EXECUTABLE(vgg16 vgg16.cpp)
target_link_libraries(vgg16 superneurons)

#CUDA_ADD_EXECUTABLE(forkfork forkfork.cpp ../include/layer/cudnn_convolution_layer.h)
#target_link_libraries(forkfork superneurons)

#CUDA_ADD_EXECUTABLE(blasx_test blasx_test.cpp)
#target_link_libraries(blasx_test superneurons)

CUDA_ADD_EXECUTABLE(test_malloc_free_speed test_malloc_free_speed.cpp)
target_link_libraries(test_malloc_free_speed superneurons)

CUDA_ADD_EXECUTABLE(residual_imagenet_50 residual_imagenet_50.cpp)
target_link_libraries(residual_imagenet_50 superneurons)

CUDA_ADD_EXECUTABLE(residual_imagenet_101 residual_imagenet_101.cpp)
target_link_libraries(residual_imagenet_101 superneurons)

CUDA_ADD_EXECUTABLE(residual_imagenet_152 residual_imagenet_152.cpp)
target_link_libraries(residual_imagenet_152 superneurons)

CUDA_ADD_EXECUTABLE(residual_imagenet_256 residual_imagenet_256.cpp)
target_link_libraries(residual_imagenet_256 superneurons)

CUDA_ADD_EXECUTABLE(residual_imagenet_512 residual_imagenet_512.cpp)
target_link_libraries(residual_imagenet_512 superneurons)

CUDA_ADD_EXECUTABLE(residual_imagenet_1024 residual_imagenet_1024.cpp)
target_link_libraries(residual_imagenet_1024 superneurons)


CUDA_ADD_EXECUTABLE(residual_imagenet_max residual_imagenet_max.cpp)
target_link_libraries(residual_imagenet_max superneurons)

#CUDA_ADD_EXECUTABLE(cuda_err_test cuda_err_test.cpp)

add_subdirectory(unittest)