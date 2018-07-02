#
# Created by ay27 on 17/3/7.
#

# link a shared library
CUDA_ADD_LIBRARY(superneurons SHARED ${SOURCE_FILES} ${CUDA_FILES})
target_link_libraries(superneurons ${THIRD_LIBS} -lpthread -lm)
#target_link_libraries(superneurons ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${GLOG_LIBRARIES})
