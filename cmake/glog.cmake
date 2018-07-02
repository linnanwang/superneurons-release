##########################################################
# find and set glog

if (NOT GLOG_ROOT)
    message(FATAL_ERROR "Please specify glog root path in config file.")
endif ()

include_directories(${GLOG_ROOT}/include)

if (APPLE)
    set(GLOG_LIBRARIES ${GLOG_ROOT}/lib/libglog.dylib)
else()
    set(GLOG_LIBRARIES ${GLOG_ROOT}/lib/libglog.so)
endif ()

list(APPEND THIRD_LIBS ${GLOG_LIBRARIES})