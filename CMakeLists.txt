cmake_minimum_required(VERSION 2.8)

##########################################################
# set the version number
set(superneurons_VERSION_MAJOR 0)
set(superneurons_VERSION_MINOR 1)
set(PROJECT_NAME SuperNeurons)

##########################################################
# set project structure
SET(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})
SET(SOURCE_DIR src)
SET(HEADER_DIR include)
SET(TESTING_DIR testing)

set(THIRD_LIBS "")

##########################################################
# include third part package
include(cmake/config.cmake)
include(cmake/cuda.cmake)
include(cmake/jpeg.cmake)
include(cmake/glog.cmake)

##########################################################
## set debug or release mode, default is debug

macro(mark_debug)
    message("Debug mode")
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")
    set(CMAKE_BUILD_TYPE Debug)
endmacro()

macro(mark_release)
    message("Release mode")
    add_definitions(-DRELEASE)
    set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_BUILD_TYPE Release)
endmacro()

# It's strange that it will arise an error when using if with OR statement.
# if ((NOT CMAKE_BUILD_TYPE) OR (${CMAKE_BUILD_TYPE} STREQUAL Debug))
if (NOT CMAKE_BUILD_TYPE)
    mark_release()
elseif (${CMAKE_BUILD_TYPE} STREQUAL Debug)
    mark_debug()
elseif (${CMAKE_BUILD_TYPE} STREQUAL Release)
    mark_release()
endif ()


#define LIVENESS
#define RECOMPUTE_ON
#define LARGER
#define LRU_ON
#define BLASX_MALLOC
#define BENCHMARK

if (DEFINED LIVENESS)
    ADD_DEFINITIONS(-DLIVENESS)
endif ()
if (DEFINED RECOMPUTE_ON)
    ADD_DEFINITIONS(-DRECOMPUTE_ON)
endif ()
if (DEFINED LARGER)
    ADD_DEFINITIONS(-DLARGER)
endif ()
if (DEFINED LRU_ON)
    ADD_DEFINITIONS(-DLRU_ON)
endif ()
if (DEFINED BLASX_MALLOC)
    ADD_DEFINITIONS(-DBLASX_MALLOC)
endif ()



##########################################################
# set compiler
SET(CMAKE_CXX_COMPILER "g++")

#if (${SUPPRESS_WARNING} STREQUAL yes)
#    message("supress warning")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
#    list(APPEND CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets)
#    list(APPEND CUDA_NVCC_FLAGS -w)
#endif ()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -W -Wno-unused-parameter")
SET(CMAKE_C_FLAGS " -std=c11")
##########################################################
# find all header files and source files
file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CUDA_FILES "${SOURCE_DIR}/*.cu")
file(GLOB_RECURSE HEADER_FILES "${HEADER_DIR}/*.h")

list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_DIR}/comm_engine.cpp")
list(REMOVE_ITEM HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_DIR}/comm_engine.h")

message("${CUDA_FILES}")

# include them
include_directories(${HEADER_DIR})
include_directories(${SOURCE_DIR})
include_directories(/usr/include)
include_directories(/usr/local/include)


##########################################################
# add src and testing
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(tools)

ADD_CUSTOM_TARGET(debug
        COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
        COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
        )

ADD_CUSTOM_TARGET(release
        COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
        COMMENT "Switch CMAKE_BUILD_TYPE to Release"
        )

##########################################################
# print message
macro(status text)
    message(STATUS ${text})
endmacro()

status("")
status("***************** SuperNeurons Configuration Summary *****************")
status("Genernal:")
status("  Version           :   ${superneurons_VERSION_MAJOR}.${superneurons_VERSION_MINOR}")
status("  System            :   ${CMAKE_SYSTEM_NAME}")
status("  C++ complier      :   ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")
status("  Build type        :   ${CMAKE_BUILD_TYPE}")
status("  CXX flags         :   ${CMAKE_CXX_FLAGS}")
#status("  Glog path         :   ${GLOG_ROOT}")
status("  link flags        :   ${THIRD_LIBS}")
status("")
status("CUDA:")
status("  CUDA path         :   ${CUDA_TOOLKIT_ROOT_DIR}")
status("  CUDNN path        :   ${CUDNN_ROOT_DIR}")
status("  NVCC flags        :   ${CUDA_NVCC_FLAGS}")
status("")
status("Target GPUs:")
status("  arch code         :   ${ARCH_CODE}")
status("libjpeg:")
status("  include dir       :   ${JPEG_INCLUDE_DIR}")
status("  libraries         :   ${JPEG_LIBRARIES}")
status("******************************** end *********************************")
