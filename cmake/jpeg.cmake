##########################################################
# read libjpeg lib

find_package(JPEG)
if (JPEG_FOUND)
    message("Found libjpeg")
else()
    message("do not found libjpeg auto, using path from config file")
    set(JPEG_INCLUDE_DIR ${LIBJPEG_INCLUDE_DIR})
    set(JPEG_LIBRARIES ${LIBJPEG_LIB_FILE})
endif ()

include_directories(${JPEG_INCLUDE_DIR})
list(APPEND THIRD_LIBS ${JPEG_LIBRARIES})