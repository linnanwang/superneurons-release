#
# Created by ay27 on 17/3/7.
#

# convert mnist testing
add_executable(convert_mnist convert_mnist.cpp ../include/util/image_reader.h)
target_link_libraries(convert_mnist superneurons)

add_executable(convert_cifar10 convert_cifar10.cpp)
target_link_libraries(convert_cifar10 superneurons)

add_executable(convert_cifar100 convert_cifar100.cpp)
target_link_libraries(convert_cifar100 superneurons)

add_executable(compute_image_mean compute_image_mean.cpp)
target_link_libraries(compute_image_mean superneurons)

add_executable(convert_jpeg convert_jpeg.cpp)
target_link_libraries(convert_jpeg superneurons)