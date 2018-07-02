#include <stdio.h>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return -1;
    if (count == 0) return -1;
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            printf("%d%d;", prop.major, prop.minor);
        }
    }
    return 0;
}
