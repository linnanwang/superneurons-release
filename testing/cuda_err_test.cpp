//
// Created by ay27 on 9/22/17.
//

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

using namespace std;

int main() {

    int *x;
    cudaError_t er1 = cudaMalloc((void**)&x, 15 * 1024 * 1024 * 1024);
    printf("first return code: %d\n", er1);

    try {
        cudaError_t er = cudaMalloc((void**)&x, 10);
        printf("er = %d\n", er);
    } catch (...) {
        printf("catch exception\n");
    }

    cudaError_t er2 = cudaGetLastError();
    printf("reset err code, %d, %d\n", er2, cudaGetLastError());

    try {
        cudaError_t er3 = cudaMalloc((void**)&x, 10);
        printf("er3 = %d\n", er3);
    } catch (...) {
        printf("catch exception\n");
    }

}