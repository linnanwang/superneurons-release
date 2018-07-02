//
// Created by ay27 on 17/6/21.
//

#ifndef SUPERNEURONS_PRINT_UTIL_H
#define SUPERNEURONS_PRINT_UTIL_H

#include <stdio.h>
#include <string>

template <class value_type>
void print_array(value_type *dat, size_t N, size_t C, size_t H, size_t W, const char* msg=NULL) {
    printf("\n=========================================\n");
    if (msg != NULL) {
        printf("%s", msg);
        printf("\n");
    }
    printf("shape : N=%lu, C=%lu, H=%lu, W=%lu\n", N, C, H, W);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            printf("------ n=%lu, c=%lu ------\n", n, c);
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    printf("%6.2f ", dat[((n * C + c) * H + h) * W + w]);
                }
                printf("\n");
            }
        }
    }
    printf("=========================================\n");
}

template <class value_type>
void print_array(value_type *dat, size_t N, size_t C, size_t H, size_t W, std::string msg) {
    print_array(dat, N,C,H,W, msg.data());
}

#endif //SUPERNEURONS_PRINT_UTIL_H
