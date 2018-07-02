//
// Created by ay27 on 8/5/17.
//

#include <gpu_malloc.h>

int main()
{
    int i;
    blasx_gpu_malloc_t *mymem0 = blasx_gpu_singleton::get_blasx_gpu_malloc_t(0);
    blasx_gpu_malloc_t *mymem1 = blasx_gpu_singleton::get_blasx_gpu_malloc_t(1);
    printf("mymem success\n");

    float *A = (float *)blasx_gpu_malloc(mymem0, sizeof(float)*11);
    float *B = (float *)blasx_gpu_malloc(mymem0, sizeof(float)*7);
    float *C = (float *)blasx_gpu_malloc(mymem0, sizeof(float)*12345);
    blasx_gpu_free(mymem0, (void *)A);
    blasx_gpu_free(mymem0, (void *)B);
    blasx_gpu_free(mymem0, (void *)C);
    float *A_prior = NULL;
    float *B_prior = NULL;
    float *C_prior = NULL;

    for (i = 0; i < 20; i++) {
        int SIZE = sizeof(float)*11*13*(i+1);
        printf("==>i:%d malloc SIZE:%d\n", i, SIZE);
        A = (float *)blasx_gpu_malloc(mymem1, sizeof(float)*SIZE);
        B = (float *)blasx_gpu_malloc(mymem1, sizeof(float)*SIZE);
        C = (float *)blasx_gpu_malloc(mymem1, sizeof(float)*SIZE);

        cudaMemcpy(((char*)B)+1, A+2, 3, cudaMemcpyDeviceToDevice);

        if (A == NULL || B == NULL || C == NULL) {
            blasx_gpu_free(mymem1, (void*) A_prior);
            blasx_gpu_free(mymem1, (void*) B_prior);
            blasx_gpu_free(mymem1, (void*) C_prior);
            printf("not enought mem: free A B C\n");
        } else {
            printf("A:%p B:%p C:%p\n", A, B, C);
            printf("\n");
        }
        A_prior = A;
        B_prior = B;
        C_prior = C;

        cudaDeviceSynchronize();
    }

    blasx_gpu_singleton::destroy_all_instance();
    return 0;
}
