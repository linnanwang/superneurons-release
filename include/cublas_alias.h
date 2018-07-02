//
// Created by ay27 on 7/18/17.
//

#ifndef SUPERNEURONS_CUBLAS_ALIAS_H
#define SUPERNEURONS_CUBLAS_ALIAS_H

#include <stdio.h>
#include <cstdlib>
#include <cublas_v2.h>
#include <util/error_util.h>

template <class value_type>
void cublas_axpy(cublasHandle_t *handle,
                 int n,
                 const value_type *alpha,  /* host or device pointer */
                 const value_type *x,
                 int incx,
                 value_type *y,
                 int incy) {
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "data type not supported so far!@cublas_alias.h line 19\n");
        exit(1);
    } else if (sizeof(value_type) == 4) {
        checkCublasErrors(cublasSaxpy(*handle, n, (const float*)alpha, (const float*)x, incx, (float*)y, incy));
    } else if (sizeof(value_type) == 8) {
        checkCublasErrors(cublasDaxpy(*handle, n, (const double*)alpha, (const double*)x, incx, (double*)y, incy));
    } else {
        fprintf(stderr, "data type not supported so far!@cublas_alias.h line 26\n");
        exit(1);
    }
}

template <class value_type>
void cublas_dot(cublasHandle_t *handle,
                 int n,
                 const value_type* x,
                 int incx,
                 const value_type* y,
                 int incy,
                 value_type* result) {
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "data type not supported so far!@cublas_alias.h line 19\n");
        exit(1);
    } else if (sizeof(value_type) == 4) {
        checkCublasErrors(cublasSdot(*handle, n, (const float*) x, incx, (const float*) y, incy, (float*) result));
    } else if (sizeof(value_type) == 8) {
        checkCublasErrors(cublasDdot(*handle, n, (const double*) x, incx, (const double*) y, incy, (double*) result) );
    } else {
        fprintf(stderr, "data type not supported so far!@cublas_alias.h line 26\n");
        exit(1);
    }
}

template <class value_type>
void cublas_scal(cublasHandle_t *handle,
                 int n,
                 const value_type *alpha,  /* host or device pointer */
                 value_type *x,
                 int incx) {
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "data type not supported so far!@cublas_alias.h line 41\n");
        exit(1);
    } else if (sizeof(value_type) == 4) {
        checkCublasErrors(cublasSscal(*handle, n, (const float*)alpha, (float*)x, incx));
    } else if (sizeof(value_type) == 8) {
        checkCublasErrors(cublasDscal(*handle, n, (const double*)alpha, (double*)x, incx));
    } else {
        fprintf(stderr, "data type not supported so far!@cublas_alias.h line 48\n");
        exit(1);
    }
}

template <class value_type>
void cublas_gemv(cublasHandle_t* handle,
                 cublasOperation_t trans,
                 int m, int n, const value_type *alpha,
                 const value_type *A, int lda,
                 const value_type *x, int incx,
                 const value_type *beta,
                 value_type *y, int incy) {
    if (sizeof(value_type) == 2) {
        fprintf(stderr, "data type not supported so far!@cublas_alias.h line 41\n");
        exit(1);
    } else if (sizeof(value_type) == 4) {
        checkCublasErrors(cublasSgemv(*handle, trans,
                                      m, n, (const float*) alpha,
                                      (const float*) A, lda,
                                      (const float*) x, incx,
                                      (const float*) beta,
                                      (float*) y, incy ));
    } else if (sizeof(value_type) == 8) {
        checkCublasErrors(cublasDgemv(*handle, trans,
                                      m, n, (const double*) alpha,
                                      (const double*) A, lda,
                                      (const double*) x, incx,
                                      (const double*) beta,
                                      (double*) y, incy ));

    } else {
        fprintf(stderr, "data type not supported so far!@cublas_alias.h line 48\n");
        exit(1);
    }
}

#endif //SUPERNEURONS_CUBLAS_ALIAS_H
