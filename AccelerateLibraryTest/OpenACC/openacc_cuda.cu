//
// Created by root on 2020/11/24.
//

#include "stdio.h"
#include "stdlib.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"

#define M 10
#define N 10
#define P 10

// nvcc openacc_cuda.cu -o openacc_cuda -lcurand -lcublas
int main() {
    float *__restrict__ d_A, *__restrict__ d_B, *__restrict__ d_C;
    float *d_row_sums;
    float total_sum;
    curandGenerator_t rand_state;
    cublasHandle_t cublas_handle;

    curandCreateGenerator(&rand_state, CURAND_RNG_PSEUDO_DEFAULT);
    cublasCreate_v2(&cublas_handle);
    printf("curand and cublas created success!\n");

    cudaMalloc((void **) &d_A, M * N * sizeof(float));
    cudaMalloc((void **) &d_B, N * P * sizeof(float));
    cudaMalloc((void **) &d_C, M * P * sizeof(float));
    cudaMalloc((void **) &d_row_sums, M * sizeof(float ));
    printf("Memory allocated successfully on device!\n");

    curandGenerateUniform(rand_state, d_A, M * N);
    curandGenerateUniform(rand_state, d_B, N * P);
    printf("Data initialized successfully on device!\n");

#pragma acc parallel loop gang deviceptr(d_A, d_B, d_C) // cannot put device data directly into openacc area for openacc 20.7
    {
        for (int i = 0; i < M; i++) { // A row
#pragma acc loop worker vector
            {
                for (int j = 0; j < P; j++) { // B col
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) { // A col and B row
//                        printf("sum for (%d, %d) at %d\n", i, j, k);
                        sum += d_A[i * N + k] * d_B[k * P + j]; // segmentation fault here
                    }
                    d_C[i * P + j] = sum;
                }
            }
        }
    }
    printf("Matrix multiplication and add performed successfully!\n");

    cublasSetPointerMode_v2(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);

    for (int i = 0; i < M; i++) {
        cublasSasum_v2(cublas_handle, P, d_C + i * P, 1, d_row_sums + i);
    }

    cublasSetPointerMode_v2(cublas_handle, CUBLAS_POINTER_MODE_HOST);

    cublasSasum_v2(cublas_handle, M, d_row_sums, 1, &total_sum);
    cudaDeviceSynchronize();
    printf("cublas sum for each row successfully!\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    printf("Memory free successfully on device!\n");

    printf("sum = %.4f\n", total_sum);

    return 0;
}