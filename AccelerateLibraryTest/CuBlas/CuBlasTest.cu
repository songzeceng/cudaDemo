//
// Created by root on 2020/11/24.
//

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "stdio.h"

int M = 1024;
int N = 1024;

void generate_random_vector(int n, float** outX) {
    int i = 0;
    double rMax = (double ) RAND_MAX;
    float* X = (float *) malloc(n * sizeof(float ));
    for (; i < n; i++) {
        X[i] = ((double ) rand()) / rMax * 100.0;
    }
    *outX = X;
}

void generate_random_dense_matrix(int m, int n, float** outX) {
    int i = 0, j = 0;
    double rMax = (double ) RAND_MAX;
    float *A = (float *) malloc(m * n * sizeof(float ));

    for (; j < n; j++) {
        for (; i < m; i++) {
            int r = rand();
            float* curr = A + (j * m + i);
            if (r % 3 > 0) {
                *curr = 0.0f;
            } else {
                *curr = ((double ) r) / rMax * 100;
            }
        }
    }
    *outX = A;
}

//  nvcc -lcublas -o CuBlasTest CuBlasTest.cu
int main() {
    int i;
    float *A, *dA, *X, *dX, *Y, *dY;
    float alpha = 3.0f, beta=4.0f;
    cublasHandle_t handle;

    // allocate memory
    A = (float *) malloc(sizeof(float ) * M * N);
    X = (float *) malloc(sizeof(float ) * N);
    Y = (float *) malloc(sizeof(float ) * M);
    cudaMalloc(&dA, sizeof(float ) * M * N);
    cudaMalloc(&dX, sizeof(float ) * N);
    cudaMalloc(&dY, sizeof(float) * M);

    // generate input
    srand(3432);
    generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    generate_random_vector(M, &Y);

    // create cublas handle
    cublasCreate(&handle);

    // get cublas data from original input
    cublasSetMatrix(M, N, sizeof(float), A, M, dA, N);
    cublasSetVector(N, sizeof(float ), X, 1, dX, 1);
    cublasSetVector(M, sizeof(float ), Y, 1, dY, 1);

    // perform matrix multiplication y = alpha * op(A) * x  + beta * y
    cublasSgemv_v2(handle, CUBLAS_OP_N, M, N, &alpha, dA, M, dX, 1, &beta, dY, 1);

    // get result from cublas and print demo data
    cublasGetVector(M, sizeof(float ), dY, 1, Y, 1);

    for (i = 0; i < 10; i++) {
        printf("%.2f\t", Y[i]);
    }
    printf("\n");

    // free memory
    free(A);
    free(X);
    free(Y);
    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
    return 0;
}