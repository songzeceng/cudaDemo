//
// Created by root on 2020/11/23.
//

#include "stdio.h"
#include "cuda_runtime.h"
#include "cusparse_v2.h"

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

int generate_random_dense_matrix(int m, int n, float** outX) {
    int i = 0, j = 0;
    double rMax = (double ) RAND_MAX;
    float *A = (float *) malloc(m * n * sizeof(float ));
    int totalNnz = 0;

    for (; j < n; j++) {
        for (; i < m; i++) {
            int r = rand();
            float* curr = A + (j * m + i);
            if (r % 3 > 0) {
                *curr = 0.0f;
            } else {
                *curr = ((double ) r) / rMax * 100;
            }

            if (*curr != 0.0f) {
                totalNnz++;
            }
        }
    }
    *outX = A;
    return totalNnz;
}

//  nvcc -lcusparse -o cusparse cuSparseTest.cu
int main() {
    int totalNnz;
    float *A, *dA, *dCsrValAm, *X, *Y, *dX, *dY;
    int *dNnzPerRow, *dCsrRowPtrA, *dCsrColIndA;
    float alpha = 3.0f, beta = 4.0f;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    // Allocate host memory
    A = (float *) malloc(sizeof(float ) * M * N);
    X = (float *) malloc(sizeof(float ) * N);
    Y = (float *) malloc(sizeof(float ) * M);

    // generate input
    srand(3325);
    int trueNnz = generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    generate_random_vector(M, &Y);

    // create cusparse handle
    cusparseCreate(&handle);

    // allocate device memory
    cudaMalloc(&dX, sizeof(float ) * N);
    cudaMalloc(&dY, sizeof(float ) * M);
    cudaMalloc(&dA, sizeof(float ) * M * N);
    cudaMalloc(&dNnzPerRow, sizeof(float ) * M);

    // create matrix descriptor
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // copy data to device
    cudaMemcpy(dX, X, sizeof(float ) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, sizeof(float ) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, A, sizeof(float ) * M * N, cudaMemcpyHostToDevice);

    // compute the number of non-zero elements in each row of A
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dA, M, dNnzPerRow, &totalNnz);

    if (totalNnz == trueNnz) {
        printf("Non-zero element count matches which is %d!\n", totalNnz);
    }

    // allocate memory for csr vectors
    cudaMalloc(&dCsrValAm, sizeof(float ) * totalNnz);
    cudaMalloc(&dCsrRowPtrA, sizeof(int ) * (M + 1));
    cudaMalloc(&dCsrColIndA, sizeof(int ) * totalNnz);

    // get csr vectors from matrix in device memory
    cusparseSdense2csr(handle, M, N, descr, dA, M, dNnzPerRow, dCsrValAm, dCsrRowPtrA, dCsrColIndA);

    // y = alpha * op(A) * x  + beta * y, where op is non-transpose here.
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, totalNnz, &alpha, descr,
                   dCsrValAm, dCsrRowPtrA, dCsrColIndA, dX, &beta, dY);

    // get result from device and show first 10 of it
    cudaMemcpy(Y, dY, sizeof(float ) * M, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%.2f\t", Y[i]);
    }
    printf("\n");

    // free resource
    free(A);
    free(X);
    free(Y);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dA);
    cudaFree(dNnzPerRow);
    cudaFree(dCsrValAm);
    cudaFree(dCsrRowPtrA);
    cudaFree(dCsrColIndA);

    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);

    return 0;
}
