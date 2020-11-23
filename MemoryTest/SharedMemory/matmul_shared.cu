#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace std;

typedef struct {
    int width;
    int height;
    int stride; // width of sub matrix
    float* data;
} Matrix;

#define BLOCK_SIZE 2

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

__device__ float GetElement(const Matrix A, int row, int col) {
    return A.data[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.data[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix sub;

    sub.width = BLOCK_SIZE;
    sub.height = BLOCK_SIZE;
    sub.stride = A.stride;
    sub.data = &A.data[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return sub;
}

void showMatrix(Matrix m);

void matrixMultimal();

void MatMaul(const Matrix A, const Matrix B, Matrix C) {
    Matrix dA;
    dA.height = A.height;
    dA.width = A.width;
    dA.stride = A.width;
    size_t size = dA.width * dA.height * sizeof(float);
    cudaMalloc(&dA.data, size);
    cudaMemcpy(dA.data, A.data, size, cudaMemcpyHostToDevice);

    Matrix dB;
    dB.height = B.height;
    dB.width = B.width;
    dB.stride = B.width;
    size = dB.width * dB.height * sizeof(float);
    cudaMalloc(&dB.data, size);
    cudaMemcpy(dB.data, B.data, size, cudaMemcpyHostToDevice);

    Matrix dC;
    dC.height = C.height;
    dC.width = C.width;
    dC.stride = C.width;
    size = dC.width * dC.height * sizeof(float);
    cudaMalloc(&dC.data, size);
//    cudaMemcpy(dC.data, C.data, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    MatMulKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);

    cudaMemcpy(C.data, dC.data, size, cudaMemcpyDeviceToHost);

    cudaFree(dA.data);
    cudaFree(dB.data);
    cudaFree(dC.data);

    showMatrix(C);
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    float value = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int i = 0; i < A.width / BLOCK_SIZE; i++) {
        Matrix Asub = GetSubMatrix(A, blockRow, i);
        Matrix Bsub = GetSubMatrix(B, i, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            value += As[row][j] * Bs[j][col];
        }

        __syncthreads();
    }

    SetElement(Csub, row, col, value);
}

void showMatrix(Matrix m) {
    cout << "size(" << m.height << ", " << m.width << ")" << endl;
    for (int i = 0; i < m.height; i++) {
        for (int j = 0; j < m.width; j++) {
            cout << m.data[i * m.width + j] << "\t";
        }
        cout << endl;
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
    }

    matrixMultimal();

    return 0;
}

void matrixMultimal() {
    Matrix A;
    Matrix B;
    Matrix C;

    A.height = 4;
    A.width = 4;
    A.stride = BLOCK_SIZE;
    A.data = (float *)malloc(A.width * A.height * sizeof(float));
    for (int i = 0; i < A.height * A.width; i++) {
        A.data[i] = i + 1;
    }

    B.height = 4;
    B.width = 4;
    B.stride = BLOCK_SIZE;
    B.data = (float *)malloc(B.width * B.height * sizeof(float));
    for (int i = 0; i < B.height * B.width; i++) {
        B.data[i] = 2 * i;
    }

    C.width = A.height;
    C.height = B.width;
    C.stride = BLOCK_SIZE;
    C.data = (float *)malloc(C.width * C.height * sizeof(float));

    showMatrix(A);
    showMatrix(B);

    MatMaul(A, B, C);
}
