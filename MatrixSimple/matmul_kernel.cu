#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace std;

typedef struct {
    int width;
    int height;
    float* data;
} Matrix;

#define BLOCK_SIZE 2

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void showMatrix(Matrix m);

void MatMaul(const Matrix A, const Matrix B, Matrix C) {
    Matrix dA;
    dA.height = A.height;
    dA.width = A.width;
    size_t size = dA.width * dA.height * sizeof(float);
    cudaMalloc(&dA.data, size);
    cudaMemcpy(dA.data, A.data, size, cudaMemcpyHostToDevice);

    Matrix dB;
    dB.height = B.height;
    dB.width = B.width;
    size = dB.width * dB.height * sizeof(float);
    cudaMalloc(&dB.data, size);
    cudaMemcpy(dB.data, B.data, size, cudaMemcpyHostToDevice);

    Matrix dC;
    dC.height = C.height;
    dC.width = C.width;
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
    float value = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < A.width; i++) {
        value += A.data[row * A.width + i] * B.data[i * B.width + col];
    }

    C.data[row * C.width + col] = value;
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
    Matrix A;
    Matrix B;
    Matrix C;

    A.height = 4;
    A.width = 2;
    A.data = (float *)malloc(A.width * A.height * sizeof(float));
    for (int i = 0; i < A.height * A.width; i++) {
        A.data[i] = i + 1;
    }

    B.height = 2;
    B.width = 4;
    B.data = (float *)malloc(B.width * B.height * sizeof(float));
    for (int i = 0; i < B.height * B.width; i++) {
        B.data[i] = 2 * i;
    }

    C.width = A.height;
    C.height = B.width;
    C.data = (float *)malloc(C.width * C.height * sizeof(float));

    showMatrix(A);
    showMatrix(B);

    MatMaul(A, B, C);
    return 0;
}
