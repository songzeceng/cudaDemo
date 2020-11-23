//
// Created by root on 2020/11/11.
//

#include "cuda_runtime.h"
#include "iostream"

__global__ void addMatrix(int* a, int* b, int* c, int nx, int ny) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = 0;
    for (; idy < ny; idy++) {
        int index = idy * nx + idx;
        c[index] = a[index] + b[index];
    }
}

int main() {
    int x = 5, y = 2;
    int size = x * y * sizeof(int);

    int *a = (int*) malloc(size);
    int *b = (int*) malloc(size);
    int *c = (int*) malloc(size);

    for (int i = 0; i < x * y; i++) {
        a[i] = i * 2;
        b[i] = i + 1;
    }

    int* h_a;
    int* h_b;
    int* h_c;
    cudaMalloc(&h_a, size);
    cudaMalloc(&h_b, size);
    cudaMalloc(&h_c, size);

    cudaMemcpy(h_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_b, b, size, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((x + block.x - 1) / block.x, (y + block.y - 1) / block.y);

    addMatrix<<<grid, block>>>(h_a, h_b, h_c, x, y);

    cudaMemcpy(c, h_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < x * y; i++) {
        std::cout << c[i] << std::endl;
    }

    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
