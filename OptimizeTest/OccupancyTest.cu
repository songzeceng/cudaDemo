//
// Created by songzeceng on 2020/11/11.
//

#include "stdio.h"
#include "cuda_runtime.h"

void getOccupancy();

__global__ void MyKernel(int* a, int* b, int* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    c[idx] = a[idx] + b[idx];
}

__global__ void MyKernel2(int* a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        a[idx] = a[idx] + 3;
    }
}

int main() {
//    getOccupancy();

    int blockSize, minGridSize, gridSize;
    int size;

    scanf("%d", &size);

    int *h_data = (int *) malloc(size * sizeof(int));
    int *d_data;

    for (int i = 0; i < size; i++) {
        h_data[i] = i * i;
    }
    cudaMalloc(&d_data, size * sizeof(int ));
    cudaMemcpy(d_data, h_data, size * sizeof(int ), cudaMemcpyHostToDevice);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)MyKernel2, 0, size);

    gridSize = (size + blockSize - 1) / blockSize;
    MyKernel2<<<gridSize, blockSize>>>(d_data, size);

    cudaMemcpy(h_data, d_data, size * sizeof(int ), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
       printf("%d\n", h_data[i]);
    }

    cudaFree(d_data);
    free(h_data);
    return 0;
}

void getOccupancy() {
    int numBlocks;
    int blockSize = 32;
    int device = 0, activeWraps, maxWarps;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, MyKernel, blockSize, 0);

    activeWraps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    printf("Occupancy: %.3f\n", activeWraps / (double ) maxWarps);
}
