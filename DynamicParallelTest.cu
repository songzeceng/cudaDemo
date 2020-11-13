//
// Created by songzeceng on 2020/11/13.
// 编译：nvcc DynamicParallelTest.cu -gencode arch=compute_61,code=sm_61 -rdc=true -o DynamicParallelTest

#include "cuda_runtime.h"
#include "stdio.h"

__device__ void printArray(int *data, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d\t", data[i]);
    }
}

__global__ void child_launch(int *data, int n) {
//    data[threadIdx.x] = data[threadIdx.x] + 1000;
    printf("Hello ");
}

__global__ void parent_launch(int *data, int n) {
//    data[threadIdx.x] = threadIdx.x;
//
//    __syncthreads();
//
//    printArray(data, n);
//
//    if (threadIdx.x == 0) {
//        child_launch<<<1, 6>>>(data, n);
//    }
//#endif

    child_launch<<<1, 1>>>(data, n);
    cudaDeviceSynchronize();

    printf(" World!\n");

    int *d = (int *) malloc(10 * sizeof(int));
    int *a = (int *) malloc(20 * sizeof(int));
    int *a_1;
    cudaMalloc(&a_1, 50 * sizeof(int));
    int *a_2;
    cudaMalloc(&a_2, 90 * sizeof(int));

    free(d);
    free(a);
    cudaFree(a_1);
    cudaFree(a_2);

//    printArray(data, n);
//
//    __syncthreads();
}

int main() {
    cudaSetDevice(0);

    int size = 6;
    int *h_data = (int *) malloc(size * sizeof(int));
    int *d_data;

    for (int i = 0; i < size; i++) {
        h_data[i] = 0;
    }
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    parent_launch<<<1, 1>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    free(h_data);
    return 0;
}
