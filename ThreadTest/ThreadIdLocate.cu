//
// Created by root on 2020/11/11.
//

#include "cuda_runtime.h"
#include <stdio.h>

__global__ void LocateThreadIdKernel() {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

//    printf("%d, %d. %d\n", threadIdx.x, threadIdx.y, threadIdx.z);

    printf("Thread coordinate: (%d, %d, %d)\n", x, y, z);
}

int main () {
    int x = 10, y = 15, z = 20;
    dim3 block(2, 3, 4);
    dim3 grid((x + block.x - 1) / block.x, (y + block.y - 1) / block.y, (z + block.z - 1) / block.z);
    LocateThreadIdKernel<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}