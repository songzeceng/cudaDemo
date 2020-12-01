//
// Created by root on 2020/11/30.
//

#include "kernel.cuh"
#include "stdio.h"

#define TPB 64
#define ATOMIC 1 // 0 for non-atomic add

__global__ void dotKernel(int *d_res, int *d_a, int *d_b, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) {
        return;
    }

    int s_dix = threadIdx.x;
    __shared__ int s_prod[TPB];
    s_prod[s_dix] = d_a[idx] * d_b[idx];
    __syncthreads();

    if (s_dix == 0) {
        int blockSum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            blockSum += s_prod[i];
        }
        printf("Block %d, block sum = %d\n", blockIdx.x, blockSum);
        if (ATOMIC) {
            atomicAdd(d_res, blockSum);
        } else {
            *d_res += blockSum;
        }
    }
}

void dotLauncher(int *res, int *a, int *b, int n) {
    int *d_res;
    int *d_a = 0, *d_b = 0;

    cudaMalloc(&d_res, sizeof(int ));
    cudaMalloc(&d_a, n * sizeof(int ));
    cudaMalloc(&d_b, n * sizeof(int ));

    cudaMemset(d_res, 0, sizeof(int ));
    cudaMemcpy(d_a, a, n * sizeof(int ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int ), cudaMemcpyHostToDevice);

    dotKernel<<<(n + TPB - 1) / TPB, TPB>>>(d_res, d_a, d_b, n);
    cudaMemcpy(res, d_res, sizeof(int ), cudaMemcpyDeviceToHost);

    cudaFree(d_res);
    cudaFree(d_a);
    cudaFree(d_b);
}