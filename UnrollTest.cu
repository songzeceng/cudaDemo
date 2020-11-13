//
// Created by root on 2020/11/12.
//

#include "cuda_runtime.h"
#include "stdio.h"

__global__ void unrollTestKernel(int *count) {
#pragma unroll 4
    for (int i = 0; i < 20; i++) {
        (*count)++;
    }
}

int main() {
    int *n_h = (int *) malloc(sizeof(int ) );
    *n_h = 0;
    int *h_d;
    cudaMalloc(&h_d, sizeof(int ));
    cudaMemcpy(h_d, n_h, sizeof(int ), cudaMemcpyHostToDevice);

    unrollTestKernel<<<1, 1>>>(h_d);

    cudaMemcpy(n_h, h_d, sizeof(int ), cudaMemcpyDeviceToHost);

    printf("count = %d\n", *n_h); // 20

    cudaFree(h_d);
    free(n_h);

    return 0;
}
