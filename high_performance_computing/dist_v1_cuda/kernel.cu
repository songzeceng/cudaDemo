//
// Created by songzeceng on 2020/11/26.
//
#include "stdio.h"
#include "cuda_runtime.h"

#define N 64
#define TPB 32

__device__ float scale(int i, int n) {
    return ((float ) i) / (n - 1);
}

__device__ float distance(float x1, float x2) {
    return sqrt((x2 - x1) * (x2 - x1));
}

__global__ void distanceKernel(float *d_out, float ref, int len) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float x = scale(i, len);
    d_out[i] = distance(x, ref);
}

int main() {
    float ref = 0.5f;

    float *d_out;
    float *h_out = (float *) malloc(N * sizeof(float ));
    cudaMalloc(&d_out, N * sizeof(float ));

    distanceKernel<<<N / TPB, TPB>>>(d_out, ref, N);

    cudaMemcpy(h_out, d_out, N * sizeof(float ), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%.2f\t", h_out[i]);
    }
    printf("\n");

    free(h_out);
    cudaFree(d_out);

    return 0;
}
