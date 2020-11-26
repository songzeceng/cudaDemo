//
// Created by songzeceng on 2020/11/26.
//

#include "cuda_runtime.h"
#include "stdio.h"

#define N 64
#define TPB 32

float scale(int i, int n) {
    return ((float ) i) / (n - 1);
}

__device__ float distance(float x1, float x2) {
    return sqrt((x2 - x1) * (x2 - x1));
}

__global__ void distanceKernel(float *d_out, float *d_in, float ref) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float x = d_in[i];
    d_out[i] = distance(x, ref);
}

int main() {
    float ref = 0.5f;
    float *in;
    float *out;

    cudaMallocManaged(&in, N * sizeof(float ));
    cudaMallocManaged(&out, N * sizeof(float ));

    for (int i = 0; i < N; ++i) {
        in[i] = scale(i, N);
    }

    distanceKernel<<<N / TPB, TPB>>>(out, in, ref);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i) {
        printf("%.2f\t", out[i]);
    }
    printf("\n");

    cudaFree(in);
    cudaFree(out);
    return 0;
}