//
// Created by songzeceng on 2020/11/26.
//

#include "cuda_runtime.h"
#include "kernel.h"

#define TPB 32

__device__ float distance(float x1, float x2) {
    return sqrt((x2 - x1) * (x2 - x1));
}

__global__ void distanceKernel(float *d_out, float *d_in, float ref) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float x = d_in[i];
    d_out[i] = distance(x, ref);
}

void distanceArray(float *out, float *in, float ref, int len) {
    float *d_in;
    float *d_out;
    cudaMalloc(&d_in, len * sizeof(float ));
    cudaMalloc(&d_out, len * sizeof(float ));
    cudaMemcpy(d_in, in, len * sizeof(float ), cudaMemcpyHostToDevice);

    distanceKernel<<<len / TPB, TPB>>>(d_out, d_in, ref);

    cudaMemcpy(out, d_out, len * sizeof(float ), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}