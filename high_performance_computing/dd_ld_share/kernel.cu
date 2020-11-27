//
// Created by songzeceng on 2020/11/26.
//

#include "cuda_runtime.h"
#include "kernel.h"
#include <stdio.h>

#define TPB 128
#define RAD 1

__global__ void ddKernel(float *d_out, float *d_in, int size, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }

    int s_idx = threadIdx.x;
    extern __shared__ float s_in[];

    s_in[s_idx] = d_in[i];

    __syncthreads();
    if (threadIdx.x > 0) {
        float value = (s_in[s_idx - 1] - 2.f * s_in[s_idx] + s_in[s_idx + 1]) / (h * h);
        d_out[i] = value;
    }
}

void ddParallel(float *out, float *in, int n, float h) {
    float *d_in, *d_out;
    int nBytes = n * sizeof(float );
    cudaMalloc(&d_in, nBytes);
    cudaMalloc(&d_out, nBytes);
    cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice);

    ddKernel<<<(n + TPB - 1) / TPB, TPB, (TPB + RAD) * sizeof(float )>>>(d_out, d_in, n, h);

    cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}