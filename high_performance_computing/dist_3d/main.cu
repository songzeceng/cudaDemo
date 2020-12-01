//
// Created by root on 2020/12/1.
//

#include "stdio.h"
#include "stdlib.h"
#include "cuda_runtime.h"

#define W 32
#define H 32
#define D 32
#define TX 8
#define TY 8
#define TZ 8

int divUp(int a, int b) {
    return (a + b - 1) / b;
}

__device__ float distance(int c, int r, int s, float3 pos) {
    return sqrtf((c - pos.x) * (c - pos.x) + (r - pos.y) * (r - pos.y) + (s - pos.z) * (s - pos.z));
}

__global__ void distanceKernel(float *d_out, int w, int h, int d, float3 pos) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int s = blockDim.z * blockIdx.z + threadIdx.z;
    int i = c + r * w + s * w * h;
    if (c >= w || r >= h || s >= d) {
        return;
    }
    d_out[i] = distance(c, r, s, pos);
}

int main() {
    float *out = (float *) malloc(W * H * D * sizeof(float ));
    float *d_out;
    cudaMalloc(&d_out, W * H * D * sizeof(float ));

    float3 pos = {0.0f, 0.0f, 0.0f};
    dim3 block(TX, TY, TZ);
    dim3 grid(divUp(W, TX), divUp(H, TY), divUp(D, TZ));
    distanceKernel<<<grid, block>>>(d_out, W, H, D, pos);
    cudaMemcpy(out, d_out, W * H * D * sizeof(float ), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    free(out);
    return 0;
}