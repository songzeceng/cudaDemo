//
// Created by root on 2020/11/26.
//

#include "kernel.h"
#include "cuda_runtime.h"

#define TX 32
#define TY 32

__device__ char clip(int n) {
    return n < 0 ? 0 : n > 255 ? 255 : n;
}

__global__ void distanceKernel(uchar4 *d_out, int w, int h, int2 pos) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = y * w  + x;

    if (x >= w || y >= h) {
        return;
    }

    int d = sqrtf((x - pos.x) * (x - pos.x) + (y - pos.y) * (y - pos.y));
    char intensity = clip(255 - d);

    d_out[idx].x = intensity;
    d_out[idx].y = intensity;
    d_out[idx].z = 0;
    d_out[idx].w = 255;
}

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos) {
    dim3 block(TX, TY);
    dim3 grid((w + TX - 1) / TX, (h + TY - 1) / TY);

    distanceKernel<<<grid, block>>>(d_out, w, h, pos);
}