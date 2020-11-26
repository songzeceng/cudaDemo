//
// Created by root on 2020/11/26.
//

#include "kernel.h"
#include "cuda_runtime.h"

#define TX 32
#define TY 32
#define LEN 5.f
#define TIME_STEP 0.005f
#define FINAL_TIME 10.f

__device__ float scale(int i, int w) {
    return 2 * LEN * (((1.f * i) / w) - 0.5f);
}

__device__ float f(float x, float y, float param, float sys) {
    if (sys == 1) {
        return x - 2 * param * y;
    }
    if (sys == 2) {
        return -x + param * (1 - x * x) * y;
    }
    return -x - 2 * param * y;
}

__device__ float2 euler(float x, float y, float dt, float tFinal, float param, float sys) {
    float dx = 0.f, dy = 0.f;
    for (float t = 0; t < tFinal; t += dt) {
        dx = dt * y;
        dy = dt * f(x, y , param, sys);
        x += dx;
        y += dy;
    }
    return make_float2(x, y);
}

__device__ char clip(float n) {
    return n < 0 ? 0 : n > 255 ? 255 : n;
}

__global__ void stabImageKernel(uchar4 *d_out, int w, int h, float p, int s) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = y * w  + x;

    if (x >= w || y >= h) {
        return;
    }

    float x0 = scale(x, w);
    float y0 = scale(y, h);
    float dist_0 = sqrt(x0 * x0 + y0 * y0);
    float2 pos = euler(x0, y0, TIME_STEP, FINAL_TIME, p, s);
    float dist_f = sqrt(pos.x * pos.x + pos.y * pos.y);
    float dist_r = dist_f / dist_0;

    d_out[idx].x = clip(dist_r * 255);
    d_out[idx].y = (x == w / 2 || y == h / 2) ? 255 : 0;
    d_out[idx].z = clip((1 / dist_r) * 255);
    d_out[idx].w = 255;
}

void kernelLauncher(uchar4 *d_out, int w, int h, float p, int s) {
    dim3 block(TX, TY);
    dim3 grid((w + TX - 1) / TX, (h + TY - 1) / TY);

    stabImageKernel<<<grid, block>>>(d_out, w, h, p, s);
}