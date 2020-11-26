//
// Created by root on 2020/11/26.
//

#include "cuda_runtime.h"
#include "stdio.h"

#define W 50
#define H 50
#define TX 32
#define TY 32

__device__ char clip(int n) {
    return n < 0 ? 0 : n > 255 ? 255 : n;
}

__global__ void distanceKernel(uchar4 *d_out, int w, int h, float2 pos) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = x + y * w;

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

int main() {
    int size = W * H;
    int nBytes = size * sizeof(uchar4);
    uchar4 *out = (uchar4 *) malloc(nBytes);
    uchar4 *d_out;
    cudaMalloc(&d_out, nBytes);

    float2 pos = {0.0f, 0.0f};
    dim3 block(TX, TY);
    dim3 grid((W + TX - 1) / TX, (H + TY - 1) / TY);

    distanceKernel<<<grid, block>>>(d_out, W, H, pos);
    cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            printf("(%d, %d, %d, %d)\t",
                   out[i * W + j].x, out[i * W + j].y, out[i * W + j].z, out[i * W + j].w);
        }
        printf("\n");
    }

    free(out);
    cudaFree(d_out);

    return 0;
}