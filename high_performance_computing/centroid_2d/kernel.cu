//
// Created by root on 2020/11/30.
//

#include "kernel.cuh"
#include "stdio.h"
#include <helper_math.h>

#define TPB 64

__global__ void centroidKernel(uchar4 *d_img, int *d_centroidCol, int *d_centroidRow, int *d_pixelCount,
                               int width, int height) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int s_idx = threadIdx.x;
    int row = idx / width;
    int col = idx % width;

    __shared__ uint4 s_img[TPB];

    if ((d_img[idx].x < 255 || d_img[idx].y < 255 || d_img[idx].z < 255) && idx < width * height) {
        s_img[s_idx].x = col;
        s_img[s_idx].y = row;
        s_img[s_idx].z = 1;
    } else {
        s_img[s_idx].x = 0;
        s_img[s_idx].y = 0;
        s_img[s_idx].z = 0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (s_idx < s) {
            s_img[s_idx] += s_img[s_idx + s];
            __syncthreads();
        }
    }

    if (s_idx == 0) {
        atomicAdd(d_centroidCol, s_img[0].x);
        atomicAdd(d_centroidRow, s_img[0].y);
        atomicAdd(d_pixelCount, s_img[0].z);
    }
}

void centroidParallel(uchar4 *img, int width, int height) {
    uchar4 *d_img = 0;
    int *d_centroidRow = 0, *d_centroidCol = 0, *d_pixelCount = 0;
    int centroidRow = 0, centroidCol = 0, pixelCount = 0;

    cudaMalloc(&d_img, width * height * sizeof(uchar4));
    cudaMemcpy(d_img, img, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMalloc(&d_centroidRow, sizeof(int ));
    cudaMalloc(&d_centroidCol, sizeof(int ));
    cudaMalloc(&d_pixelCount, sizeof(int ));
    cudaMemset(d_centroidRow, 0, sizeof(int ));
    cudaMemset(d_centroidCol, 0, sizeof(int ));
    cudaMemset(d_pixelCount, 0, sizeof(int ));

    centroidKernel<<<(width * height + TPB - 1) / TPB, TPB>>>(d_img, d_centroidCol, d_centroidRow,
                                                              d_pixelCount, width, height);
    cudaMemcpy(&centroidCol, d_centroidCol, sizeof(int ), cudaMemcpyDeviceToHost);
    cudaMemcpy(&centroidRow, d_centroidRow, sizeof(int ), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pixelCount, d_pixelCount, sizeof(int ), cudaMemcpyDeviceToHost);

    centroidCol /= pixelCount;
    centroidRow /= pixelCount;
    printf("Centroid: (col: %d, row: %d) based on %d pixels\n", centroidCol, centroidRow, pixelCount);

    for (int col = 0; col < width; col++) {
        img[centroidRow * width + col].x = 255;
        img[centroidRow * width + col].y = 0;
        img[centroidRow * width + col].z = 0;
    }
    for (int row = 0; row < height; row++) {
        img[row * width + centroidCol].x = 255;
        img[row * width + centroidCol].y = 0;
        img[row * width + centroidCol].z = 0;
    }

    cudaFree(d_img);
    cudaFree(d_centroidRow);
    cudaFree(d_centroidCol);
    cudaFree(d_pixelCount);
}