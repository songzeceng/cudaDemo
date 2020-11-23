//
// Created by root on 2020/11/19.
//

#include "stdio.h"
#include <cuda_runtime.h>

#define DIM 128

__global__ void reduceGmem(int *g_idata, int *g_odata, int n) {
    int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n) {
        return;
    }

    // if current thread id is less than half of block dim, reduce in place
    if (blockDim.x >= 1024 && tid < 512) {
        idata[tid] += idata[tid + 512];
    }

    if (blockDim.x >= 512 && tid < 256) {
        idata[tid] += idata[tid + 256];
    }

    if (blockDim.x >= 256 && tid < 128) {
        idata[tid] += idata[tid + 128];
    }

    if (blockDim.x >= 128 && tid < 64) {
        idata[tid] += idata[tid + 64];
    }
    __syncthreads();

    // unrolling warp into the first thread of this warp
    if (tid < 32) {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32]; // I only applied block dim = 128, so tid + 64 has been reduced, but tid + 32 not
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result stored in thread 0 into output
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceGmemUnrolling4(int *g_idata, int *g_odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x; // one thread per block processes 4 data

    int *idata = g_idata + blockDim.x * blockIdx.x * 4;

    if (idx >= n) {
        return;
    }

    // process 4 data per thread
    int a = 0, b = 0, c = 0, d = 0;
    a = g_idata[idx];
    if (idx + blockDim.x < n) {
        b = g_idata[idx + blockDim.x];
    }
    if (idx + 2 * blockDim.x < n) {
        c = g_idata[idx + blockDim.x * 2];
    }
    if (idx + 3 * blockDim.x < n) {
        d = g_idata[idx + blockDim.x * 3];
    }
    g_idata[idx] = a + b + c + d;

    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) {
        idata[tid] += idata[tid + 512];
    }
    if (blockDim.x >= 512 && tid < 256) {
        idata[tid] += idata[tid + 256];
    }
    if (blockDim.x >= 256 && tid < 128) {
        idata[tid] += idata[tid + 128];
    }
    if (blockDim.x >= 128 && tid < 64) {
        idata[tid] += idata[tid + 64];
    }

    __syncthreads();

    if (tid < 32) {
        volatile int *s_vmem = idata;
        s_vmem[tid] += s_vmem[tid + 32];
        s_vmem[tid] += s_vmem[tid + 16];
        s_vmem[tid] += s_vmem[tid + 8];
        s_vmem[tid] += s_vmem[tid + 4];
        s_vmem[tid] += s_vmem[tid + 2];
        s_vmem[tid] += s_vmem[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceSMemUnrolling4(int *g_idata, int *g_odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + tid;

    if (idx >= n) {
        return;
    }

    extern __shared__ int smem[]; // I use dynamic shared memory to reduce data

    int a = 0, b = 0, c = 0, d = 0; // Each thread still processes 4 data
    a = g_idata[idx];
    if (idx + blockDim.x < n) {
        b = g_idata[idx + blockDim.x];
    }
    if (idx + 2 * blockDim.x < n) {
        c = g_idata[idx + 2 * blockDim.x];
    }
    if (idx + 3 * blockDim.x < n) {
        d = g_idata[idx + 3 * blockDim.x];
    }
    smem[tid] = a + b + c + d; // Save result of 4 data into shared memory
    __syncthreads();

    // Reduce data for block using shared memory
    if (blockDim.x >= 1024 && tid < 512) {
        smem[tid] += smem[tid + 512];
    }
    if (blockDim.x >= 512 && tid < 256) {
        smem[tid] += smem[tid + 256];
    }
    if (blockDim.x >= 256 && tid < 128) {
        smem[tid] += smem[tid + 128];
    }
    if (blockDim.x >= 128 && tid < 64) {
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        volatile int* s_vmem = smem;
        s_vmem[tid] += s_vmem[tid + 32];
        s_vmem[tid] += s_vmem[tid + 16];
        s_vmem[tid] += s_vmem[tid + 8];
        s_vmem[tid] += s_vmem[tid + 4];
        s_vmem[tid] += s_vmem[tid + 2];
        s_vmem[tid] += s_vmem[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = smem[0];
    }
}

void test(int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += i;
    }
    printf("Target is %d\n", sum);
}

int main() {
    int size = 1 << 22;
    int blockSize = DIM;

    test(size); // verify the result

    dim3 blockDim(blockSize);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    printf("grid:(%d), block:(%d)\n", gridDim.x, blockDim.x);

    int nBytes = size * sizeof(int);
    int *h_idata = (int *) malloc(nBytes);
    int *h_odata = (int *) malloc(gridDim.x * sizeof(int));
    // Valid output per block is stored in the first thread of each block.
    // So the number of output to be added is equal to the grid dim

    int *d_odata;
    int *d_idata;

    cudaMalloc(&d_idata, nBytes);
    cudaMalloc(&d_odata, gridDim.x * sizeof(int));

    for (int i = 0; i < size; i++) {
        h_idata[i] = i;
    }

    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    reduceGmem<<<gridDim, blockDim>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_odata, d_odata, gridDim.x * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < gridDim.x; i++) {
        sum += h_odata[i];
    }
    printf("\n=========\n");
    printf("sum = %d\n", sum);

    memset(h_odata, 0, gridDim.x * sizeof(int));
    cudaMemset(d_odata, 0, gridDim.x * sizeof(int));
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

    dim3 gridDim_(gridDim.x / 4);
    reduceGmemUnrolling4<<<gridDim_, blockDim>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, gridDim_.x * sizeof(int), cudaMemcpyDeviceToHost);
    sum = 0;
    for (int i = 0; i < gridDim_.x; i++) {
        sum += h_odata[i];
    }
    printf("\n=========\n");
    printf("sum = %d\n", sum);

    memset(h_odata, 0, gridDim.x * sizeof(int));
    cudaMemset(d_odata, 0, gridDim.x * sizeof(int));
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

    reduceSMemUnrolling4<<<gridDim_, blockDim, DIM * sizeof(int )>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, gridDim_.x * sizeof(int), cudaMemcpyDeviceToHost);
    sum = 0;
    for (int i = 0; i < gridDim_.x; i++) {
        sum += h_odata[i];
    }
    printf("\n=========\n");
    printf("sum = %d\n", sum);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}