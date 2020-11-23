//
// Created by root on 2020/11/20.
//

#include "stdio.h"
#include "cuda_runtime.h"

#define BDIM 32
#define RADIUS 4

#define a0     0.00000f
#define a1     0.80000f
#define a2    -0.20000f
#define a3     0.03809f
#define a4    -0.00357f

__constant__ float coef[RADIUS + 1];

// constant memory is 64KB for each processor, which is good at uniform read
__global__ void stencil_ld(float *in, float *out) {
    __shared__ float smem[BDIM + 2 * RADIUS];

    int idx = threadIdx.x + blockIdx.x * blockDim.x; // index in global memory

    int sidx = threadIdx.x + RADIUS; // index in shared memory

    smem[sidx] = in[idx]; // thread index + R is the medium data

    if (threadIdx.x < RADIUS) {
        // First four threads get data from thread index(left) and thread index + R + dim(right) into shared memory
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM] = in[idx + BDIM];
    }

    __syncthreads();

    // calculate stencil
    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i <= RADIUS; i++) {
        tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
    }

    out[idx] = tmp;
}

// restrict memory is 48KB for each processor, which is only suitable for scatter read
__global__ void stencil_ld_readonly(float *in, float *out, float *__restrict__ dcoef) {
    __shared__ float smem[BDIM + 2 * RADIUS];

    int idx = threadIdx.x + blockIdx.x * blockDim.x; // index in global memory

    int sidx = threadIdx.x + RADIUS; // index in shared memory

    smem[sidx] = in[idx]; // thread index + R is the medium data

    if (threadIdx.x < RADIUS) {
        // First four threads get data from thread index(left) and thread index + R + dim(right) into shared memory
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM] = in[idx + BDIM];
    }

    __syncthreads();

    // calculate stencil
    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i <= RADIUS; i++) {
        tmp += dcoef[i] * (smem[sidx + i] - smem[sidx - i]);
    }

    out[idx] = tmp;
}

void setup_coef() {
    const float h_coef[] = {a0, a1, a2, a3, a4};
    cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float));
}

int main() {
    int isize = 16;

    size_t nBytes = (isize + 2 * RADIUS) * sizeof(float);

    // allocate host memory
    float *h_in = (float *) malloc(nBytes);
    float *hostRef = (float *) malloc(nBytes);
    float *gpuRef = (float *) malloc(nBytes);

    float *d_in, *d_out, *d_coef;
    cudaMalloc((float **) &d_in, nBytes);
    cudaMalloc((float **) &d_out, nBytes);
    cudaMalloc((float **) &d_coef, (RADIUS + 1) * sizeof(float ));

    for (int i = 0; i < isize + 2 * RADIUS; i++) {
        h_in[i] = (float) i;
    }

    cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

    setup_coef();

    dim3 block(BDIM, 1);
    dim3 grid((isize + block.x - 1) / block.x, 1);

    stencil_ld<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < isize + 2 * RADIUS; i++) {
        printf("%f->", gpuRef[i]);
    }
    printf("\n========\n");

    cudaMemset(d_out, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    const float h_coef[] = {a0, a1, a2, a3, a4};
    cudaMemcpy(d_coef, h_coef, (RADIUS + 1) * sizeof(float ), cudaMemcpyHostToDevice);

    stencil_ld_readonly<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS, d_coef);
    cudaDeviceSynchronize();
    cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < isize + 2 * RADIUS; i++) {
        printf("%f->", gpuRef[i]);
    }

    return 0;
}