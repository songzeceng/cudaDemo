//
// Created by root on 2020/11/23.
//

#include "stdio.h"
#include "cuda_runtime.h"

#define NSTREAM 4

#define n_repeat 32

__global__ void sumArrays(float *A, float *B, float *C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        for (int  i = 0; i < n_repeat; i++) {
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main() {
    int nElem = 256;
    int nBytes = nElem * sizeof(float);

    float *gpu_ref;
    float *host_ref;
    float *d_A;
    float *d_B;
    float *h_A;
    float *h_B;

    cudaHostAlloc((void **) &host_ref, nBytes, cudaHostAllocDefault); // Async copy needs pinned host memory
    cudaMalloc(&gpu_ref, nBytes);
    cudaMalloc(&d_A, nBytes);
    cudaMalloc(&d_B, nBytes);
    cudaHostAlloc((void **) &h_A, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void **) &h_B, nBytes, cudaHostAllocDefault);

    for (int i = 0; i < nElem; i++) {
        h_A[i] = i;
        h_B[i] = i + 1;
    }

    int iElem = nElem / NSTREAM;
    int iBytes = iElem * sizeof(float );
    cudaStream_t *streams = (cudaStream_t *) malloc(NSTREAM * sizeof(cudaStream_t));
    for (int i = 0; i < NSTREAM; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(1);
    dim3 grid(iElem);

//    // Deep first
//    for (int i = 0; i < NSTREAM; i++) {
//        int ioffset = i * iElem;
//        cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, streams[i]);
//        cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, streams[i]);
//        sumArrays<<<grid, block, 0, streams[i]>>>(&d_A[ioffset], &d_B[ioffset], &gpu_ref[ioffset], iElem);
//        cudaMemcpyAsync(&host_ref[ioffset], &gpu_ref[ioffset], iBytes, cudaMemcpyDeviceToHost, streams[i]);
//    }

    // Breadth first
    for (int i = 0; i < NSTREAM; i++) {
        int ioffset = i * iElem;
        cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, streams[i]);
    }
    for (int i = 0; i < NSTREAM; i++) {
        int ioffset = i * iElem;
        cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, streams[i]);
    }
    for (int i = 0; i < NSTREAM; i++) {
        int ioffset = i * iElem;
        sumArrays<<<grid, block, 0, streams[i]>>>(&d_A[ioffset], &d_B[ioffset], &gpu_ref[ioffset], iElem);
    }
    for (int i = 0; i < NSTREAM; i++) {
        int ioffset = i * iElem;
        cudaMemcpyAsync(&host_ref[ioffset], &gpu_ref[ioffset], iBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < nElem; i++) {
        printf("%.2f\t", host_ref[i]);
    }

    return 0;
}
