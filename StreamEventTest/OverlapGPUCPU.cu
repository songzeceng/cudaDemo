//
// Created by root on 2020/11/23.
//

#include "stdio.h"
#include "cuda_runtime.h"

#define NSTREAM 4

#define n_repeat 32

__global__ void kernel(float *g_data, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    g_data[idx] = g_data[idx] + n;
}

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data) {
    printf("callback from stream %d\n", *((int *) data));
}

int main() {
    int nElem = 256;
    int nBytes = nElem * sizeof(float);

    float *d_A;
    float *h_A;

    cudaMalloc(&d_A, nBytes);
    cudaHostAlloc((void **) &h_A, nBytes, cudaHostAllocDefault);

    for (int i = 0; i < nElem; i++) {
        h_A[i] = i;
    }

    int iElem = nElem / NSTREAM;
    int iBytes = iElem * sizeof(float);
    cudaStream_t *streams = (cudaStream_t *) malloc(NSTREAM * sizeof(cudaStream_t));
    for (int i = 0; i < NSTREAM; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(1);
    dim3 grid(iElem);

    cudaEvent_t stop;
    cudaEventCreate(&stop);

    int stream_id[NSTREAM] = {0, 1, 2, 3};
    // Deep first
    for (int i = 0; i < NSTREAM; i++) {
        int ioffset = i * iElem;
        cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, streams[i]);
        kernel<<<grid, block, 0, streams[i]>>>(&d_A[ioffset], NSTREAM);
        cudaMemcpyAsync(&h_A[ioffset], &d_A[ioffset], iBytes, cudaMemcpyDeviceToHost, streams[i]);
        cudaEventRecord(stop, streams[NSTREAM - 1]);
        cudaStreamAddCallback(streams[i], my_callback, (void *) &stream_id[i], 0);
    }

    int count = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        // Event has not accomplished in the specified stream
        // cudaEventQuery cannot co-exists with timing using cudaEventElapsed
        count++;
    }

//    printf("count = %d\n", count);
//    for (int i = 0; i < nElem; i++) {
//        printf("%.2f\t", h_A[i]);
//    }

    return 0;
}
