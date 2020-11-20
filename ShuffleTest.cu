//
// Created by root on 2020/11/20.
//

#include "stdio.h"
#include "cuda_runtime.h"

#define BDIMX 16
#define SEGM 4

#define DIM 32
#define SMEMDIM 4

__global__ void test_shuffle_broadcast(int *d_out, int *d_in, int srcLane) {
    int value = d_in[threadIdx.x];
    value = __shfl(value, srcLane, BDIMX);
    // change data with srcLane thread. cyclic is supported if necessary.
    // if there is a dependency, the smaller thread index executes earlier.Else executions will do in parallel.
    // when thread 0 is changing data with thread 1 and thread 1 is changing with thread 3, the former executes first
    d_out[threadIdx.x] = value;
}

__global__ void test_shuffle_up(int *d_out, int *d_in, int delta) {
    int value = d_in[threadIdx.x];
    value = __shfl_up(value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shuffle_down(int *d_out, int *d_in, int delta) {
    int value = d_in[threadIdx.x];
    value = __shfl_down(value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shuffle_cycle(int *d_out, int *d_in, int delta) {
    int value = d_in[threadIdx.x];
    int value_ = __shfl(value, threadIdx.x + delta, BDIMX);
    d_out[threadIdx.x] = value_;
}

__global__ void test_shuffle_butterfly(int *d_out, int *d_in, int delta) {
    int value = d_in[threadIdx.x];
    int value_ = __shfl_xor(value, delta, BDIMX);
    // change data with next thread
    // t0 with t1, t2 with t3, if no t5, data in t4 will be 0
    d_out[threadIdx.x] = value_;
}

__global__ void test_shuffle_xor_array(int *d_out, int *d_in, int mask) {
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) {
        value[i] = d_in[idx + i];
        value[i] = __shfl_xor(value[i], mask, SEGM);
        d_out[idx + i] = value[i];
    }
}

__inline__ __device__ void swap(int *value, int laneIdx, int mask, int firstIdx, int secondIdx) {
    bool pred = ((laneIdx % 2) == 0); // first thread per pair or not
    if (pred) {
        int temp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = temp;
    }

    value[secondIdx] = __shfl_xor(value[secondIdx], mask, BDIMX);

    if (pred) {
        int temp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = temp;
    }
}

__global__ void test_shuffle_warp(int *d_out, int *d_in, int mask, int firstIndex, int secondIndex) {
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) {
        value[i] = d_in[idx + i];
    }

    swap(value, threadIdx.x, mask, firstIndex, secondIndex);

    for (int i = 0; i < SEGM; i++) {
        d_out[idx + i] = value[i];
    }
}

__inline__ __device__ int warpReduce(int mySum) {
    mySum += __shfl_xor(mySum, 16);
    mySum += __shfl_xor(mySum, 8);
    mySum += __shfl_xor(mySum, 4);
    mySum += __shfl_xor(mySum, 2);
    mySum += __shfl_xor(mySum, 1);
    return mySum;
}

__global__ void reduceShuffle(int *g_idata, int *g_odata, int n) {
    __shared__ int smem[SMEMDIM];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int mySum = g_idata[idx]; // current data

    int laneIdx = threadIdx.x % warpSize; // index in warp
    int warpIdx = threadIdx.x / warpSize; // warp index

    mySum = warpReduce(mySum); // reduce sum in a warp

    if (laneIdx == 0) {
        smem[warpIdx] = mySum; // if it`s the first thread in the warp, store warp sum into shared memory
    }

    __syncthreads();

    mySum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0; // get each warp`s sum

    if (warpIdx == 0) {
        mySum = warpReduce(mySum); // if it`s the first warp, reduce sums of each warp for the block
    }

    if (threadIdx.x == 0) {
        g_odata[blockIdx.x] = mySum; // store the block sum into the first thread of this block.
    }
}

int main() {
    int nx = BDIMX;
    int nBytes = nx * sizeof(int);

    int *h_in = (int *) malloc(nBytes);
    int *h_out = (int *) malloc(nBytes);

    for (int i = 0; i < nx; i++) {
        h_in[i] = i;
    }

    int *d_in;
    int *d_out;

    cudaMalloc(&d_in, nBytes);
    cudaMalloc(&d_out, nBytes);

    cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BDIMX);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x);

    test_shuffle_broadcast<<<gridDim, blockDim>>>(d_out, d_in, 2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    printf("\n============\n");
    for (int i = 0; i < nx; i++) {
        printf("%d->", h_out[i]);
    }

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);
    test_shuffle_up<<<gridDim, blockDim>>>(d_out, d_in, 2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    printf("\n============\n");
    for (int i = 0; i < nx; i++) {
        printf("%d->", h_out[i]);
    }

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);
    test_shuffle_down<<<gridDim, blockDim>>>(d_out, d_in, 2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    printf("\n============\n");
    for (int i = 0; i < nx; i++) {
        printf("%d->", h_out[i]);
    }

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);
    test_shuffle_cycle<<<gridDim, blockDim>>>(d_out, d_in, 2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    printf("\n============\n");
    for (int i = 0; i < nx; i++) {
        printf("%d->", h_out[i]);
    }

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);
    test_shuffle_butterfly<<<gridDim, blockDim>>>(d_out, d_in, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    printf("\n============\n");
    for (int i = 0; i < nx; i++) {
        printf("%d->", h_out[i]);
    }

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);
    test_shuffle_xor_array<<<1, BDIMX / SEGM>>>(d_out, d_in, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    printf("\n============\n");
    for (int i = 0; i < nx; i++) {
        printf("%d->", h_out[i]);
    }

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);
    test_shuffle_warp<<<1, BDIMX / SEGM>>>(d_out, d_in, 1, 0, 3);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    printf("\n============\n");
    for (int i = 0; i < nx; i++) {
        printf("%d->", h_out[i]);
    }

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);
    blockDim.x = DIM;
    gridDim.x = (nx + blockDim.x - 1) / blockDim.x;
    reduceShuffle<<<gridDim, blockDim>>>(d_in, d_out, nx);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    printf("\n============\n");
    for (int i = 0; i < nx; i++) {
        printf("%d->", h_out[i]);
    }

    free(h_out);
    cudaFree(d_out);
    free(h_in);
    cudaFree(d_in);
}
