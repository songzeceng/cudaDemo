//
// Created by root on 2020/11/12.
//

#include "cuda_runtime.h"
#include "stdio.h"


__device__ int *m = NULL, *n = NULL;
__device__ volatile int *m_v = NULL, *n_v = NULL;
__device__ int A, B, A_v, B_v;

__device__ void write() {
//    m = 10;
//    n = 20;
//
//    m_v = 10;
//    n_v = 20;

    (*m)++;
    (*n)++;

    (*m_v)++;
    (*n_v)++;
}

__device__ void read() {
    A = *m;
    B = *n;

    A_v = *m_v;
    B_v = *n_v;
}

__global__ void testKernel(int *count) {
    if ((blockIdx.x * blockDim.x + threadIdx.x) == 0 && threadIdx.y == 0) {
        m = &A;
        n = &B;
        m_v = &A_v;
        n_v = &B_v;

        (*m) = 0;
        (*n) = 0;
        (*m_v) = 0;
        (*n_v) = 0;

//        printf("thread 0\n");
    } else {
//        printf("following thread\n");
    }
    write();

//    __shared__ int t;
//    t++;
//
//    printf("t = %d\n", t);
//    __threadfence();
//    __syncthreads();

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    *count = 1 + idx_x + idx_y * blockDim.x * gridDim.x;
    printf("count = %d\n", *count);

    read();

//    printf("A = %d, B = %d, A_v = %d, B_v = %d\n", A, B, A_v, B_v);
}

int *m_h = NULL, *n_h = NULL;
volatile int *m_v_h = NULL, *n_v_h = NULL;
int A_h, B_h, A_v_h, B_v_h;

void writeH() {
//    m = 10;
//    n = 20;
//
//    m_v = 10;
//    n_v = 20;

    (*m_h)++;
    (*n_h)++;

    (*m_v_h)++;
    (*n_v_h)++;
}

void readH() {
    A_h = *m_h;
    B_h = *n_h;

    A_v_h = *m_v_h;
    B_v_h = *n_v_h;
}

void testHost(int epoch) {
    if (epoch == 0) {
        m_h = &A_h;
        n_h = &B_h;
        m_v_h = &A_v_h;
        n_v_h = &B_v_h;

        (*m_h) = 0;
        (*n_h) = 0;
        (*m_v_h) = 0;
        (*n_v_h) = 0;
    }
    writeH();
    readH();

    printf("A_h = %d, B_h = %d, A_v_h = %d, B_v_h = %d\n", A_h, B_h, A_v_h, B_v_h);
}

int main() {
    int x = 10, y = 2;
    dim3 block(2, 2);
    dim3 grid((x + block.x - 1) / block.x, (y + block.y - 1) / block.y);

    int count = 0;
    int *countD;
    cudaMalloc(&countD, sizeof(int));
    cudaMemcpy(countD, &count, sizeof(count), cudaMemcpyHostToDevice);
    testKernel<<<grid, block>>>(countD);
//    cudaError_t err = cudaDeviceSynchronize();
//    printf("result:%s\n", cudaGetErrorString(err));

    cudaMemcpy(&count, countD, sizeof(count), cudaMemcpyDeviceToHost);
//    printf("count = %d\n", count);

    cudaMemcpy(&A_h, &A, sizeof(A), cudaMemcpyDeviceToHost);
    cudaMemcpy(&B_h, &B, sizeof(B), cudaMemcpyDeviceToHost);
    cudaMemcpy(&A_v_h, &A_v, sizeof(A_v), cudaMemcpyDeviceToHost);
    cudaMemcpy(&B_v_h, &B_v, sizeof(B_v), cudaMemcpyDeviceToHost);

    printf("===============================\n");

    printf("A_h = %d, B_h = %d, A_v_h = %d, B_v_h = %d\n", A_h, B_h, A_v_h, B_v_h);

//    for (int i = 0; i < x * y; i++) {
//        testHost(i);
//    }

    return 0;
}