//
// Created by root on 2020/11/24.
//

#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "stdio.h"

__global__ void device_api_kernel(curandState *states, float *out, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // init curand state for each thread
    curand_init(9444, tid, 0, states + tid);

    int nthreads = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += nthreads) {
        float rand = curand_uniform(states + tid);
        out[i] = rand * 2;
    }
}

__global__ void host_api_kernel(float *values, float *out, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int nthreads = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += nthreads) {
        float rand = values[i];
        out[i] = rand * 2;
    }
}

void cuda_device_rand() {
    // use device api to generate random numbers

    static curandState *states = NULL;
    static float *dRand = NULL, *hRand = NULL;
    static int dRand_length = 1000000;

    int block = 256;
    int grid = 30;

    cudaMalloc(&dRand, sizeof(float) * dRand_length);
    cudaMalloc(&states, sizeof(curandState) * block * grid);
    hRand = (float *) malloc(sizeof(float) * dRand_length);
    device_api_kernel<<<grid, block>>>(states, dRand, dRand_length);

    cudaMemcpy(hRand, dRand, sizeof(float) * dRand_length, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%.2f\t", hRand[i]);
    }
    printf("\n");

    free(hRand);
    cudaFree(dRand);
}

void cuda_host_rand() {
    // generate random data with host api

    static curandGenerator_t randGen;
    static float *dRand = NULL, *dOut, *hOut;
    static int dRand_length = 1000000, dRand_used = 1000000;

    cudaMalloc(&dRand, sizeof(float) * dRand_length);
    cudaMalloc(&dOut, sizeof(float) * dRand_length);
    hOut = (float *) malloc(sizeof(float) * dRand_length);
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);

    curandGenerateUniform(randGen, dRand, dRand_length); // the new data are in device memory

    host_api_kernel<<<30, 256>>>(dRand, dOut, dRand_length);

    cudaMemcpy(hOut, dOut, sizeof(float) * dRand_length, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%.2f\t", hOut[i]);
    }
    printf("\n");

    free(hOut);
    cudaFree(dOut);
    cudaFree(dRand);
}

// nvcc -lcurand RandKernel.cu -o RandKernel
int main() {
    printf("host:\n");
    cuda_host_rand();

    printf("device:\n");
    cuda_device_rand();
    return 0;
}