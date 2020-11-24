//
// Created by root on 2020/11/24.
//

#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "stdio.h"

__global__ void initialize_state(curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // init curand state for each thread
    curand_init(9444, tid, 0, states + tid);
}

__global__ void refill_randoms(float *dRand, int N, curandState* states) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    curandState *state = states + tid;

    for (int i = tid; i < N; i += nthreads) {
        dRand[i] = curand_uniform(state);
        // generate random number following uniform distribution for each thread
        // the number of random numbers is N in total
    }
}

float cuda_device_rand() {
    // use device api to generate random numbers

    static curandState *states = NULL;
    static float *dRand = NULL, *hRand = NULL;
    static int dRand_length = 1000000, dRand_used = 1000000;

    int block = 256;
    int grid = 30;

    if (dRand == NULL) {
        // if dRand is null, then allocate memory and initialize states
        cudaMalloc(&dRand, sizeof(float ) * dRand_length);
        cudaMalloc(&states, sizeof(curandState) * block * grid);
        hRand = (float *) malloc(sizeof(float ) * dRand_length);
        initialize_state<<<grid, block>>>(states);
    }

    if (dRand_used == dRand_length) {
        // if all random data have been traversed, we should generate a new batch of data
        refill_randoms<<<grid, block>>>(dRand, dRand_length, states);
        cudaMemcpy(hRand, dRand, sizeof(float ) * dRand_length, cudaMemcpyDeviceToHost);
        dRand_used = 0;
    }
    return hRand[dRand_used++];
}

float cuda_host_rand() {
    // generate random data with host api

    static curandGenerator_t randGen;
    static float *dRand = NULL, *hRand = NULL;
    static int dRand_length = 1000000, dRand_used = 1000000;

    if (dRand == NULL) {
        // if dRand is null, then allocate memory and create generator
        cudaMalloc(&dRand, sizeof(float ) * dRand_length);
        hRand = (float *) malloc(sizeof(float ) * dRand_length);
        curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    }

    if (dRand_used == dRand_length) {
        // if all random data generated have been traversed, we should generate a batch of new data
        curandGenerateUniform(randGen, dRand, dRand_length); // the new data are in device memory
        cudaMemcpy(hRand, dRand, sizeof(float ) * dRand_length, cudaMemcpyDeviceToHost);
        dRand_used = 0;
    }
    return hRand[dRand_used++];
}

// nvcc -lcurand ReplaceRand.cu -o ReplaceRand
int main() {
    for (int i = 0; i < 256; i++) {
        float h = cuda_host_rand();
        float d = cuda_device_rand();
        printf("h = %.2f, d = %.2f\n", h, d);
    }
    return 0;
}