//
// Created by root on 2020/11/23.
//

#include "stdio.h"
#include "cuda_runtime.h"

#define N 256

int n_streams = 4;

__global__ void kernel_1() {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.2);
    }
//    printf("sum=%.4f in kernel_1\n", sum);
}

__global__ void kernel_2() {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.2);
    }
//    printf("sum=%.4f in kernel_2\n", sum);
}

__global__ void kernel_3() {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.2);
    }
//    printf("sum=%.4f in kernel_3\n", sum);
}

__global__ void kernel_4() {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum = sum + tan(0.1) * tan(0.2);
    }
//    printf("sum=%.4f in kernel_4\n", sum);
}

int main() {
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1); // set max_connections to 1
    printf("CUDA_DEVICE_MAX_CONNECTIONS: %s\n", getenv("CUDA_DEVICE_MAX_CONNECTIONS"));

    cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(cudaStream_t));
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    float elapsed_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(1);
    dim3 grid(1);

//    cudaEventRecord(start);
//    // Deep-first schedule 0.6253ms
//    for (int i = 0; i < n_streams; i++) {
//        kernel_1<<<grid, block, 0, streams[i]>>>();
//        kernel_2<<<grid, block, 0, streams[i]>>>();
//        kernel_3<<<grid, block, 0, streams[i]>>>();
////        kernel_3<<<grid, block, 0>>>();
//        kernel_4<<<grid, block, 0, streams[i]>>>();
//    }
//    cudaEventRecord(stop);
//
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&elapsed_time, start, stop);
//    printf("time elapsed between start and stop: %.4f\n", elapsed_time);

    cudaEventRecord(start);
    // Breadth-first schedule 0.2397ms
    for (int i = 0; i < n_streams; i++) {
        kernel_1<<<grid, block, 0, streams[i]>>>();
    }
    for (int i = 0; i < n_streams; i++) {
        kernel_2<<<grid, block, 0, streams[i]>>>();
    }
    for (int i = 0; i < n_streams; i++) {
        kernel_3<<<grid, block, 0, streams[i]>>>();
    }
    for (int i = 0; i < n_streams; i++) {
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("time elapsed between start and stop: %.4f", elapsed_time);
}
