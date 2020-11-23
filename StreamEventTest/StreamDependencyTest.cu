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
    cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(cudaStream_t));
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    float elapsed_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t *kernelEvent = (cudaEvent_t *) malloc(n_streams * sizeof(cudaEvent_t));
    for (int i = 0; i < n_streams; i++) {
        cudaEventCreateWithFlags(&kernelEvent[i], cudaEventDisableTiming);
    }

    dim3 block(1);
    dim3 grid(1);

    cudaEventRecord(start);
    for (int i = 0; i < n_streams; i++) {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
        cudaEventRecord(kernelEvent[i], streams[i]);
        cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0); // The last stream waits for the target event to complete
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("time elapsed between start and stop: %.4f", elapsed_time);
}
