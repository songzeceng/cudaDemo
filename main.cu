#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void MyKernel(float *dev0Ptr, float *dev1Ptr, int begin, int size) {
    for (int i = begin; i < begin + size; i++) {
        dev1Ptr[i] = dev0Ptr[i] * 2;
    }
}

void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data) {
    printf("Inside callback %d with status %d\n", (int) data, status);
}

int main() {

    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&stream[i]);
    }

//    int priority_high, priority_low;
//    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
//
//    cudaStream_t st_high, st_low;
//    cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
//    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);

    float *hostPtr;
    int size = 16;
    cudaMallocHost(&hostPtr, 2 * size * sizeof(float ));

    float *dev0Ptr;
    cudaMalloc(&dev0Ptr, 2 * size * sizeof(float ));
    float *dev1Ptr;
    cudaMalloc(&dev1Ptr, 2 * size * sizeof(float ));

    for (int i = 0; i < 2 * size; i++) {
        hostPtr[i] = i + 1;
    }

//    for (int i = 0; i < 2; i++) {
//        cudaMemcpyAsync(&dev0Ptr[i * size], &hostPtr[i * size], size * sizeof(float ), cudaMemcpyHostToDevice, stream[i]);
//
//        MyKernel<<<100, 512, 0, stream[i]>>>(dev0Ptr, dev1Ptr, i * size, size);
//
//        cudaMemcpyAsync(&hostPtr[i * size], &dev1Ptr[i * size], size * sizeof(float ), cudaMemcpyDeviceToHost, stream[i]);
//
//        cudaStreamAddCallback(stream[i], MyCallback, (void *) i, 0);
//
//        cudaStreamDestroy(stream[i]);
//    }
//
//    for (int i = 0; i < 2 * size; i++) {
//        printf("%f\t", hostPtr[i]);
//    }
//    printf("\n");

///////////////////////////////////////////////////////////

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < 2; ++i) {
        cudaMemcpyAsync(dev0Ptr + i * size, hostPtr + i * size,
                        size * sizeof(float ), cudaMemcpyHostToDevice, stream[i]);
        MyKernel<<<1, 1, 0, stream[i]>>>
                (dev0Ptr, dev1Ptr, i * size, size);
        cudaMemcpyAsync(hostPtr + i * size, dev1Ptr + i * size,
                        size * sizeof(float ), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "time: " << elapsedTime << std::endl;

    for (int i = 0; i < 2 * size; i++) {
        printf("%f\t", hostPtr[i]);
    }
    printf("\n");

    ///////////////////////////////////////////////////////

//    for (int i = 0; i < 2; i++) {
//        cudaMemcpyAsync(dev0Ptr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
//    }
//
//    for (int i = 0; i < 2; i++) {
//        MyKernel<<<100, 512, 0, stream[i]>>>(dev0Ptr + i * size, dev1Ptr + i * size, size);
//    }
//
//    for (int i = 0; i < 2; i++) {
//        cudaMemcpyAsync(hostPtr + i * size, dev1Ptr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
//    }

    return 0;
}
