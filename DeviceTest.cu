//
// Created by songzeceng on 2020/11/8.
//

#include <stdio.h>

struct __align__(8) {
  int x;
  int y;
} A;

struct __align__(16) {
    int x;
    int y;
    int z;
} B;

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",device, deviceProp.major, deviceProp.minor);
    }
    return 0;
}