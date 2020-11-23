//
// Created by songzeceng on 2020/11/16.
//

#include "cuda_runtime.h"
#include "stdio.h"

__device__ int myAtomicAdd(int *address, int increment) {
    int expected = *address;
    int oldValue = atomicCAS(address, expected, expected + increment);
    // if value changed after *address and before atomicCAS, the oldValue will be different from expected.

    while (oldValue != expected) {
        expected = oldValue;
        oldValue = atomicCAS(address, expected, expected + increment);
    }
    return oldValue;
}

//__device__ float myAtomicAddFloat(float *ptr, float increment) {
//    int *ptr_int = reinterpret_cast<int *>(ptr);
//    // cannot get correct value in Pascal 6.1. We can use atomicAdd(float*, float) directly.
//
//    float current_value = *ptr;
//    int expected = __float2int_rn(current_value);
//    int desired = __float2int_rn(current_value + increment);
//
//    int oldValue = atomicCAS(ptr_int, expected, desired);
//    printf("Before CAS: current_value:%f, *ptr_int:%d, expected:%d, desired:%d, oldValue:%d\n",
//           current_value, *ptr_int, expected, desired, oldValue);
//    while (oldValue != expected) {
//        expected = oldValue;
//        desired = __float2int_rn(__int2float_rn(oldValue) + increment);
//        oldValue = atomicCAS(ptr_int, expected, desired);
//    }
//    printf("After CAS: current_value:%f, expected:%d, desired:%d, oldValue:%d\n",
//           current_value, expected, desired, oldValue);
//    return __int2float_rn(oldValue);
//}

__global__ void TestKernel(int *a) {
//    printf("%d\n", __popc(10));
//    printf("%d\n", __popc(9));
//    printf("%d\n", __popc(8));
//    printf("%d\n", __popc(11));
//    printf("%d\n", __popc(7));
//    printf("%d\n", __popc(15));
//    printf("%d\n", __popc(1));
//    printf("%d\n", __popc(0));
    // __popc()：返回二进制中1的数量

//    printf("%d\n", __ffs(0));
//    printf("%d\n", __ffs(1));
//    printf("%d\n", __ffs(7));
//    printf("%d\n", __ffs(8));
//    printf("%d\n", __ffs(9));
//    printf("%d\n", __ffs(10));
    // __ffs()：返回最低的1位，比如10(1010)返回2

    int x = atomicAdd(a, 5);
    x = myAtomicAdd(a, 4);
//    printf("%d, %d\n", *a, x);
}

//__global__ void testKernel2(float* value, float increment) {
//    printf("%f\n", myAtomicAddFloat(value, increment));
//}

// Use nvcc --ptx -o IntrinsicsTest.ptx IntrinsicsTest.cu to get ptx file
__global__ void intrinsic(float* ptr) {
    *ptr = __powf(*ptr, 2.1f);
}

__global__ void standard(float* ptr) {
    *ptr = powf(*ptr, 2.1f);
}

__shared__ int a[1];

__global__ void unsafe(int* values_read) {
    int old = a[0];
    a[0] = old + 1;
    values_read[threadIdx.x] = old;
}

__global__ void atomic(int* values_read) {
    values_read[threadIdx.x] = atomicAdd(a, 1);
}

int main() {
    int *h_a = (int *)malloc(sizeof(int ));
    *h_a = 4;

    int *a;
    cudaMalloc(&a, sizeof(int ));
    cudaMemcpy(a, h_a, sizeof(int ), cudaMemcpyHostToDevice);

    TestKernel<<<1, 10>>>(a);

    cudaMemcpy(h_a, a, sizeof(int ), cudaMemcpyDeviceToHost);

    printf("%d\n", *h_a);

    int thread_num = 10;
    int nBytes = thread_num * sizeof(int );
    int *values_read_h = (int *) malloc(nBytes);
    int *values_read_d;
    cudaMalloc(&values_read_d, nBytes);

    atomic<<<1, 10>>>(values_read_d);
    cudaMemcpy(values_read_h, values_read_d, nBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < thread_num; i++) {
        printf("%d\t", values_read_h[i]);
    }
    printf("\n");

    memset(values_read_h, 0, nBytes);
    cudaMemset(values_read_d, 0, nBytes);

    unsafe<<<1, 10>>>(values_read_d);
    cudaMemcpy(values_read_h, values_read_d, nBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < thread_num; i++) {
        printf("%d\t", values_read_h[i]);
    }
    printf("\n");

//    float *h_f = (float *)malloc(sizeof(float ));
//    *h_f = 4.0f;
//
//    float *f;
//    cudaMalloc(&f, sizeof(float ));
//    cudaMemcpy(f, h_f, sizeof(float ), cudaMemcpyHostToDevice);
//
//    testKernel2<<<1, 10>>>(f, 1.2);
//
//    cudaMemcpy(h_f, f, sizeof(float ), cudaMemcpyDeviceToHost);

//    printf("%f\n", *h_f);

    cudaFree(a);
    cudaFree(values_read_d);
    free(h_a);
    free(values_read_h);
    return 0;
}
