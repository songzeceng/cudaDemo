//
// Created by songzeceng on 2020/11/16.
//

#include "cuda_runtime.h"
#include "stdio.h"

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
    printf("%d, %d\n", *a, x);
}

int main() {
    int *h_a = (int *)malloc(sizeof(int ));
    *h_a = 4;

    int *a;
    cudaMalloc(&a, sizeof(int ));
    cudaMemcpy(a, h_a, sizeof(int ), cudaMemcpyHostToDevice);

    TestKernel<<<1, 1>>>(a);
    cudaDeviceSynchronize();

    cudaFree(a);
    free(h_a);
    return 0;
}
