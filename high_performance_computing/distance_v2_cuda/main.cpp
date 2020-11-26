//
// Created by songzeceng on 2020/11/26.
//

#include "stdio.h"
#include "stdlib.h"
#include "kernel.h"

#define N 64

float scale(int i, int n) {
    return ((float ) i) / (n - 1);
}

/**
 * 编译命令：
 *  nvcc -c kernel.cu -o kernel.o
 *  nvcc -c main.cpp -o main.o
 *  nvcc main.o kernel.o -o main.exe
 * 运行命令：
 *  main.exe
 */
int main() {
    float ref = 0.5f;
    float *in = (float *) malloc(N * sizeof(float ));
    float *out = (float *) malloc(N * sizeof(float ));

    for (int i = 0; i < N; ++i) {
        in[i] = scale(i, N);
    }

    distanceArray(out, in, ref, N);

    for (int i = 0; i < N; ++i) {
       printf("%.2f\t", out[i]);
    }
    printf("\n");

    free(in);
    free(out);

    return 0;
}