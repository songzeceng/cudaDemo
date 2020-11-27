//
// Created by songzeceng on 2020/11/26.
//

#include "stdio.h"
#include "stdlib.h"
#include "kernel.h"
#include "math.h"

#define N 256

int main() {
    float PI = 3.1415926;
    float h = 2 * PI / N;

    float x[N] = {0.0f};
    float u[N] = {0.0f};
    float result[N] = {0.0f};

    for (int i = 0; i < N; ++i) {
        x[i] = h * i;
        u[i] = sin(x[i]);
    }

    ddParallel(result, u, N, h);

    for (int i = 0; i < N; ++i) {
       printf("%.2f\t", result[i]);
    }
    printf("\n");

    return 0;
}