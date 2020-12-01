//
// Created by root on 2020/11/30.
//

#include "kernel.cuh"
#include <stdio.h>
#include <stdlib.h>

#define N 1024

int main() {
    int res = 0;
    int *a = (int *) malloc(N * sizeof(int ));
    int *b = (int *) malloc(N * sizeof(int ));

    for (int i = 0; i < N; i++) {
        a[i] = 1;
        b[i] = 1;
    }
    dotLauncher(&res, a, b, N);
    printf("Res = %d\n", res);

    free(a);
    free(b);

    return 0;
}