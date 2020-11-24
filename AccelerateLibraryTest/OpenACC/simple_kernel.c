//
// Created by root on 2020/11/24.
//
#include <stdio.h>
#include <stdlib.h>

int main() {
    int N = 256;
    int *A = (int *) malloc(sizeof(int) * N);
    int *B = (int *) malloc(sizeof(int) * N);
    int *C = (int *) malloc(sizeof(int) * N);
    int *D = (int *) malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i + 1;
    }

    // for simple compile: pgcc -acc simple_kernel.c -o simple_kernel
    // to see how automatic parallel realizes: pgcc -acc simple_kernel.c -o simple_kernel -Minfo=accel


//    #pragma acc kernels // parallelize following code in {}
#pragma acc kernels if (N > 128) // if N > 128, we parallelize following code in {}
    {
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
        for (int i = 0; i < N; i++) {
            D[i] = C[i] * B[i];
        }
    }

    for (int i = 0; i < 20; i++) {
        printf("%d\t", C[i]);
    }
    printf("\n");

    // We can use async-wait to sync async tasks
#pragma acc kernels async(0) // perform async task with id 0
    {

    }

#pragma acc kernels wait(0) async(1) // perform async task 1 after task 0 is finished
    {

    }

#pragma acc kernels wait(1) async(2) // perform async task 2 after task 1 is finished
    {

    }

#pragma acc wait(2) // wait for task 0 being finished

#pragma acc wait // wait for all parallel code being finished

#pragma acc kernels if (N > 128) async // this async code will execute if N > 128
    {

    }

    // parallel two loops
#pragma acc parallel // parallel macro provides more options for us to parallelize code than kernel macro
    {
#pragma acc loop
        {
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }
        }
#pragma acc loop
        {
            for (int i = 0; i < N; i++) {
                D[i] = C[i] * B[i];
            }
        }
    }

    // We can use data macro to explicitly move data between host and device
#pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N], D[0:N])
// copy in means copy to device, copy out means copy to host

// The range of data transportation is the whole array, in which case we can remove the range to simplify our code like this:
// #pragma acc data copyin(A, B) copyout(C, D)
    {
#pragma acc loop
        {
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }
        }
#pragma acc loop
        {
            for (int i = 0; i < N; i++) {
                D[i] = C[i] * B[i];
            }
        }
    }

    // Macro data is synchronous, while enter data and exit data are asynchronous
#pragma acc enter data copyin(A[0:N], B[0:N]) async(0)

// do work in host

#pragma acc kernels async(1) wait(0) // Apparently we should do asynchronous work after data transformation
    {
        for (int i = 0; i < N; i++) {
            A[i] = B[i] * 4 + i;
        }
    }

#pragma acc exit data copyout(A) async(2) wait(1) // Get A out after task 1 is over

// do work in host

#pragma wait(2) // wait for copy of A finished

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}
