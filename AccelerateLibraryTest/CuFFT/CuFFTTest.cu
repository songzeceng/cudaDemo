//
// Created by root on 2020/11/24.
//

#include "cuda_runtime.h"
#include "cufft.h"
#include "stdio.h"

int nprint = 30;

void generate_fake_samples(int N, float **out) {
    float *result = (float *) malloc(sizeof(float ) * N);
    double delta = M_PI / 20.0;

    for (int i = 0; i < N; i++) {
        result[i] = cos(i * delta);
    }

    *out = result;
}

void real_to_complex(float *r, cufftComplex **complex, int N) {
    (*complex) = (cufftComplex *) malloc(sizeof(cufftComplex) * N);

    for (int i = 0; i < N; i++) {
        (*complex)[i].x = r[i];
        (*complex)[i].y = 0;
    }
}

// nvcc -lcufft CuFFTTest.cu -o CuFFTTest
int main() {
    int N = 2048;
    float *samples;
    cufftHandle plan;
    cufftComplex *dComplexSamples, *complexSamples, *complexFreq;

    // allocate memory
    samples = (float *) malloc(N * sizeof(float ));
    complexSamples = (cufftComplex *) malloc(N * sizeof(cufftComplex));
    complexFreq = (cufftComplex *) malloc(N * sizeof(cufftComplex));
    cudaMalloc(&dComplexSamples, sizeof(cufftComplex) * N);

    // generate input
    generate_fake_samples(N, &samples);
    printf("Initial samples:\n");
    for (int i = 0; i < nprint; i++) {
        printf("%.2f\t", samples[i]);
    }
    printf("\n");
    real_to_complex(samples, &complexSamples, N);

    // create cufft plan with type complex to complex
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    // copy data to device
    cudaMemcpy(dComplexSamples, complexSamples, N * sizeof(cufftComplex ), cudaMemcpyHostToDevice);

    // execute forward fourier transform
    cufftExecC2C(plan, dComplexSamples, dComplexSamples, CUFFT_FORWARD);

    // get data from device and print demo data
    cudaMemcpy(complexFreq, dComplexSamples, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);
    printf("Fourier coefficient:\n");
    for (int i = 0; i < nprint; i++) {
        printf("(%.2f, %.2f)\t", complexFreq[i].x , complexFreq[i].y);
    }
    printf("\n");

    free(samples);
    free(complexSamples);
    free(complexFreq);
    cudaFree(dComplexSamples);
    cufftDestroy(plan);

    return 0;
}