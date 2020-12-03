//
// Created by root on 2020/12/3.
//

#include "curand.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/generate.h"
#include "thrust/count.h"
#include "stdio.h"

#define N (1 << 20)

//  nvcc pi_thust_curand.cu -o pi_thust_curand --expt-extended-lambda -lcurand
int main() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 44559);

    thrust::device_vector<float> d_x(N);
    thrust::device_vector<float> d_y(N);

    float *p_x = thrust::raw_pointer_cast(&d_x[0]);
    float *p_y = thrust::raw_pointer_cast(&d_y[0]);
    curandGenerateUniform(gen, p_x, N);
    curandGenerateUniform(gen, p_y, N);
    curandDestroyGenerator(gen);

    int insideCount = thrust::count_if(thrust::make_zip_iterator(
            thrust::make_tuple(d_x.begin(), d_y.begin())), thrust::make_zip_iterator(
            thrust::make_tuple(d_x.end(), d_y.end())), []
    __device__(thrust::tuple < float, float > pair)
    {
        return (pow(thrust::get<0>(pair), 2) + pow(thrust::get<1>(pair), 2)) < 1.f;
    });

    printf("pi = %f\n", insideCount * 4.f / N);

    return 0;
}