//
// Created by root on 2020/12/3.
//

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/generate.h"
#include "thrust/count.h"
#include "stdio.h"

#define N (1 << 20)

// nvcc lambda.cu -o lambda --expt-extended-lambda
int main() {
    thrust::host_vector<float> h_x(N), h_y(N);

    thrust::generate(h_x.begin(), h_x.end(), rand);
    thrust::generate(h_y.begin(), h_y.end(), rand);

    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    int insideCount = thrust::count_if(thrust::make_zip_iterator(
            thrust::make_tuple(d_x.begin(), d_y.begin())), thrust::make_zip_iterator(
            thrust::make_tuple(d_x.end(), d_y.end())), []
    __device__(thrust::tuple < float, float > pair)
    {
        return (pow(thrust::get<0>(pair) / RAND_MAX, 2) + pow(thrust::get<1>(pair) / RAND_MAX, 2)) < 1.f;
    });

    printf("pi = %f\n", insideCount * 4.f / N);

    return 0;
}