//
// Created by root on 2020/12/3.
//

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/reduce.h"
#include "thrust/generate.h"
#include "thrust/transform.h"
#include "math.h"
#include "stdio.h"

#define N (1 << 20)

using namespace thrust::placeholders;

int main() {
    thrust::host_vector<float> h_x(N);
    thrust::host_vector<float> h_y(N);
    thrust::generate(h_x.begin(), h_x.end(), rand);
    thrust::generate(h_y.begin(), h_y.end(), rand);

    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    thrust::transform(d_x.begin(), d_x.end(), d_x.begin(), _1 / RAND_MAX); // change data value to [0, 1]
    thrust::transform(d_y.begin(), d_y.end(), d_y.begin(), _1 / RAND_MAX);

    thrust::device_vector<float> d_inCircle(N);
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_inCircle.begin(), (_1 * _1 + _2 * _2) < 1);
    // first1, last1, first2, result, operation(1 for true, 0 for false if op is judgement)

    float pi = thrust::reduce(d_inCircle.begin(), d_inCircle.end()) * 4.f / N;
    // reduce to get the sum of d_inCircle.
    printf("pi = %f\n", pi);


    return 0;
}