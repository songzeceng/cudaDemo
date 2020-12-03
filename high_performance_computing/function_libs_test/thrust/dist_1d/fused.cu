//
// Created by root on 2020/12/3.
//
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/transform.h"
#include "math.h"
#include "stdio.h"

#define N 64

struct DistanceFrom {
    float mRef;
    int mN;

    DistanceFrom(float ref, int n) : mRef(ref), mN{n} {}

    __host__ __device__ float operator()(const float &x) {
        float scaledX = x / (mN - 1);
        return std::sqrt((scaledX - mRef) * (scaledX - mRef));
    }
};

int main() {
    float ref = 0.5f;

    thrust::device_vector<float> dvec_dist(N);
    thrust::transform(thrust::counting_iterator<float>(0), thrust::counting_iterator<float>(N), dvec_dist.begin(),
                      DistanceFrom(ref, N));
    // instantiate function object DistanceFrom.

    thrust::host_vector<float> h_dist(N);
    h_dist = dvec_dist;
    for (int i = 0; i < N; i++) {
        printf("x = %.3f, dist = %.3f\n", 1.f * i / (N - 1), h_dist[i]);
    }

    return 0;
}
