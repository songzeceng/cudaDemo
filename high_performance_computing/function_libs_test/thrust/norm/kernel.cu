//
// Created by root on 2020/12/3.
//

#include "thrust/device_vector.h"
#include "thrust/inner_product.h"
#include "math.h"
#include "stdio.h"

#define N (1024 * 1024)

int main() {
    thrust::device_vector<float> d_vec(N, 1.2f);
    float norm = sqrt(thrust::inner_product(d_vec.begin(), d_vec.end(), d_vec.begin(), 0.0f));
    // parameters: begin and end of first vector, begin of second vector(the end of it is determined by the range of the first vector) and result init value
    printf("norm = %.2f\n", norm);
    return 0;
}