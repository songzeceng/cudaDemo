//
// Created by root on 2020/11/26.
//

#ifndef CUDADEMO_KERNEL_H
#define CUDADEMO_KERNEL_H

struct uchar4;
typedef struct {
    int x, y;
    float rad;
    int chamfer;
    float t_s, t_a, t_g;
} BC;

void kernelLauncher(uchar4 *d_out, float *d_temp, int w, int h, BC bc);

void resetTemperature(float *d_temp, int w, int h, BC bc);

#endif //CUDADEMO_KERNEL_H
