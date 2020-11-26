//
// Created by root on 2020/11/26.
//

#ifndef CUDADEMO_KERNEL_H
#define CUDADEMO_KERNEL_H

struct uchar4;
struct int2;

void kernelLauncher(uchar4 *d_out, int w, int h, float p, int s);

#endif //CUDADEMO_KERNEL_H
