//
// Created by root on 2020/11/26.
//

#ifndef CUDADEMO_KERNEL_H
#define CUDADEMO_KERNEL_H

struct uchar4;

void sharpenParallel(uchar4 *arr, int w, int h);

#endif //CUDADEMO_KERNEL_H
