//
// Created by root on 2020/11/30.
//

#ifndef CUDADEMO_KERNEL_CUH
#define CUDADEMO_KERNEL_CUH

struct uchar4;

void centroidParallel(uchar4 *img, int width, int height);


#endif //CUDADEMO_KERNEL_CUH