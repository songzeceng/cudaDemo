//
// Created by root on 2020/12/3.
//

#define cimg_display 0
#include "CImg.h"
#include "cuda_runtime.h"
#include "npp.h"
#include "stdlib.h"

#define kNumCh 3

void sharpenNPP(Npp8u *arr, int w, int h) {
    Npp8u *d_in = 0, *d_out = 0;
    Npp32f *d_filter = 0;
    Npp32f filter[9] = {-1.0, -1.0, -1.0, -1.0, 9.0, -1.0, -1.0, -1.0, -1.0};
    cudaMalloc(&d_out, kNumCh * w * h * sizeof(Npp8u));
    cudaMalloc(&d_filter, 9 * sizeof(Npp32f));
    cudaMemcpy(d_filter, filter, 9 * sizeof(Npp32f), cudaMemcpyHostToDevice);

    NppiSize oKernelSize = {3, 3};
    NppiPoint oAnchor = {1, 1}; // the anchor point is 9.0 in filter array.
    NppiSize oSrcSize = {w, h};
    NppiPoint oSrcOffset = {0, 0}; // process from the first one of data
    NppiSize oSizeROI = {w, h}; // Region of interest area, here is the whole data.

    nppiFilterBorder32f_8u_C3R(d_in, kNumCh * w * h * sizeof(Npp8u), oSrcSize, oSrcOffset,
                               d_out, kNumCh * w * h * sizeof(Npp8u), oSizeROI,
                               d_filter, oKernelSize, oAnchor, NPP_BORDER_REPLICATE);
    // NPP_BORDER_REPLICATE means if the data is out of area border, we replicate one from the nearest position.
    // 32f_8u_C3R: 32 bit float of template, 8 bit unsigned char of input data, 3 color channel and rectangle ROI.

    cudaMemcpy(arr, d_out, kNumCh * w * h * sizeof(Npp8u), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_filter);
}

int main() {
    cimg_library::CImg<unsigned char > image("butterfly.bmp");
    int w = image.width();
    int h = image.height();
    Npp8u *arr = (Npp8u *) malloc(kNumCh * w * h * sizeof(Npp8u));

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int ch = 0; ch < kNumCh; ch++) {
                arr[kNumCh * (r * w + c) + ch] = image(c, r, ch);
            }
        }
    }

    sharpenNPP(arr, w, h);

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int ch = 0; ch < kNumCh; ch++) {
                image(c, r, ch) = arr[kNumCh * (r * w + c) + ch];
            }
        }
    }

    image.save_bmp("out.bmp");
    free(arr);
    return 0;
}
