//
// Created by root on 2020/11/19.
//

#include "stdio.h"
#include "cuda_runtime.h"

#define BDIMX 32
#define BDIMY 32

__global__ void transposeNaiveGem(int *in, int *out, int nx, int ny) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nx && y < ny) {
        out[x * ny + y] = in[y * nx + x];
    }
}

__global__ void transposeNaiveGem2(int *in, int *out, int nx, int ny) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nx && y < ny) {
        out[y * nx + x] = in[x * ny + y];
        // store is combined but load not
        // Bytes count per request in Pascal seems lower than 128
    }
}

__global__ void transposeSmem(int *in, int *out, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread x coordinate in block(origin matrix)
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread y coordinate in block(origin matrix)

    __shared__ int tile[BDIMX][BDIMY];
//    __shared__ int tile[BDIMX][BDIMY + 1]; // We can append one column per row to eliminate store bank conflict

    int ti = iy * nx + ix; // thread data index in matrix / coordinate in origin matrix

    int bidx = threadIdx.y * blockDim.x + threadIdx.x; // thread index in block
    int irow = bidx / blockDim.y; // row of current thread in block
    int icol = bidx % blockDim.y; // column of current thread in block

    ix = blockIdx.y * blockDim.y + icol; // x coordinate in transpose matrix
    iy = blockIdx.x * blockDim.x + irow; // y coordinate in transpose matrix
    int to = iy * ny + ix; // coordinate in transpose matrix

    if (ix < nx && iy < ny) {
        tile[threadIdx.x][threadIdx.y] = in[ti]; // We got bank conflict here, but both the throughput and speed are still higher than global memory
        __syncthreads();
        out[to] = tile[irow][icol]; // Just change the index of data in array
    }
}

__global__ void transposeSmemUnrollPad(int *in, int *out, int nx, int ny) {
    int ix = blockDim.x * blockIdx.x * 2 + threadIdx.x; // thread start x index in block
    int iy = blockDim.y * blockIdx.y + threadIdx.y; // thread start y index in block

    __shared__ int tile[BDIMX * 2][BDIMY + 1];

    int ti = iy * nx + ix; // start data index in block

    int bidx = threadIdx.y * blockDim.x + threadIdx.x; // thread start index in block
    int irow = bidx / blockDim.x; // row in origin matrix
    int icol = bidx % blockDim.x; // column in origin matrix

    int outputIndex = ix * ny + blockIdx.y * BDIMY + irow; // data index in the output array

    if (icol < nx && irow < ny) {
        tile[icol][irow] = in[ti];
        __syncthreads();
        out[outputIndex] = tile[icol][irow];
    }

    if (icol + blockDim.x < nx && irow < ny) {
        tile[icol + blockDim.x][irow] = in[ti + blockDim.x];
        __syncthreads();
        out[outputIndex + blockDim.x * ny] = tile[icol + blockDim.x][irow];
    }
}

int main() {
    int nx = 1024, ny = 1024;
    int nBytes = nx * ny * sizeof(int);

    int *h_in = (int *) malloc(nBytes);
    int *h_out = (int *) malloc(nBytes);

    for (int i = 0; i < nx * ny; i++) {
        h_in[i] = i;
    }

    int *d_in;
    int *d_out;

    cudaMalloc(&d_in, nBytes);
    cudaMalloc(&d_out, nBytes);

    cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BDIMX, BDIMY);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    transposeNaiveGem<<<gridDim, blockDim>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nx * ny; i++) {
//        printf("%d->", h_out[i]);
//    }
//
//    printf("\n====================\n");

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);

    dim3 blockDim_(BDIMX, BDIMY);
    dim3 gridDim_((nx + 2 * blockDim_.x - 1) / (2 * blockDim_.x), (ny + blockDim_.y - 1) / blockDim_.y);

    printf("grid_: (%d, %d), block_: (%d, %d)\n", gridDim_.x, gridDim_.y, blockDim_.x, blockDim_.y);
    transposeSmemUnrollPad<<<gridDim_, blockDim_>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nx * ny; i++) {
//        printf("%d->", h_out[i]);
//    }
//
//    printf("\n====================\n");

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);

    transposeSmem<<<gridDim, blockDim>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nx * ny; i++) {
//        printf("%d->", h_out[i]);
//    }

    memset(h_out, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);

    transposeSmem<<<gridDim, blockDim>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nx * ny; i++) {
//        printf("%d->", h_out[i]);
//    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}
