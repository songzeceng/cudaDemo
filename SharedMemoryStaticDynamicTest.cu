#include "cuda_runtime.h"
#include "stdio.h"

#define BDIMX 32
#define BDIMY 16

#define IPAD 2 // Transactions = BDIMY * sizeof(T) / 8
#define IPAD_D 2

__global__ void setRowReadRow(int* out) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * gridDim.x * blockDim.x + x;

    __shared__ int tile[BDIMY][BDIMX];

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadRow(int* out) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * gridDim.x * blockDim.x + x;
    int idxInBlock = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ int tile[BDIMX][BDIMY];

    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    out[idx] = tile[idxInBlock / blockDim.y][idxInBlock % blockDim.y];
}

__global__ void setRowReadColPad(int* out) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * blockDim.x * gridDim.x + x;
    int idxInBlock = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ int tile[BDIMY][BDIMX + IPAD]; // pad for eliminating bank conflict
    // We had better let the shape of shared memory close to a square.
    // The way of read and write should be specified according to the shape of shared memory

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[idxInBlock % blockDim.y][idxInBlock / blockDim.y];
}

// We can see shared memory with pad takes the least time, maybe this is space used for time
__global__ void setRowReadColDyn(int* out) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * blockDim.x * gridDim.x + x;

    extern __shared__ int tile[];
    tile[threadIdx.x + threadIdx.y * blockDim.x] = idx;

    __syncthreads();

    int idxInBlock = threadIdx.y * blockDim.x + threadIdx.x;
    int col = idxInBlock / blockDim.y;
    int row = idxInBlock % blockDim.y;

    out[idx] = tile[col + row * blockDim.x];
}

__global__ void setRowReadColDynPad(int* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * blockDim.x * gridDim.x + x;

    extern __shared__ int tile[];

    tile[threadIdx.x + (blockDim.x + IPAD_D) * threadIdx.y] = idx;
    // Now the shared memory`s dim.x needs to be blockDim.x + IPAD
    // Just let the row widder because it`s a one-dim array

    __syncthreads();

    int idxInBlock = threadIdx.y * (blockDim.x) + threadIdx.x;
    int col = idxInBlock / blockDim.y;
    int row = idxInBlock % blockDim.y;

    out[idx] = tile[col + row * (blockDim.x + IPAD_D)];
}

// As we can see, if the coordinates written to and read from the shared memory are same(both x or both y), then the result matrix is the input matrix
// else(write into x and read from y or write into y and read from x) will get the transpose matrix
int main() {
    int x = BDIMX, y = BDIMY;
    int nData = x * y;
    int nBytes = nData * sizeof(int );

    int* outH = (int *) malloc(nBytes);
    int* outD;
    cudaMalloc(&outD, nBytes);

    dim3 blockDim(BDIMX, BDIMY);
    dim3 gridDim((x + blockDim.x - 1) / blockDim.x, (y + blockDim.y - 1) / blockDim.y);

    printf("blockDim:(%d, %d), gridDim: (%d, %d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    setRowReadRow<<<gridDim, blockDim>>>(outD);
    cudaDeviceSynchronize();
    cudaMemcpy(outH, outD, nBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nData; i++) {
        printf("%d->", outH[i]);
    }
    printf("\n\n");
    memset(outH, 0, nData);

    setColReadRow<<<gridDim, blockDim>>>(outD);
    cudaDeviceSynchronize();
    cudaMemcpy(outH, outD, nBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nData; i++) {
        printf("%d->", outH[i]);
    }
    printf("\n\n");
    memset(outH, 0, nData);

    setRowReadColDyn<<<gridDim, blockDim, BDIMX * BDIMY * sizeof(int )>>>(outD);
    cudaDeviceSynchronize();
    cudaMemcpy(outH, outD, nBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nData; i++) {
        printf("%d->", outH[i]);
    }
    printf("\n\n");
    memset(outH, 0, nData);

    setRowReadColPad<<<gridDim, blockDim>>>(outD);
    cudaDeviceSynchronize();
    cudaMemcpy(outH, outD, nBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nData; i++) {
        printf("%d->", outH[i]);
    }
    printf("\n\n");
    memset(outH, 0, nData);

    setRowReadColDynPad<<<gridDim, blockDim, (BDIMX + IPAD_D) * BDIMY * sizeof(int )>>>(outD);
    cudaDeviceSynchronize();
    cudaMemcpy(outH, outD, nBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nData; i++) {
        printf("%d->", outH[i]);
    }
    printf("\n\n");

    cudaFree(outD);
    free(outH);

    return 0;
}