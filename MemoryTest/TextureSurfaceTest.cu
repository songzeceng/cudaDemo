//
// Created by songzeceng on 2020/11/10.
//

#include "cuda_runtime.h"
#include "texture_fetch_functions.h"
#include "surface_indirect_functions.h"
#include "surface_functions.h"

#include <stdio.h>

void textureObjTest();

void bindTextureRef();

void surfaceObjTest();

void surfaceRefCallKernel();

void surfaceRefBind();

int width = 5, height = 2;
int size = width * height;
float *h_data = (float *) malloc(size * sizeof(float));
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
const surface<void, cudaSurfaceType2D> inputSurf, outputSurf;

__global__ void transformKernel(float *output, cudaTextureObject_t texObj, int width, int height, float theta) {
    unsigned int x = blockIdx.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockIdx.y + threadIdx.y;

    float u = x / (float) width - 0.5f;
    float v = y / (float) height - 0.5f;

    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    output[y * width + x] = tex2D<float>(texObj, tu, tv);
}

__global__ void transformKernelRef(float *output, int width, int height, float theta) {
    unsigned int x = blockIdx.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockIdx.y + threadIdx.y;

    float u = x / (float) width - 0.5f;
    float v = y / (float) height - 0.5f;

    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    output[y * width + x] = tex2D(texRef, tu, tv);
}

__global__ void transformKernelSurfaceObj(cudaSurfaceObject_t inputSurObj, cudaSurfaceObject_t outputSurObj,
                                          int width, int height) {
    unsigned int x = blockIdx.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height) {
        uchar4 data;
        surf2Dread(&data, inputSurObj, 4 * x, y);
        surf2Dwrite(data, outputSurObj, 4 * x, y);
    }
}

__global__ void transformKernelSurfaceRef(int width, int height) {
    unsigned int x = blockIdx.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height) {
        uchar4 data;
        surf2Dread(&data, inputSurf, 4 * x, y);
        surf2Dwrite(data, outputSurf, 4 * x, y);
    }
}

int main() {
//    textureObjTest();
//    bindTextureRef();
//    surfaceObjTest();
//    surfaceRefBind();

    //    surfaceRefCallKernel();

    return 0;
}

void surfaceRefBind() {
    /////////////////////////////////低级api，绑定到cuda数组//////////////////////////////////////////////
//    const surface<void, cudaSurfaceType2D> surfRef;
//    const surfaceReference* surRefPtr;
//    cudaGetSurfaceReference(&surRefPtr, "surRef");
//    cudaChannelFormatDesc desc;
//    cudaArray* cuArray;
//    cudaGetChannelDesc(&desc, cuArray);
//    cudaBindSurfaceToArray(surfRef, cuArray);

    /////////////////////////////////高级api，绑定到cuda数组//////////////////////////////////////////////
//    cudaArray* cuArray;
//    cudaChannelFormatDesc des = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
//    cudaMallocArray(&cuArray, &des, width, height, cudaArraySurfaceLoadStore);
//    surface<void, cudaSurfaceType2D> surfRef;
//    cudaBindSurfaceToArray(surfRef, cuArray);
}

void surfaceRefCallKernel() {
    /////////////////////////////////表面引用调用核函数//////////////////////////////////////////////
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }

    cudaChannelFormatDesc des = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &des, width, height, cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &des, width, height, cudaArraySurfaceLoadStore);

    cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaBindSurfaceToArray(inputSurf, cuInputArray);
    cudaBindSurfaceToArray(outputSurf, cuOutputArray);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    transformKernelSurfaceRef<<<gridDim, blockDim>>>(width, height);

    float *h_result = (float *) malloc(size * sizeof(float));
    cudaMemcpyFromArray(h_result, cuOutputArray, 0, 0, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        printf("%f\n", h_result[i]);
    }

    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);
}

void surfaceObjTest() {
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }

    cudaChannelFormatDesc des = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &des, width, height, cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &des, width, height, cudaArraySurfaceLoadStore);

    cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    struct cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;

    resourceDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t cuInputObj = 0, cuOutputObj = 0;
    cudaCreateSurfaceObject(&cuInputObj, &resourceDesc);
    resourceDesc.res.array.array = cuOutputArray;
    cudaCreateSurfaceObject(&cuOutputObj, &resourceDesc);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    transformKernelSurfaceObj<<<gridDim, blockDim>>>(cuInputObj, cuOutputObj, width, height);

    float *h_result = (float *) malloc(size * sizeof(float));
    cudaMemcpyFromArray(h_result, cuOutputArray, 0, 0, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        printf("%f\n", h_result[i]);
    }

    cudaDestroySurfaceObject(cuInputObj);
    cudaDestroySurfaceObject(cuOutputObj);
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);
}

void bindTextureRef() {
    /////////////////////////////////低级api，绑定到线性内存//////////////////////////////////////////////
//    texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
//    const textureReference *texRefPtr;
//
//    cudaGetTextureReference(&texRefPtr, &texRef);
//    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
//
//    float *devPtr;
//    size_t pitch;
//    cudaMallocPitch((void **) &devPtr, &pitch, width * sizeof(float), height);
//
//    size_t offset;
//    cudaBindTexture2D(&offset, texRefPtr, devPtr, &desc, width, height, pitch);

    //////////////////////////////////高级API，绑定到线性内存////////////////////////////////////////////////

//    texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
//    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
//
//    float *devPtr;
//    size_t pitch;
//    cudaMallocPitch((void **) &devPtr, &pitch, width * sizeof(float), height);
//
//    size_t offset;
//    cudaBindTexture2D(&offset, texRef, devPtr, desc, width, height, pitch);

    //////////////////////////////////低级API，绑定到cuda数组////////////////////////////////////////////////
//    texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
//    const textureReference* texRefPtr;
//
//    cudaGetTextureReference(&texRefPtr, &texRef);
//
//    cudaChannelFormatDesc desc;
//
//    cudaArray *cudaArray;
//    cudaMallocArray(&cudaArray, &desc, width, height);
//
//    cudaGetChannelDesc(&desc, cudaArray);
//
//    cudaBindTextureToArray(texRef, cudaArray);

    //////////////////////////////////高级API，绑定到cuda数组////////////////////////////////////////////////
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }

    cudaArray *cudaArray;
    cudaMallocArray(&cudaArray, &desc, width, height);
    cudaMemcpyToArray(cudaArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);

    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = 1;

    cudaBindTextureToArray(texRef, cudaArray);

    float *output;
    cudaMalloc(&output, size * sizeof(float));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    printf("Ready to call kernel.\n");
    transformKernelRef<<<dimGrid, dimBlock>>>(output, width, height, 0.5f);

    float *h_result = (float *) malloc(size * sizeof(float));
    cudaMemcpy(h_result, output, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Kernel is finished and here`s the result:\n");
    for (int i = 0; i < size; i++) {
        printf("%f\n", h_result[i]);
    }

    cudaFreeArray(cudaArray);
    cudaFree(output);
}

void textureObjTest() {
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }

    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray *cudaArray;

    printf("data initialized.\n");

    cudaMallocArray(&cudaArray, &desc, width, height);

    cudaMemcpyToArray(cudaArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    printf("Texture object created.\n");

    float *output;
    cudaMalloc(&output, size * sizeof(float));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    printf("Ready to call kernel.\n");
    transformKernel<<<dimGrid, dimBlock>>>(output, texObj, width, height, 0.5f);

    float *h_result = (float *) malloc(size * sizeof(float));
    cudaMemcpy(h_result, output, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Kernel is finished and here`s the result:\n");
    for (int i = 0; i < size; i++) {
        printf("%f\n", h_result[i]);
    }

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cudaArray);
    cudaFree(output);
}
