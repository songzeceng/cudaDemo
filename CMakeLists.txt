cmake_minimum_required(VERSION 3.13)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.0/bin/nvcc)
project(CudaDemo CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(NVCC_FLAGS_EXTRA "-gencode arch=compute_61,code=sm_61 -rdc=true")

add_executable(CudaDemo main.cu)
add_executable(CudaDemo_ DeviceTest.cu)
add_executable(TextureSurfaceTest TextureSurfaceTest.cu)
add_executable(OccupancyTest OccupancyTest.cu)
add_executable(matmul_kernel matmul_kernel.cu)
add_executable(matmul_shared matmul_shared.cu)
add_executable(SumMatrix SumMatrix.cu)
add_executable(ThreadIdLocate ThreadIdLocate.cu)
add_executable(ThreadFenceTest ThreadFenceTest.cu)
add_executable(UnrollTest UnrollTest.cu)
add_executable(DynamicParallelTest DynamicParallelTest.cu)
add_executable(IntrinsicsTest IntrinsicsTest.cu)
add_executable(SharedMemoryStaticDynamicTest SharedMemoryStaticDynamicTest.cu)
add_executable(SharedMemoryReduceTest SharedMemoryReduceTest.cu)
add_executable(transposeMatrixWithSharedMemory transposeMatrixWithSharedMemory.cu)

set_target_properties(
        CudaDemo
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)