cmake_minimum_required(VERSION 3.17)
project(CudaDemo CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaDemo main.cu)
add_executable(CudaDemo_ DeviceTest.cu DeviceTest.cuh)
add_executable(TextureSurfaceTest TextureSurfaceTest.cu)

set_target_properties(
        CudaDemo
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)