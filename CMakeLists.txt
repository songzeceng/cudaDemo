cmake_minimum_required(VERSION 3.13)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.0/bin/nvcc)
project(CudaDemo CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(NVCC_FLAGS_EXTRA "-gencode arch=compute_61,code=sm_61 -rdc=true")

add_executable(CudaDemo main.cu)
add_executable(CudaDemo_ PropertyTest/DeviceTest.cu)
add_executable(TextureSurfaceTest MemoryTest/TextureSurfaceTest.cu)
add_executable(OccupancyTest OptimizeTest/OccupancyTest.cu)
add_executable(matmul_kernel MatrixSimple/matmul_kernel.cu)
add_executable(matmul_shared MemoryTest/SharedMemory/matmul_shared.cu)
add_executable(SumMatrix MatrixSimple/SumMatrix.cu)
add_executable(ThreadIdLocate ThreadTest/ThreadIdLocate.cu)
add_executable(ThreadFenceTest ThreadTest/ThreadFenceTest.cu)
add_executable(UnrollTest OptimizeTest/UnrollTest.cu)
add_executable(DynamicParallelTest OptimizeTest/DynamicParallelTest.cu)
add_executable(IntrinsicsTest InstructionTest/IntrinsicsTest.cu)
add_executable(SharedMemoryStaticDynamicTest MemoryTest/SharedMemory/SharedMemoryStaticDynamicTest.cu)
add_executable(SharedMemoryReduceTest MemoryTest/SharedMemory/SharedMemoryReduceTest.cu)
add_executable(transposeMatrixWithSharedMemory MemoryTest/SharedMemory/transposeMatrixWithSharedMemory.cu)
add_executable(ConstantMemoryTest MemoryTest/ConstantMemoryTest.cu)
add_executable(ShuffleTest InstructionTest/ShuffleTest.cu)
add_executable(ConcurrentKernelInNotNullStream StreamEventTest/ConcurrentKernelInNotNullStream.cu)
add_executable(StreamDependencyTest StreamEventTest/StreamDependencyTest.cu)
add_executable(OverlapKernelAndDataProcessTest StreamEventTest/OverlapKernelAndDataProcessTest.cu)
add_executable(OverlapGPUCPU StreamEventTest/OverlapGPUCPU.cu)
add_executable(dist_v1_cuda high_performance_computing/dist_v1_cuda/kernel.cu)
add_executable(dist_v2_cuda high_performance_computing/distance_v2_cuda/kernel_unified.cu)
add_executable(dist_2d high_performance_computing/dist_2d/dist_2d.cu)

set_target_properties(
        CudaDemo
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)