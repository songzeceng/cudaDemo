NVCC = /usr/local/cuda-10.0/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"
INC = -I/usr/local/cuda-10.0/samples/common/inc/

all: main

main: main.o kernel.o
	$(NVCC) $^ -o $@

main.o: main.cpp kernel.cuh
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

kernel.o: kernel.cu kernel.cuh
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@