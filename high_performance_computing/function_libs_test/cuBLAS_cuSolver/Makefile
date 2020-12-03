NVCC = /usr/local/cuda-10.0/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"
LIBS = -lcublas_static -lcusolver_static -lculibos

all: main

main: main.cpp
	$(NVCC) $^ -o $@ $(LIBS) $(NVCC_FLAFS)
