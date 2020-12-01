NVCC = /usr/local/cuda-10.0/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"
INC = -I/usr/local/cuda-10.0/sample/common/inc/

all: main

main: main.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@