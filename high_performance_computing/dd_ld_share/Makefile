UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framwork,GLUT
else
	LDFLAGS += -L/usr/local/cuda-10.0/samples/commons/lib/linux/x86_64 -lglut -lGL -lGLU -lGLEW
endif
NVCC = /usr/local/cuda-10.0/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"
INC = -I/usr/local/cuda-10.0/sample/common/inc/

all: main

main: main.o kernel.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@