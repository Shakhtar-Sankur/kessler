CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fPIC -Wall -Wextra -fopenmp -ffp-contract=off
LIB = kessler/_libkessler.so

NVCC ?= nvcc
CUDA_ARCH ?= -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
CUDA_LIB = kessler/_libkessler_cuda.so

all: $(LIB)

cuda: $(CUDA_LIB)

$(CUDA_LIB): src/cuda.cu include/kessler/sgp4.hpp include/kessler/screen.hpp
	$(NVCC) -O3 -std=c++17 -Xcompiler -fPIC -shared -Iinclude --fmad=false $(CUDA_ARCH) src/cuda.cu -o $@

$(LIB): src/capi.cpp src/screen.cpp src/mc.cpp include/kessler/sgp4.hpp include/kessler/screen.hpp
	$(CXX) $(CXXFLAGS) -Iinclude -shared src/capi.cpp src/screen.cpp src/mc.cpp -o $@

clean:
	rm -f $(LIB) $(CUDA_LIB)

.PHONY: all cuda clean
