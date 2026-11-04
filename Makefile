CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_89
CXXFLAGS ?= -O3 -Wall -Wextra
NVCCFLAGS ?= -O3

SOURCES := src/cualign.cpp src/CuAlignCpu.cpp
HEADERS := src/CuAlign.hpp src/Graph.hpp
CUDA_HOST_OBJECTS := build/cualign.o build/CuAlignCpu.o

.PHONY: all cuda test test-cuda
all: cualign
cuda: cualign.cuda

cualign: $(SOURCES) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++11 -ffp-contract=off $(LDFLAGS) -o $@ $(SOURCES) $(LDLIBS)

build/%.o: src/%.cpp $(HEADERS)
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++11 -ffp-contract=off -DCUALIGN_CUDA -c $< -o $@

build/CuAlignCuda.o: src/CuAlignCuda.cu src/CuAlign.hpp
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -std=c++14 -arch=$(CUDA_ARCH) --fmad=false -c $< -o $@

cualign.cuda: $(CUDA_HOST_OBJECTS) build/CuAlignCuda.o
	$(NVCC) -arch=$(CUDA_ARCH) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test: cualign
	python3 test/check_cualign.py ./cualign

test-cuda: cualign cualign.cuda
	python3 test/check_cualign.py ./cualign ./cualign.cuda
