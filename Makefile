CXX ?= g++
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda

BUILD_DIR := build
BIN_DIR := bin

CXXFLAGS ?= -O3 -fopenmp -march=native -std=c++17
NVCCFLAGS ?= -O3 -Xcompiler -fopenmp -arch=sm_86 -std=c++17
LDLIBS ?= -L$(CUDA_HOME)/lib64 -lcudart -lcublas -fopenmp -lstdc++fs

.PHONY: all masked unprotected clean

all: masked unprotected

masked: $(BIN_DIR)/fusion_masked_artifact
unprotected: $(BIN_DIR)/fusion_unprotected_artifact

$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

$(BIN_DIR)/fusion_masked_artifact: $(BUILD_DIR)/main_masked_fusion_v5.o $(BUILD_DIR)/cpa_gpu_masked_fusion_v5.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/main_masked_fusion_v5.o: src/main_masked_fusion_v5.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/cpa_gpu_masked_fusion_v5.o: src/cpa_gpu_masked_fusion_v5.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/fusion_unprotected_artifact: $(BUILD_DIR)/main_fusion_unmask_v0.o $(BUILD_DIR)/cpa_gpu_fusion_unmask_v0.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/main_fusion_unmask_v0.o: src/main_fusion_unmask_v0.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/cpa_gpu_fusion_unmask_v0.o: src/cpa_gpu_fusion_unmask_v0.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

	