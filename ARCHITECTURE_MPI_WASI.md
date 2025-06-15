# Architecture Design: MPI and WASI-NN Integration for llm.c

## Overview

This document outlines the architectural design for transforming llm.c to support:
1. Enhanced MPI integration for distributed training coordination
2. WASI-NN backend for WebAssembly-based inference

## Current State Analysis

### MPI in llm.c
- Currently used only for NCCL initialization in multi-node setups
- Minimal integration in `llmc/zero.cuh`
- Three initialization methods: MPI, TCP, filesystem

### WASI-NN Requirements
- Inference-only API (no training support)
- Model loading from opaque byte sequences
- Backend-agnostic design
- Supports ONNX, TensorFlow, OpenVINO formats

## Proposed Architecture

### 1. Compute Backend Abstraction Layer

Create a unified interface for compute operations that can target either CUDA or WASI-NN:

```c
// llmc/backend.h
typedef enum {
    BACKEND_CUDA,
    BACKEND_WASI_NN,
    BACKEND_CPU
} backend_type_t;

typedef struct {
    backend_type_t type;
    void* context;  // Backend-specific context
    
    // Function pointers for operations
    void (*matmul)(void* ctx, const float* A, const float* B, float* C, int M, int N, int K);
    void (*layernorm)(void* ctx, float* out, const float* inp, const float* weight, 
                      const float* bias, int N, int C);
    void (*attention)(void* ctx, float* out, const float* inp, int B, int T, int C, int NH);
    // ... other operations
} compute_backend_t;
```

### 2. MPI Enhancement Strategy

#### 2.1 MPI Communication Wrapper
Create a unified communication layer that abstracts MPI operations:

```c
// llmc/mpi_comm.h
typedef struct {
    int rank;
    int size;
    MPI_Comm comm;
    
    // High-level operations
    void (*all_reduce)(void* data, size_t count, MPI_Datatype type);
    void (*broadcast)(void* data, size_t count, MPI_Datatype type, int root);
    void (*gather)(void* sendbuf, void* recvbuf, size_t count, MPI_Datatype type);
} mpi_context_t;
```

#### 2.2 Distributed Training Coordinator
Extend MPI usage beyond NCCL initialization:

```c
// llmc/distributed.h
typedef struct {
    mpi_context_t* mpi_ctx;
    compute_backend_t* backend;
    
    // Model parallel configuration
    int pipeline_stages;
    int tensor_parallel_size;
    int data_parallel_size;
    
    // Gradient synchronization
    void (*sync_gradients)(void* gradients, size_t size);
    void (*sync_parameters)(void* parameters, size_t size);
} distributed_config_t;
```

### 3. WASI-NN Backend Implementation

#### 3.1 Model Export Pipeline
Convert trained models to ONNX format for WASI-NN compatibility:

```python
# tools/export_to_wasi.py
def export_gpt2_to_onnx(checkpoint_path, output_path):
    """Export GPT-2 checkpoint to ONNX format for WASI-NN"""
    # Load checkpoint
    # Convert to ONNX
    # Optimize for inference
    # Save ONNX model
```

#### 3.2 WASI-NN Inference Engine
Implement inference operations using WASI-NN API:

```c
// llmc/wasi_nn_backend.c
typedef struct {
    graph_t model;
    execution_context_t ctx;
    tensor_t* input_tensors;
    tensor_t* output_tensors;
} wasi_nn_context_t;

void wasi_nn_load_model(wasi_nn_context_t* ctx, const uint8_t* model_data, size_t size) {
    // Load ONNX model using WASI-NN
    load(&model_data, 1, GRAPH_ENCODING_ONNX, EXECUTION_TARGET_CPU, &ctx->model);
}

void wasi_nn_forward(wasi_nn_context_t* ctx, float* input, float* output) {
    // Set input tensors
    // Execute inference
    // Get output tensors
}
```

### 4. Build System Updates

#### 4.1 Makefile Modifications
```makefile
# WASI target support
WASI_SDK ?= /opt/wasi-sdk
WASI_CC = $(WASI_SDK)/bin/clang
WASI_CFLAGS = --target=wasm32-wasi --sysroot=$(WASI_SDK)/share/wasi-sysroot

# Build targets
train_gpt2_wasi: train_gpt2_wasi.c
    $(WASI_CC) $(WASI_CFLAGS) -o $@ $< -lwasi-nn

# MPI-enhanced build
train_gpt2_mpi: train_gpt2_mpi.c
    mpicc -o $@ $< $(CFLAGS) $(LDFLAGS) -lmpi
```

### 5. Implementation Phases

#### Phase 1: Backend Abstraction
1. Create backend interface header
2. Refactor existing CUDA code to use backend interface
3. Implement CPU backend using existing train_gpt2.c code

#### Phase 2: Enhanced MPI Integration
1. Create MPI communication wrapper
2. Implement distributed training coordinator
3. Add support for model parallelism patterns

#### Phase 3: WASI-NN Backend
1. Implement model export tools
2. Create WASI-NN backend implementation
3. Build WASI target support

#### Phase 4: Integration and Testing
1. Integrate all components
2. Create test suite for different backends
3. Performance benchmarking

## Usage Examples

### MPI Distributed Training
```bash
# Compile with enhanced MPI support
make train_gpt2_mpi

# Run across 4 nodes with model parallelism
mpirun -np 16 ./train_gpt2_mpi \
    --tensor-parallel 2 \
    --pipeline-parallel 2 \
    --data-parallel 4
```

### WASI-NN Inference
```bash
# Export model to ONNX
python tools/export_to_wasi.py model.bin model.onnx

# Compile WASI module
make train_gpt2_wasi

# Run inference with WASI runtime
wasmtime --wasi-nn model.wasm --model model.onnx
```

## Benefits

1. **Portability**: WASI-NN enables running models in WebAssembly environments
2. **Scalability**: Enhanced MPI support for better distributed training
3. **Flexibility**: Backend abstraction allows easy addition of new compute targets
4. **Performance**: Maintain existing CUDA performance while adding new capabilities

## Challenges and Considerations

1. **WASI-NN Limitations**: Currently inference-only, no training support
2. **Performance Overhead**: Abstraction layer may introduce slight overhead
3. **Model Conversion**: Need robust ONNX export pipeline
4. **Testing Complexity**: Multiple backends increase testing surface area