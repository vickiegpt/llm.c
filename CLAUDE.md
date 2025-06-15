# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### CPU Build
```bash
make train_gpt2
# Run with: OMP_NUM_THREADS=8 ./train_gpt2
```

### Single GPU Build (Mixed Precision)
```bash
make train_gpt2cu
# Run with: ./train_gpt2cu
```

### Single GPU Build (FP32 only)
```bash
make train_gpt2fp32cu
# Run with: ./train_gpt2fp32cu
```

### Multi-GPU Build
```bash
make train_gpt2cu
# Run with: mpirun -np <number of GPUs> ./train_gpt2cu
```

### With cuDNN Support
```bash
make train_gpt2cu USE_CUDNN=1
```

### Enhanced MPI Build
```bash
make train_gpt2_mpi USE_ENHANCED_MPI=1
# Run with: mpirun -np <num_processes> ./train_gpt2_mpi --data-parallel 4 --model-parallel 2
```

### WASI-NN Build
```bash
make train_gpt2_wasi USE_WASI_NN=1
# Generates: train_gpt2_wasi.wasm (110KB WebAssembly module)
# Note: Requires WASI-NN enabled runtime for execution
```

### Test Builds
```bash
# CPU test
make test_gpt2 && ./test_gpt2

# GPU test (FP32)
make test_gpt2cu PRECISION=FP32 && ./test_gpt2cu

# GPU test (Mixed precision with cuDNN)
make test_gpt2cu USE_CUDNN=1 && ./test_gpt2cu
```

### Clean Build
```bash
make clean
```

## High-Level Architecture

llm.c implements GPT-2/GPT-3 training in pure C/CUDA with minimal dependencies. The codebase has parallel implementations in both C/CUDA and PyTorch for verification.

### Core Components

1. **Main Training Files**
   - `train_gpt2.c`: CPU implementation (~1000 lines)
   - `train_gpt2.cu`: GPU implementation with mixed precision
   - `train_gpt2_fp32.cu`: Simplified GPU implementation (FP32 only)
   - `train_gpt2.py`: PyTorch reference implementation

2. **Kernel Library** (`llmc/`)
   - Individual CUDA kernels for each operation (attention, layernorm, matmul, etc.)
   - Each kernel has multiple implementations with different performance characteristics
   - Headers provide clean interfaces to switch between implementations

3. **Model Loading and Checkpointing**
   - Models are loaded from `.bin` files containing weights
   - Debug states can be saved/loaded for testing against PyTorch
   - Supports both random initialization and loading pre-trained weights

4. **Data Loading**
   - Tokenized data is stored in `.bin` format with a 1024-byte header
   - DataLoader supports multi-GPU data parallel training
   - Efficient memory-mapped file reading for large datasets

5. **Multi-GPU Support**
   - Uses NCCL for collective operations
   - Supports data parallelism with gradient synchronization
   - Can be initialized via MPI, filesystem, or TCP sockets

### Key Design Decisions

- **Minimal Dependencies**: Only requires standard C libraries, CUDA toolkit, and optionally NCCL/MPI
- **Educational Focus**: Code is written to be readable and well-documented
- **Performance**: Achieves comparable or better performance than PyTorch while remaining simple
- **Modularity**: Kernels can be easily swapped or modified for experimentation
- **Enhanced MPI**: New MPI wrapper provides advanced distributed training capabilities
- **WASI-NN Integration**: Enables inference in WebAssembly environments with backend abstraction

## Common Development Tasks

### Running a Single Test
```bash
# For component tests
make -C dev/test test_dataloader && ./dev/test/test_dataloader

# For full model tests
make test_gpt2cu && ./test_gpt2cu
```

### Profiling
```bash
# Use nvprof or nsys for CUDA profiling
nsys profile ./train_gpt2cu
```

### Debugging
```bash
# Build with debug symbols (replace -O3 with -g in Makefile)
make train_gpt2cu CFLAGS="-g"
# Then use cuda-gdb or your preferred debugger
```

### Data Preparation
```bash
# Tokenize a dataset (example with TinyShakespeare)
python dev/data/tinyshakespeare.py
```

## Code Style Guidelines

- Keep code simple and readable - this is an educational project
- Document non-obvious optimizations
- Prefer clarity over minor performance gains
- Match existing code style in the file you're editing
- Use existing patterns for new kernel implementations