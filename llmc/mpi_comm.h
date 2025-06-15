/*
MPI Communication Wrapper for llm.c
Provides high-level MPI operations for distributed training
*/

#ifndef MPI_COMM_H
#define MPI_COMM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#if defined(USE_MPI) || defined(USE_ENHANCED_MPI)
#include <mpi.h>
#endif

// Error handling
#define MPI_CHECK(call)                                              \
    do {                                                             \
        int mpi_status = (call);                                     \
        if (mpi_status != MPI_SUCCESS) {                            \
            char mpi_error_string[MPI_MAX_ERROR_STRING];           \
            int mpi_error_string_length = 0;                        \
            MPI_Error_string(mpi_status, mpi_error_string,         \
                             &mpi_error_string_length);             \
            fprintf(stderr,                                         \
                    "[MPI ERROR] %s at %s:%d. Error: %s\n",       \
                    #call, __FILE__, __LINE__, mpi_error_string);  \
            MPI_Abort(MPI_COMM_WORLD, mpi_status);                 \
        }                                                           \
    } while (0)

// Forward declarations
typedef struct mpi_context mpi_context_t;

// Function pointer types for MPI operations
typedef void (*mpi_allreduce_fn)(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                                  size_t count, MPI_Datatype datatype, MPI_Op op);
typedef void (*mpi_broadcast_fn)(mpi_context_t* ctx, void* buffer, size_t count, 
                                  MPI_Datatype datatype, int root);
typedef void (*mpi_gather_fn)(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                               size_t count, MPI_Datatype datatype, int root);
typedef void (*mpi_scatter_fn)(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                                size_t count, MPI_Datatype datatype, int root);
typedef void (*mpi_barrier_fn)(mpi_context_t* ctx);

// MPI context structure
struct mpi_context {
    int rank;                    // Process rank
    int size;                    // Total number of processes
    int local_rank;              // Rank within the node
    int local_size;              // Number of processes on this node
    MPI_Comm world_comm;         // Global communicator
    MPI_Comm local_comm;         // Node-local communicator
    
    // Operation function pointers
    mpi_allreduce_fn all_reduce;
    mpi_broadcast_fn broadcast;
    mpi_gather_fn gather;
    mpi_scatter_fn scatter;
    mpi_barrier_fn barrier;
    
    // Performance tracking
    double total_comm_time;
    size_t total_comm_bytes;
    size_t comm_count;
};

// MPI context initialization and cleanup
#if defined(USE_MPI) || defined(USE_ENHANCED_MPI)

mpi_context_t* mpi_context_init(int* argc, char*** argv);
void mpi_context_free(mpi_context_t* ctx);

// Default implementations of MPI operations
void mpi_allreduce_impl(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                         size_t count, MPI_Datatype datatype, MPI_Op op);
void mpi_broadcast_impl(mpi_context_t* ctx, void* buffer, size_t count, 
                         MPI_Datatype datatype, int root);
void mpi_gather_impl(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                      size_t count, MPI_Datatype datatype, int root);
void mpi_scatter_impl(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                       size_t count, MPI_Datatype datatype, int root);
void mpi_barrier_impl(mpi_context_t* ctx);

// High-level operations for ML workloads
void mpi_allreduce_gradients(mpi_context_t* ctx, float* gradients, size_t count);
void mpi_broadcast_parameters(mpi_context_t* ctx, float* parameters, size_t count, int root);
void mpi_gather_statistics(mpi_context_t* ctx, float* local_stats, float* global_stats, 
                            size_t count, int root);

// Utility functions
void mpi_print_stats(mpi_context_t* ctx);
int mpi_get_device_assignment(mpi_context_t* ctx, int num_devices_per_node);

#else // !USE_MPI

// Stub implementations when MPI is not available
static inline mpi_context_t* mpi_context_init(int* argc, char*** argv) {
    mpi_context_t* ctx = (mpi_context_t*)calloc(1, sizeof(mpi_context_t));
    ctx->rank = 0;
    ctx->size = 1;
    ctx->local_rank = 0;
    ctx->local_size = 1;
    return ctx;
}

static inline void mpi_context_free(mpi_context_t* ctx) {
    free(ctx);
}

static inline void mpi_allreduce_gradients(mpi_context_t* ctx, float* gradients, size_t count) {
    // No-op in single process mode
}

static inline void mpi_broadcast_parameters(mpi_context_t* ctx, float* parameters, 
                                             size_t count, int root) {
    // No-op in single process mode
}

static inline void mpi_gather_statistics(mpi_context_t* ctx, float* local_stats, 
                                          float* global_stats, size_t count, int root) {
    if (global_stats != local_stats) {
        memcpy(global_stats, local_stats, count * sizeof(float));
    }
}

static inline void mpi_print_stats(mpi_context_t* ctx) {
    printf("MPI not enabled\n");
}

static inline int mpi_get_device_assignment(mpi_context_t* ctx, int num_devices_per_node) {
    return 0;
}

#endif // USE_MPI

#endif // MPI_COMM_H