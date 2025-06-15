/*
MPI Communication Wrapper Implementation for llm.c
*/

#include "mpi_comm.h"
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(USE_MPI) || defined(USE_ENHANCED_MPI)

// Helper function to get hostname hash for local rank calculation
static unsigned int get_hostname_hash() {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    unsigned int hash = 5381;
    for (int i = 0; hostname[i] != '\0'; i++) {
        hash = ((hash << 5) + hash) + hostname[i];
    }
    return hash;
}

// Initialize MPI context
mpi_context_t* mpi_context_init(int* argc, char*** argv) {
    mpi_context_t* ctx = (mpi_context_t*)calloc(1, sizeof(mpi_context_t));
    if (!ctx) {
        fprintf(stderr, "Failed to allocate MPI context\n");
        exit(1);
    }
    
    // Initialize MPI
    int provided;
    MPI_CHECK(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided));
    
    // Get basic MPI info
    ctx->world_comm = MPI_COMM_WORLD;
    MPI_CHECK(MPI_Comm_rank(ctx->world_comm, &ctx->rank));
    MPI_CHECK(MPI_Comm_size(ctx->world_comm, &ctx->size));
    
    // Calculate local rank using hostname hashing
    unsigned int* hostname_hashes = (unsigned int*)malloc(ctx->size * sizeof(unsigned int));
    unsigned int local_hash = get_hostname_hash();
    
    MPI_CHECK(MPI_Allgather(&local_hash, 1, MPI_UNSIGNED, 
                            hostname_hashes, 1, MPI_UNSIGNED, ctx->world_comm));
    
    // Count processes on same node
    ctx->local_rank = 0;
    ctx->local_size = 0;
    for (int i = 0; i < ctx->size; i++) {
        if (hostname_hashes[i] == local_hash) {
            if (i < ctx->rank) ctx->local_rank++;
            ctx->local_size++;
        }
    }
    
    // Create node-local communicator
    MPI_CHECK(MPI_Comm_split(ctx->world_comm, local_hash, ctx->rank, &ctx->local_comm));
    
    free(hostname_hashes);
    
    // Set function pointers
    ctx->all_reduce = mpi_allreduce_impl;
    ctx->broadcast = mpi_broadcast_impl;
    ctx->gather = mpi_gather_impl;
    ctx->scatter = mpi_scatter_impl;
    ctx->barrier = mpi_barrier_impl;
    
    // Initialize stats
    ctx->total_comm_time = 0.0;
    ctx->total_comm_bytes = 0;
    ctx->comm_count = 0;
    
    if (ctx->rank == 0) {
        printf("MPI initialized: %d processes total, %d per node\n", 
               ctx->size, ctx->local_size);
    }
    
    return ctx;
}

// Cleanup MPI context
void mpi_context_free(mpi_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->rank == 0) {
        mpi_print_stats(ctx);
    }
    
    if (ctx->local_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&ctx->local_comm);
    }
    
    MPI_CHECK(MPI_Finalize());
    free(ctx);
}

// AllReduce implementation with timing
void mpi_allreduce_impl(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                         size_t count, MPI_Datatype datatype, MPI_Op op) {
    double start_time = MPI_Wtime();
    
    MPI_CHECK(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, ctx->world_comm));
    
    double end_time = MPI_Wtime();
    ctx->total_comm_time += (end_time - start_time);
    
    // Calculate bytes transferred
    int type_size;
    MPI_Type_size(datatype, &type_size);
    ctx->total_comm_bytes += count * type_size * (ctx->size - 1);
    ctx->comm_count++;
}

// Broadcast implementation with timing
void mpi_broadcast_impl(mpi_context_t* ctx, void* buffer, size_t count, 
                         MPI_Datatype datatype, int root) {
    double start_time = MPI_Wtime();
    
    MPI_CHECK(MPI_Bcast(buffer, count, datatype, root, ctx->world_comm));
    
    double end_time = MPI_Wtime();
    ctx->total_comm_time += (end_time - start_time);
    
    // Calculate bytes transferred
    int type_size;
    MPI_Type_size(datatype, &type_size);
    ctx->total_comm_bytes += count * type_size;
    ctx->comm_count++;
}

// Gather implementation
void mpi_gather_impl(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                      size_t count, MPI_Datatype datatype, int root) {
    double start_time = MPI_Wtime();
    
    MPI_CHECK(MPI_Gather(sendbuf, count, datatype, 
                         recvbuf, count, datatype, root, ctx->world_comm));
    
    double end_time = MPI_Wtime();
    ctx->total_comm_time += (end_time - start_time);
    
    int type_size;
    MPI_Type_size(datatype, &type_size);
    ctx->total_comm_bytes += count * type_size * ctx->size;
    ctx->comm_count++;
}

// Scatter implementation
void mpi_scatter_impl(mpi_context_t* ctx, void* sendbuf, void* recvbuf, 
                       size_t count, MPI_Datatype datatype, int root) {
    double start_time = MPI_Wtime();
    
    MPI_CHECK(MPI_Scatter(sendbuf, count, datatype,
                          recvbuf, count, datatype, root, ctx->world_comm));
    
    double end_time = MPI_Wtime();
    ctx->total_comm_time += (end_time - start_time);
    
    int type_size;
    MPI_Type_size(datatype, &type_size);
    ctx->total_comm_bytes += count * type_size * ctx->size;
    ctx->comm_count++;
}

// Barrier implementation
void mpi_barrier_impl(mpi_context_t* ctx) {
    double start_time = MPI_Wtime();
    
    MPI_CHECK(MPI_Barrier(ctx->world_comm));
    
    double end_time = MPI_Wtime();
    ctx->total_comm_time += (end_time - start_time);
    ctx->comm_count++;
}

// High-level gradient allreduce for ML
void mpi_allreduce_gradients(mpi_context_t* ctx, float* gradients, size_t count) {
    // Average gradients across all processes
    ctx->all_reduce(ctx, MPI_IN_PLACE, gradients, count, MPI_FLOAT, MPI_SUM);
    
    // Scale by 1/n to get average
    float scale = 1.0f / ctx->size;
    for (size_t i = 0; i < count; i++) {
        gradients[i] *= scale;
    }
}

// Broadcast parameters from root
void mpi_broadcast_parameters(mpi_context_t* ctx, float* parameters, size_t count, int root) {
    ctx->broadcast(ctx, parameters, count, MPI_FLOAT, root);
}

// Gather statistics from all processes
void mpi_gather_statistics(mpi_context_t* ctx, float* local_stats, float* global_stats, 
                            size_t count, int root) {
    ctx->gather(ctx, local_stats, global_stats, count, MPI_FLOAT, root);
}

// Print communication statistics
void mpi_print_stats(mpi_context_t* ctx) {
    if (ctx->comm_count == 0) return;
    
    printf("MPI Communication Statistics (Rank %d):\n", ctx->rank);
    printf("  Total operations: %zu\n", ctx->comm_count);
    printf("  Total time: %.3f seconds\n", ctx->total_comm_time);
    printf("  Total data: %.3f GB\n", ctx->total_comm_bytes / 1e9);
    printf("  Average bandwidth: %.3f GB/s\n", 
           (ctx->total_comm_bytes / 1e9) / ctx->total_comm_time);
    printf("  Average latency: %.3f ms\n", 
           (ctx->total_comm_time / ctx->comm_count) * 1000);
}

// Get device assignment based on local rank
int mpi_get_device_assignment(mpi_context_t* ctx, int num_devices_per_node) {
    return ctx->local_rank % num_devices_per_node;
}

#endif // USE_MPI