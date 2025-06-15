/*
Enhanced MPI Training for GPT-2
Demonstrates distributed training with the new MPI wrapper
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#ifdef USE_ENHANCED_MPI
#include "llmc/mpi_comm.h"
#endif

// Note: This is a demonstration of MPI training - in practice you'd
// integrate with the actual GPT-2 training code from train_gpt2.c

// Enhanced MPI training configuration
typedef struct {
    int data_parallel_size;
    int model_parallel_size;
    int pipeline_parallel_size;
    float gradient_clip_norm;
    int sync_frequency;
} mpi_training_config_t;

// Global MPI context
#if defined(USE_ENHANCED_MPI) || defined(USE_MPI)
static mpi_context_t* g_mpi_ctx = NULL;
#endif

void print_usage() {
    printf("Usage: ./train_gpt2_mpi [options]\n");
    printf("Options:\n");
    printf("  --data-parallel <n>      Data parallel size (default: MPI size)\n");
    printf("  --model-parallel <n>     Model parallel size (default: 1)\n");
    printf("  --pipeline-parallel <n>  Pipeline parallel size (default: 1)\n");
    printf("  --gradient-clip <f>      Gradient clipping norm (default: 1.0)\n");
    printf("  --sync-freq <n>          Gradient sync frequency (default: 1)\n");
    printf("  --help                   Show this help\n");
}

int main(int argc, char* argv[]) {
#if defined(USE_ENHANCED_MPI) || defined(USE_MPI)
    // Initialize MPI
    g_mpi_ctx = mpi_context_init(&argc, &argv);
    if (!g_mpi_ctx) {
        fprintf(stderr, "Failed to initialize MPI context\n");
        return 1;
    }
    
    // Parse command line arguments
    mpi_training_config_t config = {
        .data_parallel_size = g_mpi_ctx->size,
        .model_parallel_size = 1,
        .pipeline_parallel_size = 1,
        .gradient_clip_norm = 1.0f,
        .sync_frequency = 1
    };
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data-parallel") == 0 && i + 1 < argc) {
            config.data_parallel_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--model-parallel") == 0 && i + 1 < argc) {
            config.model_parallel_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pipeline-parallel") == 0 && i + 1 < argc) {
            config.pipeline_parallel_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gradient-clip") == 0 && i + 1 < argc) {
            config.gradient_clip_norm = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sync-freq") == 0 && i + 1 < argc) {
            config.sync_frequency = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            if (g_mpi_ctx->rank == 0) print_usage();
            mpi_context_free(g_mpi_ctx);
            return 0;
        }
    }
    
    // Validate configuration
    if (config.data_parallel_size * config.model_parallel_size * 
        config.pipeline_parallel_size != g_mpi_ctx->size) {
        if (g_mpi_ctx->rank == 0) {
            fprintf(stderr, "Error: data_parallel * model_parallel * pipeline_parallel must equal MPI size (%d)\n", 
                    g_mpi_ctx->size);
        }
        mpi_context_free(g_mpi_ctx);
        return 1;
    }
    
    // Print configuration
    if (g_mpi_ctx->rank == 0) {
        printf("Enhanced MPI Training Configuration:\n");
        printf("  Total processes: %d\n", g_mpi_ctx->size);
        printf("  Data parallel size: %d\n", config.data_parallel_size);
        printf("  Model parallel size: %d\n", config.model_parallel_size);
        printf("  Pipeline parallel size: %d\n", config.pipeline_parallel_size);
        printf("  Gradient clipping norm: %.2f\n", config.gradient_clip_norm);
        printf("  Gradient sync frequency: %d\n", config.sync_frequency);
        printf("  Local rank: %d/%d\n", g_mpi_ctx->local_rank, g_mpi_ctx->local_size);
    }
    
    // Initialize the model (reuse existing GPT-2 initialization)
    // This is a simplified version - in practice you'd integrate with the full training loop
    
    // Simulate training loop with MPI synchronization
    srand(time(NULL) + g_mpi_ctx->rank);
    
    // Allocate dummy gradients for demonstration
    size_t num_params = 124439808; // GPT-2 124M parameters
    float* gradients = (float*)malloc(num_params * sizeof(float));
    
    // Fill with random gradients
    for (size_t i = 0; i < num_params; i++) {
        gradients[i] = ((float)rand() / RAND_MAX) * 0.01f;
    }
    
    printf("Rank %d: Starting training with %zu parameters\n", 
           g_mpi_ctx->rank, num_params);
    
    // Simulate training steps
    for (int step = 0; step < 10; step++) {
        // Simulate forward/backward pass
        usleep(100000); // 100ms
        
        // Synchronize gradients every sync_frequency steps
        if ((step + 1) % config.sync_frequency == 0) {
            double start_time = MPI_Wtime();
            
            // Average gradients across all processes
            mpi_allreduce_gradients(g_mpi_ctx, gradients, num_params);
            
            double end_time = MPI_Wtime();
            
            if (g_mpi_ctx->rank == 0) {
                printf("Step %d: Gradient sync completed in %.3f ms\n", 
                       step + 1, (end_time - start_time) * 1000);
            }
        }
        
        // Apply gradient clipping
        float grad_norm = 0.0f;
        for (size_t i = 0; i < num_params; i++) {
            grad_norm += gradients[i] * gradients[i];
        }
        grad_norm = sqrtf(grad_norm);
        
        if (grad_norm > config.gradient_clip_norm) {
            float scale = config.gradient_clip_norm / grad_norm;
            for (size_t i = 0; i < num_params; i++) {
                gradients[i] *= scale;
            }
        }
        
        if (g_mpi_ctx->rank == 0 && step % 5 == 0) {
            printf("Step %d: Gradient norm: %.6f\n", step + 1, grad_norm);
        }
    }
    
    // Print final statistics
    if (g_mpi_ctx->rank == 0) {
        printf("\nTraining completed successfully!\n");
        mpi_print_stats(g_mpi_ctx);
    }
    
    free(gradients);
    mpi_context_free(g_mpi_ctx);
    
    return 0;
    
#else
    printf("Enhanced MPI support not compiled in. Please build with USE_ENHANCED_MPI=1\n");
    return 1;
#endif
}