/*
WASI-NN Inference for GPT-2
Demonstrates neural network inference using WebAssembly System Interface
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_WASI_NN
#include "llmc/wasi_nn_backend.h"
#endif

// Token buffer size
#define MAX_SEQ_LEN 1024
#define MAX_VOCAB_SIZE 50257

typedef struct {
    char* model_path;
    int max_tokens;
    float temperature;
    int top_k;
    int seed;
} inference_config_t;

void print_usage() {
    printf("Usage: ./train_gpt2_wasi.wasm [options]\n");
    printf("Options:\n");
    printf("  --model <path>      Path to ONNX model file (required)\n");
    printf("  --max-tokens <n>    Maximum tokens to generate (default: 100)\n");
    printf("  --temperature <f>   Sampling temperature (default: 0.8)\n");
    printf("  --top-k <n>         Top-k sampling (default: 40)\n");
    printf("  --seed <n>          Random seed (default: time-based)\n");
    printf("  --help              Show this help\n");
}

// Simple tokenizer (placeholder - would use proper tokenizer in practice)
int simple_tokenize(const char* text, int* tokens, int max_tokens) {
    // This is a very simplified tokenizer for demonstration
    // In practice, you'd use the GPT-2 BPE tokenizer
    int len = strlen(text);
    int token_count = 0;
    
    for (int i = 0; i < len && token_count < max_tokens; i++) {
        // Convert each character to a token ID (simplified)
        tokens[token_count++] = (int)text[i];
    }
    
    return token_count;
}

// Sample from logits using temperature and top-k
int sample_token(float* logits, int vocab_size, float temperature, int top_k, unsigned int* seed) {
    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    
    // Find top-k indices (simplified - just takes first top_k for demo)
    int best_idx = 0;
    float best_prob = logits[0];
    
    for (int i = 1; i < vocab_size && i < top_k; i++) {
        if (logits[i] > best_prob) {
            best_prob = logits[i];
            best_idx = i;
        }
    }
    
    return best_idx;
}

int main(int argc, char* argv[]) {
#ifdef USE_WASI_NN
    // Parse command line arguments
    inference_config_t config = {
        .model_path = NULL,
        .max_tokens = 100,
        .temperature = 0.8f,
        .top_k = 40,
        .seed = (int)time(NULL)
    };
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            config.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            config.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            config.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        }
    }
    
    if (!config.model_path) {
        fprintf(stderr, "Error: Model path is required\n");
        print_usage();
        return 1;
    }
    
    printf("WASI-NN GPT-2 Inference\n");
    printf("Model: %s\n", config.model_path);
    printf("Max tokens: %d\n", config.max_tokens);
    printf("Temperature: %.2f\n", config.temperature);
    printf("Top-k: %d\n", config.top_k);
    printf("Seed: %d\n", config.seed);
    
    // Initialize random seed
    unsigned int seed = (unsigned int)config.seed;
    
    // Load the GPT-2 model
    printf("Loading GPT-2 model...\n");
    gpt2_wasi_model_t* model = gpt2_wasi_load(config.model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", config.model_path);
        return 1;
    }
    
    printf("Model loaded successfully!\n");
    printf("Model configuration:\n");
    printf("  Layers: %d\n", model->num_layers);
    printf("  Heads: %d\n", model->num_heads);
    printf("  Channels: %d\n", model->channels);
    printf("  Max sequence length: %d\n", model->max_seq_len);
    printf("  Vocabulary size: %d\n", model->vocab_size);
    
    // Interactive inference loop
    char input_text[1024];
    int tokens[MAX_SEQ_LEN];
    float* logits = (float*)malloc(model->vocab_size * sizeof(float));
    
    while (1) {
        printf("\nEnter prompt (or 'quit' to exit): ");
        fflush(stdout);
        
        if (!fgets(input_text, sizeof(input_text), stdin)) {
            break;
        }
        
        // Remove newline
        input_text[strcspn(input_text, "\n")] = 0;
        
        if (strcmp(input_text, "quit") == 0) {
            break;
        }
        
        if (strlen(input_text) == 0) {
            continue;
        }
        
        // Tokenize input
        int num_tokens = simple_tokenize(input_text, tokens, MAX_SEQ_LEN);
        printf("Tokenized input (%d tokens): ", num_tokens);
        for (int i = 0; i < num_tokens; i++) {
            printf("%d ", tokens[i]);
        }
        printf("\n");
        
        printf("Generating...\n");
        
        // Generate tokens
        for (int step = 0; step < config.max_tokens && num_tokens < MAX_SEQ_LEN - 1; step++) {
            // Run inference
            gpt2_wasi_forward(model, tokens, num_tokens, logits);
            
            // Sample next token
            int next_token = sample_token(logits, model->vocab_size, 
                                        config.temperature, config.top_k, &seed);
            
            // Add to sequence
            tokens[num_tokens++] = next_token;
            
            // Print token (simplified - would decode properly in practice)
            if (next_token >= 32 && next_token < 127) {
                printf("%c", (char)next_token);
            } else {
                printf("[%d]", next_token);
            }
            fflush(stdout);
            
            // Stop on end-of-sequence token (simplified check)
            if (next_token == 0 || next_token == 50256) {
                break;
            }
        }
        
        printf("\n");
    }
    
    // Print inference statistics
    wasi_nn_print_stats(model->model);
    
    // Cleanup
    free(logits);
    gpt2_wasi_free(model);
    
    printf("WASI-NN inference completed successfully!\n");
    return 0;
    
#else
    printf("WASI-NN support not compiled in. Please build with USE_WASI_NN=1\n");
    return 1;
#endif
}