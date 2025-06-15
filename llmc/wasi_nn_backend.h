/*
WASI-NN Backend for llm.c
Provides neural network inference using WebAssembly System Interface
*/

#ifndef WASI_NN_BACKEND_H
#define WASI_NN_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef USE_WASI_NN
#include <wasi_nn.h>
#endif

// Model formats supported by WASI-NN
typedef enum {
    MODEL_FORMAT_ONNX,
    MODEL_FORMAT_TENSORFLOW,
    MODEL_FORMAT_PYTORCH,
    MODEL_FORMAT_OPENVINO
} model_format_t;

// Execution targets
typedef enum {
    EXEC_TARGET_CPU,
    EXEC_TARGET_GPU,
    EXEC_TARGET_TPU
} execution_target_t;

// Forward declarations
typedef struct wasi_nn_model wasi_nn_model_t;
typedef struct wasi_nn_tensor wasi_nn_tensor_t;

// Tensor structure
struct wasi_nn_tensor {
    char* name;
    float* data;
    int32_t* dims;
    size_t num_dims;
    size_t num_elements;
    bool owns_data;  // Whether this tensor owns its data buffer
};

// Model context
struct wasi_nn_model {
#ifdef USE_WASI_NN
    graph graph;
    graph_execution_context context;
#endif
    model_format_t format;
    execution_target_t target;
    
    // Model metadata
    size_t num_parameters;
    size_t num_layers;
    
    // Input/output tensors
    wasi_nn_tensor_t* input_tensors;
    size_t num_inputs;
    wasi_nn_tensor_t* output_tensors;
    size_t num_outputs;
    
    // Performance tracking
    double total_inference_time;
    size_t inference_count;
};

// GPT-2 specific structure for WASI-NN
typedef struct {
    wasi_nn_model_t* model;
    
    // Model configuration
    int num_layers;
    int num_heads;
    int channels;
    int max_seq_len;
    int vocab_size;
    
    // Cached tensors for efficiency
    float* position_embeddings;
    float* token_embeddings;
    
    // Inference state
    float* key_cache;
    float* value_cache;
    int cache_seq_len;
} gpt2_wasi_model_t;

// Model loading and management
#ifdef USE_WASI_NN

wasi_nn_model_t* wasi_nn_load_model(const uint8_t* model_data, size_t model_size,
                                     model_format_t format, execution_target_t target);
void wasi_nn_free_model(wasi_nn_model_t* model);

// Tensor operations
wasi_nn_tensor_t* wasi_nn_create_tensor(const char* name, float* data, 
                                         int32_t* dims, size_t num_dims);
void wasi_nn_free_tensor(wasi_nn_tensor_t* tensor);
size_t wasi_nn_tensor_size(const wasi_nn_tensor_t* tensor);

// Inference operations
int wasi_nn_set_input(wasi_nn_model_t* model, const char* name, 
                      const float* data, size_t size);
int wasi_nn_run_inference(wasi_nn_model_t* model);
int wasi_nn_get_output(wasi_nn_model_t* model, const char* name, 
                       float* data, size_t size);

// GPT-2 specific functions
gpt2_wasi_model_t* gpt2_wasi_load(const char* model_path);
void gpt2_wasi_free(gpt2_wasi_model_t* model);
void gpt2_wasi_forward(gpt2_wasi_model_t* model, int* token_ids, int seq_len, 
                       float* logits);

// Utility functions
void wasi_nn_print_stats(wasi_nn_model_t* model);
const char* wasi_nn_error_message(int error_code);

#else // !USE_WASI_NN

// Stub implementations when WASI-NN is not available
static inline wasi_nn_model_t* wasi_nn_load_model(const uint8_t* model_data, 
                                                   size_t model_size,
                                                   model_format_t format, 
                                                   execution_target_t target) {
    fprintf(stderr, "WASI-NN support not compiled in\n");
    return NULL;
}

static inline void wasi_nn_free_model(wasi_nn_model_t* model) {}

static inline gpt2_wasi_model_t* gpt2_wasi_load(const char* model_path) {
    fprintf(stderr, "WASI-NN support not compiled in\n");
    return NULL;
}

static inline void gpt2_wasi_free(gpt2_wasi_model_t* model) {}

static inline void gpt2_wasi_forward(gpt2_wasi_model_t* model, int* token_ids, 
                                     int seq_len, float* logits) {
    fprintf(stderr, "WASI-NN support not compiled in\n");
}

#endif // USE_WASI_NN

#endif // WASI_NN_BACKEND_H