/*
WASI-NN Backend Implementation for llm.c
*/

#include "wasi_nn_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_WASI_NN

// Helper function to convert model format to WASI-NN encoding
static graph_encoding format_to_encoding(model_format_t format) {
    switch (format) {
        case MODEL_FORMAT_ONNX:
            return onnx;
        case MODEL_FORMAT_TENSORFLOW:
            return tensorflow;
        case MODEL_FORMAT_PYTORCH:
            return pytorch;
        case MODEL_FORMAT_OPENVINO:
            return openvino;
        default:
            return onnx;
    }
}

// Helper function to convert execution target
static execution_target convert_target(execution_target_t target) {
    switch (target) {
        case EXEC_TARGET_CPU:
            return cpu;
        case EXEC_TARGET_GPU:
            return gpu;
        case EXEC_TARGET_TPU:
            return tpu;
        default:
            return cpu;
    }
}

// Load a model using WASI-NN
wasi_nn_model_t* wasi_nn_load_model(const uint8_t* model_data, size_t model_size,
                                     model_format_t format, execution_target_t target) {
    wasi_nn_model_t* model = (wasi_nn_model_t*)calloc(1, sizeof(wasi_nn_model_t));
    if (!model) {
        fprintf(stderr, "Failed to allocate WASI-NN model\n");
        return NULL;
    }
    
    model->format = format;
    model->target = target;
    
    // Load the model graph
    graph_builder builder = {
        .buf = (uint8_t*)model_data,
        .size = model_size
    };
    graph_builder_array builder_array = {
        .buf = &builder,
        .size = 1
    };
    graph_encoding encoding = format_to_encoding(format);
    execution_target exec_target = convert_target(target);
    
    int status = load(&builder_array, encoding, exec_target, &model->graph);
    if (status != 0) {
        fprintf(stderr, "Failed to load WASI-NN model: %s\n", 
                wasi_nn_error_message(status));
        free(model);
        return NULL;
    }
    
    // Initialize execution context
    status = init_execution_context(model->graph, &model->context);
    if (status != 0) {
        fprintf(stderr, "Failed to initialize WASI-NN context: %s\n", 
                wasi_nn_error_message(status));
        free(model);
        return NULL;
    }
    
    printf("WASI-NN model loaded successfully\n");
    return model;
}

// Free model resources
void wasi_nn_free_model(wasi_nn_model_t* model) {
    if (!model) return;
    
    // Free input tensors
    for (size_t i = 0; i < model->num_inputs; i++) {
        wasi_nn_free_tensor(&model->input_tensors[i]);
    }
    free(model->input_tensors);
    
    // Free output tensors
    for (size_t i = 0; i < model->num_outputs; i++) {
        wasi_nn_free_tensor(&model->output_tensors[i]);
    }
    free(model->output_tensors);
    
    free(model);
}

// Create a tensor
wasi_nn_tensor_t* wasi_nn_create_tensor(const char* name, float* data, 
                                         int32_t* dims, size_t num_dims) {
    wasi_nn_tensor_t* tensor = (wasi_nn_tensor_t*)calloc(1, sizeof(wasi_nn_tensor_t));
    if (!tensor) return NULL;
    
    tensor->name = strdup(name);
    tensor->num_dims = num_dims;
    
    // Copy dimensions
    tensor->dims = (int32_t*)malloc(num_dims * sizeof(int32_t));
    memcpy(tensor->dims, dims, num_dims * sizeof(int32_t));
    
    // Calculate total elements
    tensor->num_elements = 1;
    for (size_t i = 0; i < num_dims; i++) {
        tensor->num_elements *= dims[i];
    }
    
    // Allocate or reference data
    if (data) {
        tensor->data = data;
        tensor->owns_data = false;
    } else {
        tensor->data = (float*)calloc(tensor->num_elements, sizeof(float));
        tensor->owns_data = true;
    }
    
    return tensor;
}

// Free tensor resources
void wasi_nn_free_tensor(wasi_nn_tensor_t* tensor) {
    if (!tensor) return;
    
    free(tensor->name);
    free(tensor->dims);
    if (tensor->owns_data) {
        free(tensor->data);
    }
}

// Get tensor size in bytes
size_t wasi_nn_tensor_size(const wasi_nn_tensor_t* tensor) {
    return tensor->num_elements * sizeof(float);
}

// Set input tensor
int wasi_nn_set_input(wasi_nn_model_t* model, const char* name, 
                      const float* data, size_t size) {
    // Create tensor structure
    tensor input_tensor = {
        .dimensions = NULL,  // Will be inferred
        .type = fp32,
        .data = (uint8_t*)data
    };
    
    // Find input index by name (simplified - assumes index 0 for now)
    int input_index = 0;
    
    int status = set_input(model->context, input_index, &input_tensor);
    if (status != 0) {
        fprintf(stderr, "Failed to set input: %s\n", wasi_nn_error_message(status));
        return status;
    }
    
    return 0;
}

// Run inference
int wasi_nn_run_inference(wasi_nn_model_t* model) {
    // Simple timing for WASI (without deprecated clock)
    // In a real implementation, you'd use proper WASI timing APIs
    
    int status = compute(model->context);
    if (status != 0) {
        fprintf(stderr, "Failed to run inference: %s\n", wasi_nn_error_message(status));
        return status;
    }
    
    // Increment inference count (skip timing in WASI for now)
    model->inference_count++;
    
    return 0;
}

// Get output tensor
int wasi_nn_get_output(wasi_nn_model_t* model, const char* name, 
                       float* data, size_t size) {
    // Find output index by name (simplified - assumes index 0 for now)
    int output_index = 0;
    
    uint8_t* output_data = (uint8_t*)data;
    uint32_t output_size = (uint32_t)(size * sizeof(float));
    
    int status = get_output(model->context, output_index, output_data, 
                            &output_size);
    if (status != 0) {
        fprintf(stderr, "Failed to get output: %s\n", wasi_nn_error_message(status));
        return status;
    }
    
    return 0;
}

// Load GPT-2 model for WASI-NN
gpt2_wasi_model_t* gpt2_wasi_load(const char* model_path) {
    // Read model file
    FILE* file = fopen(model_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open model file: %s\n", model_path);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    size_t model_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    uint8_t* model_data = (uint8_t*)malloc(model_size);
    fread(model_data, 1, model_size, file);
    fclose(file);
    
    // Create GPT-2 model structure
    gpt2_wasi_model_t* gpt2 = (gpt2_wasi_model_t*)calloc(1, sizeof(gpt2_wasi_model_t));
    if (!gpt2) {
        free(model_data);
        return NULL;
    }
    
    // Load the model
    gpt2->model = wasi_nn_load_model(model_data, model_size, 
                                      MODEL_FORMAT_ONNX, EXEC_TARGET_CPU);
    free(model_data);
    
    if (!gpt2->model) {
        free(gpt2);
        return NULL;
    }
    
    // Initialize GPT-2 specific parameters (would be read from model metadata)
    gpt2->num_layers = 12;
    gpt2->num_heads = 12;
    gpt2->channels = 768;
    gpt2->max_seq_len = 1024;
    gpt2->vocab_size = 50257;
    
    // Allocate KV cache
    size_t cache_size = gpt2->num_layers * 2 * gpt2->max_seq_len * gpt2->channels;
    gpt2->key_cache = (float*)calloc(cache_size, sizeof(float));
    gpt2->value_cache = (float*)calloc(cache_size, sizeof(float));
    
    return gpt2;
}

// Free GPT-2 model
void gpt2_wasi_free(gpt2_wasi_model_t* model) {
    if (!model) return;
    
    wasi_nn_free_model(model->model);
    free(model->key_cache);
    free(model->value_cache);
    free(model->position_embeddings);
    free(model->token_embeddings);
    free(model);
}

// Run GPT-2 forward pass
void gpt2_wasi_forward(gpt2_wasi_model_t* model, int* token_ids, int seq_len, 
                       float* logits) {
    // Prepare input tensor
    float* input_embeddings = (float*)malloc(seq_len * model->channels * sizeof(float));
    
    // Convert token IDs to embeddings (simplified)
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];
        // In real implementation, would look up from embedding table
        for (int j = 0; j < model->channels; j++) {
            input_embeddings[i * model->channels + j] = (float)token_id / model->vocab_size;
        }
    }
    
    // Set input
    wasi_nn_set_input(model->model, "input_ids", input_embeddings, 
                      seq_len * model->channels);
    
    // Run inference
    wasi_nn_run_inference(model->model);
    
    // Get output logits
    wasi_nn_get_output(model->model, "logits", logits, 
                       seq_len * model->vocab_size);
    
    free(input_embeddings);
}

// Print statistics
void wasi_nn_print_stats(wasi_nn_model_t* model) {
    if (model->inference_count == 0) return;
    
    printf("WASI-NN Inference Statistics:\n");
    printf("  Total inferences: %zu\n", model->inference_count);
    printf("  Total time: %.3f seconds\n", model->total_inference_time);
    printf("  Average latency: %.3f ms\n", 
           (model->total_inference_time / model->inference_count) * 1000);
}

// Error message helper
const char* wasi_nn_error_message(int error_code) {
    switch (error_code) {
        case 0: return "Success";
        case 1: return "Invalid argument";
        case 2: return "Invalid encoding";
        case 3: return "Missing memory";
        case 4: return "Busy";
        case 5: return "Runtime error";
        default: return "Unknown error";
    }
}

#endif // USE_WASI_NN