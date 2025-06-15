/*
Simple WASI Test for llm.c
Demonstrates basic WASI compilation without WASI-NN dependencies
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple demonstration of WASI compilation
int main(int argc, char* argv[]) {
    printf("WASI-enabled llm.c test program\n");
    printf("Arguments: %d\n", argc);
    
    for (int i = 0; i < argc; i++) {
        printf("  argv[%d] = %s\n", i, argv[i]);
    }
    
    // Simple computation to test basic functionality
    float test_array[10];
    for (int i = 0; i < 10; i++) {
        test_array[i] = (float)i * 1.5f;
    }
    
    printf("Test computation results:\n");
    for (int i = 0; i < 10; i++) {
        printf("  test_array[%d] = %.2f\n", i, test_array[i]);
    }
    
    printf("WASI test completed successfully!\n");
    return 0;
}