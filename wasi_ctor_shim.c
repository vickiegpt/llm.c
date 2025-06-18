/*
 * WASI constructor shim for WASM modules
 * 
 * This file provides an implementation of the __wasm_call_ctors function
 * that is required by some WASM modules compiled from C/C++ code.
 */

#include <stdio.h>

// Define the constructor function that WASM modules expect to be imported
// This is normally provided by the host environment in more complex runtimes
__attribute__((used))
void __wasm_call_ctors(void) {
    // This could be empty or contain initialization logic if needed
    printf("WASM constructor called\n");
}

// Export the function to make it available to the WASM module
__attribute__((export_name("__wasm_call_ctors")))
void export_wasm_call_ctors(void) {
    __wasm_call_ctors();
}

// Empty main function (not used, but ensures the module is valid)
int main() {
    return 0;
} 