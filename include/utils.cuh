#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <types.cuh>

// Macro para verificar errores de CUDA
#define CHECK_CUDA(call) {                                         \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}

// Función común para imprimir resultados
void imprimir_resultado(float* h_C, int batch, int M, int N, int max_rows = 8, int max_cols = 8);

Tensor wrap_device_data_as_tensor(float *device_ptr, int batch, int M, int N);

#endif
