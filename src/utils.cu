#include "utils.cuh"
#include <cuda_runtime.h>
#include <cutensor.h>
#include <cudnn.h>
#include <stdio.h>
#include <types.cuh>

void imprimir_resultado(float* h_C, int batch, int M, int N, int max_rows, int max_cols) {
    int print_M = (M < max_rows) ? M : max_rows;
    int print_N = (N < max_cols) ? N : max_cols;

    for (int b = 0; b < batch; ++b) {
        printf("=== Batch %d ===\n", b);
        for (int i = 0; i < print_M; ++i) {
            for (int j = 0; j < print_N; ++j) {
                printf("%f ", h_C[b * M * N + i * N + j]);
            }
            printf("\n");
        }
        if (print_M < M || print_N < N)
            printf("... (truncado a %dx%d)\n", print_M, print_N);
        printf("\n");
    }
}



Tensor wrap_device_data_as_tensor(float* device_ptr, int batch, int M, int N) {
    Tensor tensor;
    tensor.d_data = device_ptr;
    tensor.batch = batch;
    tensor.M = M;
    tensor.N = N;

    // === cuTENSOR descriptor ===
    cutensorHandle_t cutensor_handle;
    cutensorCreate(&cutensor_handle);

    int64_t dims[3]    = {batch, M, N};
    int64_t strides[3] = {M * N, N, 1};

    cutensorCreateTensorDescriptor(
        cutensor_handle,
        &tensor.cutensor_desc,
        3, dims, strides,
        CUTENSOR_R_32F,
        CUTENSOR_OP_IDENTITY
    );

    // === cuDNN descriptor === 
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    cudnnCreateTensorDescriptor(&tensor.cudnn_desc);
    cudnnSetTensor4dDescriptor(
        tensor.cudnn_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch, M, N, 1  // NCHW format
    );

    return tensor;
}

