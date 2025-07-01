#ifndef TYPES_CUH
#define TYPES_CUH

#include <cutensor.h>
#include <cudnn.h>

typedef struct {
    float* d_data;
    cutensorTensorDescriptor_t cutensor_desc;
    cudnnTensorDescriptor_t cudnn_desc;
    int batch, M, N;
} Tensor;

#endif