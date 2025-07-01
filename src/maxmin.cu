#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include "utils.cuh"
#include "types.cuh"

__global__ void maxmin_kernel(float *A, float *B, float *C, int M, int K, int N)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float max_val = -FLT_MAX;
        for (int k = 0; k < K; ++k)
        {
            float a = A[batch * M * K + row * K + k];
            float b = B[batch * K * N + k * N + col];
            float min_val = a < b ? a : b;
            if (min_val > max_val)
                max_val = min_val;
        }
        C[batch * M * N + row * N + col] = max_val;
    }
}

float run_maxmin(int batch, int M, int K, int N, int iterations)
{
    size_t size_A = batch * M * K * sizeof(float);
    size_t size_B = batch * K * N * sizeof(float);
    size_t size_C = batch * M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Inicializar con datos aleatorios
    for (int i = 0; i < batch * M * K; ++i)
        h_A[i] = (float)(rand() % 100);
    for (int i = 0; i < batch * K * N; ++i)
        h_B[i] = (float)(rand() % 100);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16, batch);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i)
    {
        maxmin_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    printf("Tiempo promedio por iteración (CUDA): %.6f ms\n", milliseconds / iterations);

    Tensor t = wrap_device_data_as_tensor(d_C, batch, M, N);
    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return milliseconds/iterations;
}
