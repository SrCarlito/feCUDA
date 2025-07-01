#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "utils.cuh"
#include "types.cuh"


__global__ void maxmin_kernel_reduction(float* A, float* B, float* C, int M, int K, int N) {
    extern __shared__ float sdata[];

    int batch = blockIdx.z;
    int row   = blockIdx.y;
    int col   = blockIdx.x;
    int tid   = threadIdx.x;

    float local_max = -FLT_MAX;

    // Cada hilo recorre múltiples k si es necesario
    for (int k = tid; k < K; k += blockDim.x) {
        float a = A[batch * M * K + row * K + k];
        float b = B[batch * K * N + k * N + col];
        float min_val = fminf(a, b);
        local_max = fmaxf(local_max, min_val);
    }

    // Guardar máximos parciales en memoria compartida
    sdata[tid] = local_max;
    __syncthreads();

    // Reducción paralela (máximo entre los hilos)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Hilo 0 escribe el resultado final
    if (tid == 0) {
        C[batch * M * N + row * N + col] = sdata[0];
    }
}


float run_maxmin_kernel(int batch, int M, int K, int N, int iterations) {
    size_t size_A = batch * M * K * sizeof(float);
    size_t size_B = batch * K * N * sizeof(float);
    size_t size_C = batch * M * N * sizeof(float);
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    if (!h_A || !h_B) {
        fprintf(stderr, "Error al asignar memoria para h_A o h_B\n");
        exit(EXIT_FAILURE);
    }

    // Semilla para que los resultados varíen cada vez
    srand((unsigned int)time(NULL));

    // Llenar h_A y h_B con valores aleatorios entre 0.0 y 1.0
    for (int i = 0; i < batch * M * K; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < batch * K * N; ++i) {
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    int threads = (K < 32) ? K : 32;
    ;
    dim3 blockDim(threads);
    dim3 gridDim(N, M, batch);

    size_t shared_mem_size = threads * sizeof(float);

    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        maxmin_kernel_reduction<<<gridDim, blockDim, shared_mem_size>>>(d_A, d_B, d_C, M, K, N);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    printf("Tiempo promedio por iteración (con reduccion) (CUDA): %.6f ms\n", milliseconds / iterations);
    
    Tensor t = wrap_device_data_as_tensor(h_C, batch, M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return milliseconds / iterations;
}
