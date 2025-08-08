#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include "utils.cuh"
#include "types.cuh"

// Kernel para encontrar matches entre previous_paths y result_tensor
__global__ void find_path_matches_kernel(float *previous_paths, float *result_tensor,
                                         float *result_values, float *output_paths,
                                         float *output_values, int *match_count,
                                         int num_prev_paths, int num_current_tensor,
                                         int prev_cols, int current_cols, int order)
{
    int prev_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (prev_idx < num_prev_paths && curr_idx < num_current_tensor)
    {
        // Extraer las primeras 3 columnas de previous_paths [batch, M, K]
        int prev_batch = (int)previous_paths[prev_idx * prev_cols + 0];
        int prev_m = (int)previous_paths[prev_idx * prev_cols + 1];
        int prev_k = (int)previous_paths[prev_idx * prev_cols + 2 + order];

        // Extraer las primeras 3 columnas de result_tensor [batch, M, K]
        int curr_batch = (int)result_tensor[curr_idx * current_cols + 0];
        int curr_m = (int)result_tensor[curr_idx * current_cols + 1];
        int curr_k = (int)result_tensor[curr_idx * current_cols + 2];

        // Verificar si hay match en las primeras 3 columnas
        if (prev_batch == curr_batch && prev_m == curr_m && prev_k == curr_k)
        {
            // Found a match - usar atomic add para obtener posición de salida
            int output_idx = atomicAdd(match_count, 1);

            // Construir el nuevo camino combinando previous_path + columna 4 de result_tensor
            int output_base = output_idx * (prev_cols + 1);

            // Copiar todas las columnas de previous_paths
            for (int col = 0; col < prev_cols; col++)
            {
                output_paths[output_base + col] = previous_paths[prev_idx * prev_cols + col];
            }

            // Agregar la columna 4 (índice 3) de result_tensor al final
            if (current_cols > 3)
            {
                output_paths[output_base + prev_cols] = result_tensor[curr_idx * current_cols + 3];
            }
            else
            {
                output_paths[output_base + prev_cols] = 0.0f; // Valor por defecto si no existe columna 4
            }

            // Guardar el valor correspondiente
            output_values[output_idx] = result_values[curr_idx];
        }
    }
}

void armar_caminos(const TensorResult &previous_paths, const TensorResult &result_tensor,
                   const TensorResult &result_values, TensorResult &paths,
                   TensorResult &matched_values, int order)
{

    // Validaciones
    if (previous_paths.data == nullptr || result_tensor.data == nullptr || result_values.data == nullptr)
    {
        printf("Error: Punteros nulos en entrada\n");
        return;
    }

    // Extraer dimensiones
    int num_prev_paths = previous_paths.batch;
    int prev_cols = previous_paths.M;
    int num_current_tensor = result_tensor.batch;
    int current_cols = result_tensor.M;
    int num_values = result_values.batch;

    if (num_current_tensor != num_values)
    {
        printf("Error: Número de elementos en result_tensor (%d) no coincide con result_values (%d)\n",
               num_current_tensor, num_values);
        return;
    }

    // Calcular tamaños máximos de salida
    int max_output_size = num_prev_paths * num_current_tensor;
    size_t prev_size = num_prev_paths * prev_cols * sizeof(float);
    size_t curr_size = num_current_tensor * current_cols * sizeof(float);
    size_t values_size = num_values * sizeof(float);
    size_t output_paths_size = max_output_size * (prev_cols + 1) * sizeof(float);
    size_t output_values_size = max_output_size * sizeof(float);

    // Alocar memoria en device
    float *d_previous_paths, *d_result_tensor, *d_result_values;
    float *d_output_paths, *d_output_values;
    int *d_match_count;

    CHECK_CUDA(cudaMalloc(&d_output_paths, output_paths_size));
    CHECK_CUDA(cudaMalloc(&d_output_values, output_values_size));
    CHECK_CUDA(cudaMalloc(&d_match_count, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_match_count, 0, sizeof(int)));

    // Copiar datos a device o usar punteros existentes
    if (previous_paths.is_device_ptr)
    {
        d_previous_paths = previous_paths.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_previous_paths, prev_size));
        CHECK_CUDA(cudaMemcpy(d_previous_paths, previous_paths.data, prev_size, cudaMemcpyHostToDevice));
    }

    if (result_tensor.is_device_ptr)
    {
        d_result_tensor = result_tensor.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_result_tensor, curr_size));
        CHECK_CUDA(cudaMemcpy(d_result_tensor, result_tensor.data, curr_size, cudaMemcpyHostToDevice));
    }

    if (result_values.is_device_ptr)
    {
        d_result_values = result_values.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_result_values, values_size));
        CHECK_CUDA(cudaMemcpy(d_result_values, result_values.data, values_size, cudaMemcpyHostToDevice));
    }

    // Configurar kernel
    dim3 block_size(16, 16);
    dim3 grid_size((num_prev_paths + block_size.x - 1) / block_size.x,
                   (num_current_tensor + block_size.y - 1) / block_size.y);

    // Lanzar kernel
    find_path_matches_kernel<<<grid_size, block_size>>>(
        d_previous_paths, d_result_tensor, d_result_values,
        d_output_paths, d_output_values, d_match_count,
        num_prev_paths, num_current_tensor, prev_cols, current_cols, order);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Obtener número de matches
    int match_count;
    CHECK_CUDA(cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (match_count > 0)
    {
        // Alocar memoria host para resultados
        size_t final_paths_size = match_count * (prev_cols + 1) * sizeof(float);
        size_t final_values_size = match_count * sizeof(float);

        float *h_output_paths = (float *)malloc(final_paths_size);
        float *h_output_values = (float *)malloc(final_values_size);

        // Copiar resultados a host
        CHECK_CUDA(cudaMemcpy(h_output_paths, d_output_paths, final_paths_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_output_values, d_output_values, final_values_size, cudaMemcpyDeviceToHost));

        // Configurar TensorResult de salida
        paths.data = h_output_paths;
        paths.is_device_ptr = false;
        paths.batch = match_count;
        paths.M = prev_cols + 1;
        paths.N = 1;
        paths.K = 1;

        matched_values.data = h_output_values;
        matched_values.is_device_ptr = false;
        matched_values.batch = match_count;
        matched_values.M = 1;
        matched_values.N = 1;
        matched_values.K = 1;

    }
    else
    {
        printf("Error: No se encontraron matches\n");

        // Configurar TensorResult vacíos
        paths.data = nullptr;
        paths.is_device_ptr = false;
        paths.batch = 0;
        paths.M = 0;
        paths.N = 0;
        paths.K = 0;

        matched_values.data = nullptr;
        matched_values.is_device_ptr = false;
        matched_values.batch = 0;
        matched_values.M = 0;
        matched_values.N = 0;
        matched_values.K = 0;
    }

    // Limpiar memoria device
    if (d_output_paths)
        cudaFree(d_output_paths);
    if (d_output_values)
        cudaFree(d_output_values);
    if (d_match_count)
        cudaFree(d_match_count);

    // Limpiar copias temporales si se crearon
    if (!previous_paths.is_device_ptr && d_previous_paths)
        cudaFree(d_previous_paths);
    if (!result_tensor.is_device_ptr && d_result_tensor)
        cudaFree(d_result_tensor);
    if (!result_values.is_device_ptr && d_result_values)
        cudaFree(d_result_values);
}