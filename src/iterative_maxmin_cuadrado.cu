#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <vector>
#include "utils.cuh"
#include "types.cuh"
#include "headers.cuh"

// Kernel para calcular prima = maxmin_conjugado - gen_tensor
__global__ void calculate_prima_kernel(float *maxmin_conjugado, float *gen_tensor,
                                       float *prima, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        prima[idx] = maxmin_conjugado[idx] - gen_tensor[idx];
    }
}

// Función para calcular prima (efectos de n generación)
void calculate_prima(const TensorResult &maxmin_conjugado, const TensorResult &gen_tensor,
                     TensorResult &prima)
{
    // Verificar que las dimensiones coincidan
    if (maxmin_conjugado.batch != gen_tensor.batch ||
        maxmin_conjugado.M != gen_tensor.M ||
        maxmin_conjugado.N != gen_tensor.N)
    {
        printf("Error: Dimensiones no coinciden para calcular prima\n");
        return;
    }

    int total_elements = maxmin_conjugado.batch * maxmin_conjugado.M * maxmin_conjugado.N;
    size_t size = total_elements * sizeof(float);

    // Alocar memoria para prima
    float *h_prima = (float *)malloc(size);
    float *d_maxmin_conjugado, *d_gen_tensor, *d_prima;

    // Alocar memoria device
    CHECK_CUDA(cudaMalloc(&d_prima, size));

    // Copiar datos a device
    if (maxmin_conjugado.is_device_ptr)
    {
        d_maxmin_conjugado = maxmin_conjugado.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_maxmin_conjugado, size));
        CHECK_CUDA(cudaMemcpy(d_maxmin_conjugado, maxmin_conjugado.data, size, cudaMemcpyHostToDevice));
    }

    if (gen_tensor.is_device_ptr)
    {
        d_gen_tensor = gen_tensor.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_gen_tensor, size));
        CHECK_CUDA(cudaMemcpy(d_gen_tensor, gen_tensor.data, size, cudaMemcpyHostToDevice));
    }

    // Lanzar kernel
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    calculate_prima_kernel<<<grid_size, block_size>>>(d_maxmin_conjugado, d_gen_tensor,
                                                      d_prima, total_elements);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copiar resultado a host
    CHECK_CUDA(cudaMemcpy(h_prima, d_prima, size, cudaMemcpyDeviceToHost));

    // Configurar TensorResult de salida
    prima.data = h_prima;
    prima.is_device_ptr = false;
    prima.batch = maxmin_conjugado.batch;
    prima.M = maxmin_conjugado.M;
    prima.N = maxmin_conjugado.N;
    prima.K = maxmin_conjugado.K;

    // Limpiar memoria device
    cudaFree(d_prima);
    if (!maxmin_conjugado.is_device_ptr)
        cudaFree(d_maxmin_conjugado);
    if (!gen_tensor.is_device_ptr)
        cudaFree(d_gen_tensor);
}

// Función principal iterative_maxmin_cuadrado
void iterative_maxmin_cuadrado(const TensorResult &tensor, float thr, int order,
                               std::vector<TensorResult> &result_tensor_paths,
                               std::vector<TensorResult> &result_values_paths)
{
    // Validaciones
    if (thr < 0.0f || thr > 1.0f)
    {
        printf("Error: El threshold debe estar en el rango [0,1]\n");
        return;
    }

    if (order <= 1)
    {
        printf("Error: El order debe ser mayor que 1\n");
        return;
    }

    if (tensor.data == nullptr)
    {
        printf("Error: Tensor de entrada es nulo\n");
        return;
    }

    // Copiar tensor original
    TensorResult original_tensor;
    size_t tensor_size = tensor.batch * tensor.M * tensor.N * tensor.K * sizeof(float);
    original_tensor.data = (float *)malloc(tensor_size);
    memcpy(original_tensor.data, tensor.data, tensor_size);
    original_tensor.is_device_ptr = false;
    original_tensor.owns_memory = true;
    original_tensor.batch = tensor.batch;
    original_tensor.M = tensor.M;
    original_tensor.N = tensor.N;
    original_tensor.K = tensor.K;

    // Inicializar gen_tensor como copia del tensor original
    TensorResult gen_tensor;
    gen_tensor.data = (float *)malloc(tensor_size);
    memcpy(gen_tensor.data, tensor.data, tensor_size);
    gen_tensor.is_device_ptr = false;
    gen_tensor.owns_memory = true;
    gen_tensor.batch = tensor.batch;
    gen_tensor.M = tensor.M;
    gen_tensor.N = tensor.N;
    gen_tensor.K = tensor.K;

    // Listas temporales para almacenar resultados
    std::vector<TensorResult> result_tensors_list;
    std::vector<TensorResult> result_values_list;

    // Limpiar vectores de salida
    result_tensor_paths.clear();
    result_values_paths.clear();

    for (int i = 0; i < order - 1; i++)
    {
        // Calcular min_result y maxmin_conjugado
        TensorResult min_result, maxmin_conjugado;
        maxmin(gen_tensor, original_tensor, maxmin_conjugado, min_result, false);

        // Calcular prima = maxmin_conjugado - gen_tensor
        TensorResult prima;
        calculate_prima(maxmin_conjugado, gen_tensor, prima);

        // Imprimir prima


        // Calcular indices con prima y threshold
        TensorResult result_tensor, result_values;
        indices(min_result, prima, result_tensor, result_values, thr);

        // Imprimir resultados intermedios

        result_tensors_list.push_back(result_tensor);
        result_values_list.push_back(result_values);

        // Verificar si se encontraron efectos
        if (result_values.data == nullptr || result_values.batch == 0)
        {
            if (i == 0)
            {
                printf("Error: No se encontraron efectos con threshold %.4f\n", thr);
                // Limpiar memoria y retornar
                safe_tensor_cleanup(original_tensor);
                safe_tensor_cleanup(gen_tensor);
                safe_tensor_cleanup(min_result);
                safe_tensor_cleanup(maxmin_conjugado);
                safe_tensor_cleanup(prima);
                return;
            }
            else
            {
                printf("Los efectos solo fueron encontrados hasta el orden %d\n", i + 1);
                break;
            }
        }

        printf("Orden %d: %d efectos encontrados\n", i + 1, result_values.batch);

        // Para el primer orden (i == 0), agregamos directamente los paths encontrados
        if (i >= 1)
        {
            TensorResult previous_paths;

            if (i == 1)
            {
                previous_paths = result_tensor_paths[result_tensor_paths.size() - 1];
            }
            else
            {
                previous_paths = result_tensors_list[0];
            }
            // Para órdenes superiores (i >= 1), construimos caminos usando armar_caminos

            TensorResult paths, values;
            armar_caminos(previous_paths, result_tensor, result_values, paths, values, i);

            result_tensor_paths.push_back(paths);
            result_values_paths.push_back(values);

            // Limpiar memoria temporal de caminos construidos
            safe_tensor_cleanup(paths);
            safe_tensor_cleanup(values);
            safe_tensor_cleanup(previous_paths);
        }

        gen_tensor.data = maxmin_conjugado.data; // Actualizar gen_tensor para la siguiente iteración

        gen_tensor.is_device_ptr = maxmin_conjugado.is_device_ptr;
        gen_tensor.owns_memory = false;
        gen_tensor.batch = maxmin_conjugado.batch;
        gen_tensor.M = maxmin_conjugado.M;
        gen_tensor.N = maxmin_conjugado.N;
        gen_tensor.K = maxmin_conjugado.K;

        // Limpiar memoria temporal de esta iteración usando el nuevo sistema
        safe_tensor_cleanup(min_result);
        safe_tensor_cleanup(prima);
        safe_tensor_cleanup(result_tensor);
        safe_tensor_cleanup(result_values);
    }

    // Agregar el primer resultado a las listas de caminos y valores

    result_tensor_paths.insert(result_tensor_paths.begin(), result_tensors_list[0]);
    result_values_paths.insert(result_values_paths.begin(), result_values_list[0]);

    // Limpiar memoria
    safe_tensor_cleanup(original_tensor);
    safe_tensor_cleanup(gen_tensor);

    // Limpiar y verificar dispositivo CUDA
    cuda_cleanup_and_check();

    printf("\n=== ITERATIVE_MAXMIN_CUADRADO COMPLETADO ===\n");
    printf("Total de paths encontrados: %zu\n", result_tensor_paths.size());
    printf("Total de valores encontrados: %zu\n", result_values_paths.size());
}
