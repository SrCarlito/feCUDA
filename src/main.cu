#include <iostream>
#include <vector>
#include <cstring>
#include "headers.cuh"
#include "utils.cuh"
#include "types.cuh"

int main()
{
    // === Probando función de lectura de archivos ===
    printf("\n=== Probando función leer_matriz_3d_desde_archivo ===\n");

    TensorResult tensor_desde_archivo;
    bool exito = leer_matriz_3d_desde_archivo("../datasets_txt/reflexive.txt", tensor_desde_archivo, 1, 6, 6, 1);

    if (exito)
    {
        printf("\nTensor cargado desde archivo:\n");
        imprimir_tensor(tensor_desde_archivo, 10, 10, "Tensor desde archivo", true);
    }
    else
    {
        printf("Error: No se pudo cargar el tensor desde archivo\n");
    }

    // === Probando función iterative_maxmin_cuadrado ===
    printf("\n=== Probando función iterative_maxmin_cuadrado ===\n");

    // Usar el tensor cargado desde archivo si está disponible, sino usar datos hardcodeados
    TensorResult test_tensor;
    bool usar_archivo = exito;

    if (usar_archivo)
    {
        test_tensor = tensor_desde_archivo;
        printf("Usando tensor cargado desde archivo\n");
    }
    else
    {
        printf("Error: Tensor no encontrado \n");
        return 0;
    }

    std::vector<TensorResult> result_paths_matrix;
    std::vector<TensorResult> result_values_list;

    float test_threshold = 0.3f;
    int test_order = 3;

    printf("Tensor de prueba (2x2x2x1):\n");
    imprimir_tensor(test_tensor, 10, 10, "Test Tensor", true);

    printf("\nLlamando iterative_maxmin_cuadrado con threshold=%.2f, order=%d...\n",
           test_threshold, test_order);

    iterative_maxmin_cuadrado(test_tensor, test_threshold, test_order, result_paths_matrix, result_values_list);

    printf("\nResultados de iterative_maxmin_cuadrado:\n");
    printf("Número de paths encontrados: %zu\n", result_paths_matrix.size());
    printf("Número de valores encontrados: %zu\n", result_values_list.size());

    
    printf("Paths:\n");
    for (const auto &path : result_paths_matrix)
    {
        imprimir_tensor(path, 10, 10, "Path", true);
    }

    printf("Valores:\n");
    for (const auto &value : result_values_list)
    {
        imprimir_tensor(value, 10, 1, "Value", true);
    }

    // Liberar memoria del tensor de prueba
    if (usar_archivo)
    {
        if (tensor_desde_archivo.data)
            free(tensor_desde_archivo.data);
    }
    else
    {
        if (test_tensor.data)
            free(test_tensor.data);
    }

    printf("\nPruebas completadas.\n");
    return 0;
}
