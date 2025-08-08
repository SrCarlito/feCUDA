#ifndef TYPES_CUH
#define TYPES_CUH

#include <cutensor.h>
#include <cudnn.h>
#include <string.h>
#include <cuda_runtime.h>

// Estructura para mantener información completa del tensor
struct TensorResult
{
    float *data;        // Puntero a los datos
    bool is_device_ptr; // Indica si los datos están en device o host
    bool owns_memory;   // Indica si este TensorResult es dueño de la memoria
    int batch, M, N, K; // Dimensiones del tensor (K para dimensiones adicionales)

    // Constructor completo con ownership
    TensorResult(float *d, bool is_dev, int b, int m, int n, int k = 1, bool owns = true)
        : data(d), is_device_ptr(is_dev), owns_memory(owns), batch(b), M(m), N(n), K(k)
    {
    }

    // Constructor por defecto
    TensorResult() : data(nullptr), is_device_ptr(false), owns_memory(false), batch(0), M(0), N(0), K(0) {}

    // Constructor de copia (disable ownership por defecto)
    TensorResult(const TensorResult &other)
        : data(other.data), is_device_ptr(other.is_device_ptr), owns_memory(false),
          batch(other.batch), M(other.M), N(other.N), K(other.K)
    {
        // Por defecto, las copias NO poseen la memoria para evitar double free
    }

    // Operador de asignación
    TensorResult &operator=(const TensorResult &other)
    {
        if (this != &other)
        {
            // Liberar memoria propia si la tenemos
            cleanup();

            // Copiar datos (sin ownership por defecto)
            data = other.data;
            is_device_ptr = other.is_device_ptr;
            owns_memory = false; // Por seguridad, no transferir ownership
            batch = other.batch;
            M = other.M;
            N = other.N;
            K = other.K;
        }
        return *this;
    }

    // Destructor seguro
    ~TensorResult()
    {
        cleanup();
    }

    // Función para limpiar memoria
    void cleanup()
    {
        if (data && owns_memory)
        {
            if (is_device_ptr)
            {
                cudaFree(data);
            }
            else
            {
                free(data);
            }
        }
        data = nullptr;
        owns_memory = false;
    }

    // Función para transferir ownership
    void transfer_ownership(bool should_own)
    {
        owns_memory = should_own;
    }

    // Función para obtener el tamaño en bytes
    size_t size_bytes() const
    {
        return batch * M * N * K * sizeof(float);
    }

    // Función para clonar este tensor
    TensorResult clone() const
    {
        if (!data)
            return TensorResult();

        size_t size = batch * M * N * K * sizeof(float);
        float *new_data = nullptr;

        if (is_device_ptr)
        {
            cudaMalloc(&new_data, size);
            cudaMemcpy(new_data, data, size, cudaMemcpyDeviceToDevice);
            return TensorResult(new_data, true, batch, M, N, K, true);
        }
        else
        {
            new_data = (float *)malloc(size);
            memcpy(new_data, data, size);
            return TensorResult(new_data, false, batch, M, N, K, true);
        }
    }

    // Función para obtener el número total de elementos
    size_t total_elements() const
    {
        return batch * M * N * K;
    }
};

#endif