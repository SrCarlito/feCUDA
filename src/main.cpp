#include <iostream>
#include <vector>
#include "headers.cuh"
#include "utils.cuh"
#include "types.cuh"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    std::vector<int> batch_sizes = {1, 8, 16, 32, 64, 128};
    int M = 20, K = 20, N = 20;
    int repeticiones = 1000;

    std::vector<float> tiempos_clasico;
    std::vector<float> tiempos_kernel;

    for (int batch : batch_sizes) {
        float tiempo_clasico = run_maxmin(batch, M, K, N, repeticiones);
        float tiempo_kernel = run_maxmin_kernel(batch, M, K, N, repeticiones);

        tiempos_clasico.push_back(tiempo_clasico);
        tiempos_kernel.push_back(tiempo_kernel);

        std::cout << "Batch: " << batch
                  << " | Clásico: " << tiempo_clasico << " ms"
                  << " | Kernel: " << tiempo_kernel << " ms" << std::endl;
    }

    // Graficar
    plt::figure_size(800, 600);
    plt::named_plot("Iterativo", batch_sizes, tiempos_clasico);
    plt::named_plot("reducido", batch_sizes, tiempos_kernel);


    plt::xlabel("Batch size");
    plt::ylabel("Tiempo promedio (ms)");
    plt::title("Comparación de rendimiento MaxMin");
    plt::legend();
    plt::grid(true);
    plt::save("rendimiento_batch.png");  // También puedes usar plt::show();
}
