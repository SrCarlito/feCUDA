#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "types.cuh"
// Funciones exportadas desde archivos .cu
float run_maxmin(int batch, int M, int K, int N, int iterations);
float run_maxmin_kernel(int batch, int M, int K, int N, int iterations);

#endif
