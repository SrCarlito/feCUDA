cmake_minimum_required(VERSION 3.18)
project(TuProyecto CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

# Incluir directorios de headers
include_directories(include)
include_directories(/usr/include)

# Directorios donde buscar bibliotecas compartidas
link_directories(/usr/lib/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/libcutensor/12)
link_directories(/home/carlos/forgeffects/venv/bin/python/lib)

# Incluir matplotlib-cpp y NumPy/Python
include_directories(
    /home/carlos/libs/matplotlib-cpp
    /home/carlos/forgeffects/.venv/lib/python3.12/site-packages/numpy/_core/include
    /usr/include/python3.12
)

# Archivos fuente del proyecto
file(GLOB SOURCES "src/*.cu" "src/*.cpp")

# Agregar ejecutable
add_executable(tu_ejecutable ${SOURCES})

# Enlazar el ejecutable con cuDNN, cuTENSOR y Python
target_link_libraries(tu_ejecutable
    cudnn
    cutensor
    python3.12
)

# Activar compilación separable CUDA
set_target_properties(tu_ejecutable PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
