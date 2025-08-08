cmake_minimum_required(VERSION 3.18)
project(TuProyecto CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Incluir directorios de headers
include_directories(include)
include_directories(/usr/include)   

# Directorios donde buscar bibliotecas compartidas
link_directories(/usr/lib/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/libcutensor/12)

# Archivos fuente del proyecto
file(GLOB SOURCES "src/*.cu")

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
