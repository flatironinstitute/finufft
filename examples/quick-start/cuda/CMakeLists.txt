cmake_minimum_required(VERSION 3.24)
# native builds for the GPU of this machine and needs CMP0104 NEW (3.24).
# A user -DCMAKE_CUDA_ARCHITECTURES or CUDAARCHS wins.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cufinufft_consume LANGUAGES CXX CUDA)

# docs-start: find_package_cuda
find_package(finufft REQUIRED)
# A shared install of cufinufft exports no CUDA link libraries, and main.cu
# uses Thrust's device_vector, so the consumer links cudart. nvcc finds the
# toolkit's bundled Thrust headers itself.
find_package(CUDAToolkit REQUIRED)

add_executable(app main.cu)
target_link_libraries(app PRIVATE finufft::cufinufft CUDA::cudart)
# docs-end: find_package_cuda
