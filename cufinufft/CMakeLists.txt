cmake_minimum_required(VERSION 3.18)

project(
  cufinufft
  LANGUAGES C CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60;61;70;75)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(PRECISION_INDEPENDENT_SRC src/precision_independent.cu src/profile.cu contrib/legendre_rule_fast.c contrib/utils.cpp)
set(PRECISION_DEPENDENT_SRC
  src/spreadinterp.cpp
  src/1d/spreadinterp1d.cu src/1d/cufinufft1d.cu
  src/1d/spread1d_wrapper.cu src/1d/interp1d_wrapper.cu
  src/2d/spreadinterp2d.cu src/2d/cufinufft2d.cu
  src/2d/spread2d_wrapper.cu src/2d/spread2d_wrapper_paul.cu
  src/2d/interp2d_wrapper.cu src/memtransfer_wrapper.cu
  src/deconvolve_wrapper.cu src/cufinufft.cu src/common.cu
  src/3d/spreadinterp3d.cu src/3d/spread3d_wrapper.cu
  src/3d/interp3d_wrapper.cu src/3d/cufinufft3d.cu
)

set(CUFINUFFT_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/cufinufft/contrib/cuda_samples
)

add_library(cufinufft_common_objects OBJECT ${PRECISION_INDEPENDENT_SRC})
target_include_directories(cufinufft_common_objects PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
set_property(TARGET cufinufft_common_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(cufinufft_32_objects OBJECT ${PRECISION_DEPENDENT_SRC})
target_include_directories(cufinufft_32_objects PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
target_compile_definitions(cufinufft_32_objects PUBLIC CUFINUFFT_SINGLE)
set_property(TARGET cufinufft_32_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(cufinufft_64_objects OBJECT ${PRECISION_DEPENDENT_SRC})
target_include_directories(cufinufft_64_objects PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
set_property(TARGET cufinufft_64_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(cufinufft SHARED
  $<TARGET_OBJECTS:cufinufft_common_objects>
  $<TARGET_OBJECTS:cufinufft_32_objects>
  $<TARGET_OBJECTS:cufinufft_64_objects>
)
target_link_libraries(cufinufft CUDA::cudart CUDA::cufft CUDA::nvToolsExt)
add_library(cufinufft_static STATIC
  $<TARGET_OBJECTS:cufinufft_common_objects>
  $<TARGET_OBJECTS:cufinufft_32_objects>
  $<TARGET_OBJECTS:cufinufft_64_objects>
)
target_link_libraries(cufinufft_static CUDA::cudart_static CUDA::cufft_static CUDA::nvToolsExt)
set_property(TARGET cufinufft_static PROPERTY OUTPUT_NAME cufinufft)

add_subdirectory(test)
add_subdirectory(examples)
