
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(PRECISION_INDEPENDENT_SRC precision_independent.cu profile.cu utils.cpp
  ${PROJECT_SOURCE_DIR}/contrib/legendre_rule_fast.cpp)
set(PRECISION_DEPENDENT_SRC
  spreadinterp.cpp
  1d/cufinufft1d.cu 1d/spread1d_wrapper.cu 1d/interp1d_wrapper.cu
  2d/cufinufft2d.cu 2d/spread2d_wrapper.cu 2d/interp2d_wrapper.cu
  3d/spread3d_wrapper.cu 3d/interp3d_wrapper.cu 3d/cufinufft3d.cu
  memtransfer_wrapper.cu deconvolve_wrapper.cu cufinufft.cu common.cu
)

set(CUFINUFFT_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/contrib
  ${PROJECT_SOURCE_DIR}/include/cufinufft/contrib/cuda_samples
  $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:CUDA::cufft,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:CUDA::nvToolsExt,INTERFACE_INCLUDE_DIRECTORIES>
)
set(CUFINUFFT_INCLUDE_DIRS ${CUFINUFFT_INCLUDE_DIRS} PARENT_SCOPE)

add_library(cufinufft_common_objects OBJECT ${PRECISION_INDEPENDENT_SRC})
target_include_directories(cufinufft_common_objects PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
set_property(TARGET cufinufft_common_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(cufinufft_objects OBJECT ${PRECISION_DEPENDENT_SRC})
target_include_directories(cufinufft_objects PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
set_property(TARGET cufinufft_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(cufinufft SHARED
  $<TARGET_OBJECTS:cufinufft_common_objects>
  $<TARGET_OBJECTS:cufinufft_objects>
)
target_link_libraries(cufinufft CUDA::cudart CUDA::cufft CUDA::nvToolsExt)
set_target_properties(
  cufinufft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_library(cufinufft_static STATIC
  $<TARGET_OBJECTS:cufinufft_common_objects>
  $<TARGET_OBJECTS:cufinufft_objects>
)
target_link_libraries(cufinufft_static PUBLIC CUDA::cudart_static CUDA::cufft_static CUDA::nvToolsExt)
set_target_properties(
  cufinufft_static PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# if(FINUFFT_BUILD_TESTS)
# add_subdirectory(test)
# endif()
