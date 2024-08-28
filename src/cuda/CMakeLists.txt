set(PRECISION_INDEPENDENT_SRC
    precision_independent.cu utils.cpp
    ${PROJECT_SOURCE_DIR}/contrib/legendre_rule_fast.cpp)
set(PRECISION_DEPENDENT_SRC
    spreadinterp.cpp
    1d/cufinufft1d.cu
    1d/spread1d_wrapper.cu
    1d/interp1d_wrapper.cu
    2d/cufinufft2d.cu
    2d/spread2d_wrapper.cu
    2d/interp2d_wrapper.cu
    3d/spread3d_wrapper.cu
    3d/interp3d_wrapper.cu
    3d/cufinufft3d.cu
    memtransfer_wrapper.cu
    deconvolve_wrapper.cu
    cufinufft.cu
    common.cu)

set(CUFINUFFT_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/contrib
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:CUDA::cufft,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:CUDA::nvToolsExt,INTERFACE_INCLUDE_DIRECTORIES>)
set(CUFINUFFT_INCLUDE_DIRS
    ${CUFINUFFT_INCLUDE_DIRS}
    PARENT_SCOPE)

# flush denormals to zero and enable verbose PTXAS output
set(FINUFFT_CUDA_FLAGS
    -ftz=true
    -fmad=true
    -restrict
    --extra-device-vectorization
    $<$<CONFIG:Debug>:-G
    -maxrregcount
    32>)

add_library(cufinufft_common_objects OBJECT ${PRECISION_INDEPENDENT_SRC})
target_include_directories(cufinufft_common_objects
                           PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
set_target_properties(
  cufinufft_common_objects
  PROPERTIES POSITION_INDEPENDENT_CODE ${FINUFFT_SHARED_LINKING}
             CUDA_ARCHITECTURES "${FINUFFT_CUDA_ARCHITECTURES}"
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON)
target_compile_features(cufinufft_common_objects PRIVATE cxx_std_17)
target_compile_options(
  cufinufft_common_objects
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FINUFFT_CUDA_FLAGS}>)
target_compile_options(
  cufinufft_common_objects
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FINUFFT_CUDA_FLAGS}>)

add_library(cufinufft_objects OBJECT ${PRECISION_DEPENDENT_SRC})
target_include_directories(cufinufft_objects PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
set_target_properties(
  cufinufft_objects
  PROPERTIES POSITION_INDEPENDENT_CODE ${FINUFFT_SHARED_LINKING}
             CUDA_ARCHITECTURES "${FINUFFT_CUDA_ARCHITECTURES}"
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON)
target_compile_features(cufinufft_objects PRIVATE cxx_std_17)
target_compile_options(
  cufinufft_objects PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FINUFFT_CUDA_FLAGS}>)

if(FINUFFT_SHARED_LINKING)
  add_library(cufinufft SHARED $<TARGET_OBJECTS:cufinufft_common_objects>
                               $<TARGET_OBJECTS:cufinufft_objects>)
else()
  add_library(cufinufft STATIC $<TARGET_OBJECTS:cufinufft_common_objects>
                               $<TARGET_OBJECTS:cufinufft_objects>)
endif()

set_target_properties(
  cufinufft
  PROPERTIES CUDA_ARCHITECTURES "${FINUFFT_CUDA_ARCHITECTURES}"
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(cufinufft PRIVATE cxx_std_17)

if(WIN32)
  target_link_libraries(cufinufft PUBLIC CUDA::cudart CUDA::cufft
                                         CUDA::nvToolsExt)
else()
  target_link_libraries(cufinufft PUBLIC CUDA::cudart_static CUDA::cufft_static
                                         CUDA::nvToolsExt)
endif()

file(GLOB CUFINUFFT_PUBLIC_HEADERS "${CMAKE_SOURCE_DIR}/include/cufinufft*.h")
set_target_properties(cufinufft PROPERTIES PUBLIC_HEADER
                                           "${CUFINUFFT_PUBLIC_HEADERS}")
