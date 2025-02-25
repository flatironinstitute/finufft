set(PRECISION_INDEPENDENT_SRC
    precision_independent.cu
    utils.cpp
    ${PROJECT_SOURCE_DIR}/contrib/legendre_rule_fast.cpp
)

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
    common.cu
)

set(CUFINUFFT_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/contrib
    $<TARGET_PROPERTY:CUDA::cufft,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>
)

set(CUFINUFFT_INCLUDE_DIRS ${CUFINUFFT_INCLUDE_DIRS} PARENT_SCOPE)

# flush denormals to zero and enable verbose PTXAS output
set(FINUFFT_CUDA_FLAGS
    $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    -ftz=true
    -fmad=true
    --extra-device-vectorization
    $<$<CONFIG:Debug>:-G
    -maxrregcount
    64
    >
    >
)

if(FINUFFT_SHARED_LINKING)
    add_library(
        cufinufft
        SHARED
        ${PRECISION_INDEPENDENT_SRC}
        ${PRECISION_DEPENDENT_SRC}
    )
else()
    add_library(
        cufinufft
        STATIC
        ${PRECISION_INDEPENDENT_SRC}
        ${PRECISION_DEPENDENT_SRC}
    )
    set_target_properties(
        cufinufft
        PROPERTIES
            POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
    )
endif()
target_include_directories(cufinufft PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
# set target build location
set_target_properties(
    cufinufft
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(DEFINED CIBUILDWHEEL)
    set(CUDA_LINK_TYPE Shared)
    message(STATUS RUNNING FROM CIBUILDWHEEL)
else()
    set(CUDA_LINK_TYPE Static)
endif()

set_target_properties(
    cufinufft
    PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        CUDA_RUNTIME_LIBRARY ${CUDA_LINK_TYPE}
)
target_compile_features(cufinufft PRIVATE cxx_std_17)
target_compile_options(cufinufft PRIVATE ${FINUFFT_CUDA_FLAGS})
target_link_libraries(cufinufft PUBLIC CUDA::cufft CUDA::cudart)

file(GLOB CUFINUFFT_PUBLIC_HEADERS "${CMAKE_SOURCE_DIR}/include/cufinufft*.h")
set_target_properties(
    cufinufft
    PROPERTIES PUBLIC_HEADER "${CUFINUFFT_PUBLIC_HEADERS}"
)
