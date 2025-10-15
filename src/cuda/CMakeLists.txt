include(staticAnalysis)
include(setupIWYU)
set(PRECISION_INDEPENDENT_SRC precision_independent.cu ../utils.cpp)

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
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/contrib>
    $<INSTALL_INTERFACE:include>
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:CUDA::cufft,INTERFACE_INCLUDE_DIRECTORIES>
)

set(CUFINUFFT_INCLUDE_DIRS ${CUFINUFFT_INCLUDE_DIRS} PARENT_SCOPE)

list(APPEND FINUFFT_CUDA_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --extra-device-vectorization>)

if(FINUFFT_SHARED_LINKING)
    add_library(cufinufft SHARED ${PRECISION_INDEPENDENT_SRC} ${PRECISION_DEPENDENT_SRC})
else()
    add_library(cufinufft STATIC ${PRECISION_INDEPENDENT_SRC} ${PRECISION_DEPENDENT_SRC})
endif()

enable_static_analysis(cufinufft)
add_iwyu_fix_target(cufinufft)
set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION})
add_library(finufft::cufinufft ALIAS cufinufft)
target_include_directories(cufinufft PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
# set target build location
set_target_properties(cufinufft PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

set(FINUFFT_CUDA_VISIBILITY_PRESET hidden)
if(FINUFFT_BUILD_TESTS)
    set(FINUFFT_CUDA_VISIBILITY_PRESET default)
endif()

set_target_properties(
    cufinufft
    PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION}
        POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
        LINKER_LANGUAGE CUDA
        CXX_VISIBILITY_PRESET ${FINUFFT_CUDA_VISIBILITY_PRESET}
        CUDA_VISIBILITY_PRESET ${FINUFFT_CUDA_VISIBILITY_PRESET}
        VISIBILITY_INLINES_HIDDEN YES
        CUDA_RUNTIME_LIBRARY Shared
        WINDOWS_EXPORT_ALL_SYMBOLS NO
)

if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true" AND MSVC)
    message(STATUS "GITHUB_ACTIONS on Windows using c++20 for cufinufft")
    target_compile_features(cufinufft PRIVATE cxx_std_20 cuda_std_20)
else()
    target_compile_features(cufinufft PRIVATE cxx_std_17 cuda_std_17)
endif()
target_compile_options(cufinufft PUBLIC ${FINUFFT_CUDA_FLAGS})

if(FINUFFT_SHARED_LINKING)
    target_compile_definitions(cufinufft PRIVATE FINUFFT_DLL)
    if(WIN32)
        target_compile_definitions(cufinufft PRIVATE dll_EXPORTS)
    endif()
endif()

target_link_libraries(cufinufft PRIVATE CUDA::cudart CUDA::cufft)
# Expose only when not doing fully static linking
if(NOT FINUFFT_STATIC_LINKING)
    target_link_libraries(cufinufft PUBLIC CUDA::cudart CUDA::cufft)
endif()

target_link_libraries(cufinufft PRIVATE CCCL::CCCL)
if(NOT FINUFFT_STATIC_LINKING)
    target_link_libraries(cufinufft INTERFACE CCCL::CCCL)
endif()

check_cxx_compiler_flag(-Wno-deprecated-declarations FINUFFT_HAS_NO_DEPRECATED_DECLARATIONS)
# disable deprecated warnings for tests if supported
if(FINUFFT_HAS_NO_DEPRECATED_DECLARATIONS)
    target_compile_options(cufinufft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)
endif()

file(GLOB CUFINUFFT_PUBLIC_HEADERS "${CMAKE_SOURCE_DIR}/include/cufinufft*.h")
set_target_properties(cufinufft PROPERTIES PUBLIC_HEADER "${CUFINUFFT_PUBLIC_HEADERS}")
