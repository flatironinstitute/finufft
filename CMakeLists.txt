cmake_minimum_required(VERSION 3.19)

project(finufft VERSION 2.0.4 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(FFTW REQUIRED)
find_package(OpenMP REQUIRED)

include(FetchContent)

# Install google benchmark from git
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/v1.6.0.zip
)
FetchContent_MakeAvailable(googlebenchmark)


# This set of sources is compiled twice, once in single precision and once in double precision
# The single precision compilation is done with -DSINGLE
set(FINUFFT_PRECISION_DEPENDENT_SOURCES src/finufft.cpp src/simpleinterfaces.cpp src/spreadinterp.cpp src/utils.cpp)

function(enable_asan target)
    target_compile_options(${target} PUBLIC $<$<CONFIG:DEBUG>:-fsanitize=address>)
    target_link_options(${target} PUBLIC $<$<CONFIG:DEBUG>:-fsanitize=address>)
endfunction()

function(set_finufft_options target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target} PROPERTY CMAKE_CXX_STANDARD 14)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-O3 -funroll-loops -march=native -fcx-limited-range>)
    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

# Main finufft libraries
add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
target_compile_definitions(finufft_f32 PRIVATE SINGLE)
set_finufft_options(finufft_f32)
target_link_libraries(finufft_f32 PUBLIC FFTW::Float FFTW::FloatOpenMP)

add_library(finufft_f64 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
set_finufft_options(finufft_f64)
target_link_libraries(finufft_f64 PUBLIC FFTW::Double FFTW::DoubleOpenMP)

add_library(finufft SHARED src/utils_precindep.cpp contrib/legendre_rule_fast.c)
set_property(TARGET finufft PROPERTY CMAKE_CXX_STANDARD 14)
target_link_libraries(finufft PUBLIC finufft_f32 finufft_f64 m)
target_include_directories(finufft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
enable_asan(finufft)

# Custom random123 interface definition
add_library(random123 INTERFACE)
target_include_directories(random123 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/random123/include)

# Spreader perf test executable
add_executable(spreadtestnd perftest/spreadtestnd.cpp)
target_link_libraries(spreadtestnd PUBLIC finufft)

add_executable(spreadtestndf perftest/spreadtestnd.cpp)
target_compile_definitions(spreadtestndf PRIVATE SINGLE)
target_link_libraries(spreadtestndf PUBLIC finufft)

# Spreader perf benchmark
add_executable(bench_spread perftest/benchmark_spread.cpp)
target_link_libraries(bench_spread PRIVATE finufft random123 benchmark::benchmark_main)
set_property(TARGET bench_spread PROPERTY CMAKE_CXX_STANDARD 14)
target_compile_options(bench_spread PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-O3 -funroll-loops -march=native -fcx-limited-range>)
enable_asan(bench_spread)

# One-dim type-3 perf benchmark
add_executable(bench_1d_type3 perftest/benchmark_1d_type3.cpp)
target_link_libraries(bench_1d_type3 PRIVATE finufft random123 benchmark::benchmark_main)
set_property(TARGET bench_1d_type3 PROPERTY CMAKE_CXX_STANDARD 14)
target_compile_options(bench_spread PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-O3 -funroll-loops -march=native -fcx-limited-range>)
enable_asan(bench_spread)

# Add tests defined in their own directory
add_subdirectory(test)

