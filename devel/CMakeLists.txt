project(finufft_devel)
# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.5)

# include cpm cmake, downloading it
CPMAddPackage(
    NAME
    benchmark
    GITHUB_REPOSITORY
    google/benchmark
    VERSION
    1.8.3
    OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
)

if(benchmark_ADDED)
    # patch benchmark target
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 17)
endif()

add_executable(foldrescale foldrescale.cpp)
target_link_libraries(foldrescale finufft benchmark xsimd)
add_executable(padding padding.cpp)
target_compile_features(padding PRIVATE cxx_std_17)
target_link_libraries(padding finufft xsimd)
target_compile_options(padding PRIVATE -march=native)
add_executable(guru_benchmark guru_benchmark.cpp)
target_link_libraries(guru_benchmark finufft benchmark)

add_executable(binsort_gbench binsort_bench.cpp)
target_link_libraries(binsort_gbench finufft benchmark xsimd)
target_compile_options(binsort_gbench PRIVATE ${FINUFFT_ARCH_FLAGS})
add_executable(binsort_gbenchf binsort_bench.cpp)
target_compile_definitions(binsort_gbenchf PRIVATE -DSINGLE)
target_link_libraries(binsort_gbenchf finufft benchmark xsimd)
target_compile_options(binsort_gbenchf PRIVATE ${FINUFFT_ARCH_FLAGS})
