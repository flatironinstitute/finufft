project(finufft_devel)
# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.5)


# include cpm cmake, downloading it
CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    VERSION 1.8.3
    OPTIONS "BENCHMARK_ENABLE_TESTING OFF"

)

if (benchmark_ADDED)
    # patch benchmark target
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 17)
endif()

add_executable(foldrescale foldrescale.cpp)
target_link_libraries(foldrescale finufft benchmark)
#target_compile_options(foldrescale PRIVATE -mavx2)
