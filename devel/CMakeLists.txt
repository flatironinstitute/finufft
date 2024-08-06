project(finufft_devel)
# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.5)

# include cpm cmake, downloading it
cpmaddpackage(
  NAME
  benchmark
  GITHUB_REPOSITORY
  google/benchmark
  VERSION
  1.8.3
  OPTIONS
  "BENCHMARK_ENABLE_TESTING OFF")

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
