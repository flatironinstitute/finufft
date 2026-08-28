# Standalone downstream project: verifies an *installed* cufinufft can be
# consumed via find_package(finufft) + target_link_libraries(finufft::cufinufft).
#
# The GPU twin of ../CMakeLists.txt, and the regression guard for the three
# packaging defects that surface at configure and link time, with or without a
# device: an empty link interface on the installed cufinufft, the CUDA headers
# missing from its interface include directories, and a finufftConfig.cmake that
# forgets find_dependency(CUDAToolkit).
cmake_minimum_required(VERSION 3.19)
# CXX only, deliberately: docs/install.rst promises the consumer does not have
# to enable the CUDA language, because the public headers are plain C++ and
# cufinufft carries CUDA::cudart and CUDA::cufft on its link interface.
project(cufinufft_consume LANGUAGES CXX)

find_package(finufft REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::cufinufft)
target_compile_features(app PRIVATE cxx_std_17)
