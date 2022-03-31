cmake_minimum_required(VERSION 3.19)

project(finufft VERSION 2.0.4 LANGUAGES C CXX)

if(CMAKE_CXX_COMPILER_ID IN_LIST "AppleClang;Clang;GNU")
    # Set custom compiler flags for gcc-compatible compilers
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops -fcx-limited-range")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -funroll-loops -fcx-limited-range")
endif()

include(CTest)

set(FINUFFT_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
set(FINUFFT_FFTW_SUFFIX "OpenMP" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.)")
set(FINUFFT_BUILD_EXAMPLES "OFF" CACHE BOOL "Whether to build the FINUFFT examples")
set(FINUFFT_BUILD_TESTS "OFF" CACHE BOOL "Whether to build the FINUFFT tests")
set(FINUFFT_BUILD_FORTRAN "OFF" CACHE BOOL "Whether to build the FINUFFT Fortran examples")
set(FINUFFT_BUILD_MATLAB "OFF" CACHE BOOL "Whether to build the FINUFFT Matlab interface")

# Find required packages
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)
find_package(OpenMP REQUIRED)

set(FINUFFT_FFTW_LIBRARIES "FFTW::Float" "FFTW::Double" "FFTW::Float${FINUFFT_FFTW_SUFFIX}" "FFTW::Double${FINUFFT_FFTW_SUFFIX}")

# This set of sources is compiled twice, once in single precision and once in double precision
# The single precision compilation is done with -DSINGLE
set(FINUFFT_PRECISION_DEPENDENT_SOURCES src/finufft.cpp src/simpleinterfaces.cpp src/spreadinterp.cpp src/utils.cpp fortran/finufftfort.cpp)

# Utility function to enable ASAN on debug builds
function(enable_asan target)
    if(CMAKE_CXX_COMPILER_ID IN_LIST "AppleClang;Clang;GNU")
        # Enable only on clang / gcc compilers.
        target_compile_options(${target} PUBLIC $<$<CONFIG:DEBUG>:-fsanitize=address>)
        target_link_options(${target} PUBLIC $<$<CONFIG:DEBUG>:-fsanitize=address>)
    endif()
endfunction()

# Utility function to set finufft compilation options.
function(set_finufft_options target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target} PROPERTY CMAKE_CXX_STANDARD 14)
    enable_asan(${target})
    target_compile_options(${target} PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

# Main finufft libraries
add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
target_compile_definitions(finufft_f32 PRIVATE SINGLE)
set_finufft_options(finufft_f32)
target_link_libraries(finufft_f32 PUBLIC ${FINUFFT_FFTW_LIBRARIES})

add_library(finufft_f64 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
set_finufft_options(finufft_f64)
target_link_libraries(finufft_f64 PUBLIC ${FINUFFT_FFTW_LIBRARIES})

add_library(finufft SHARED src/utils_precindep.cpp contrib/legendre_rule_fast.c)
set_finufft_options(finufft)
target_link_libraries(finufft PUBLIC finufft_f32 finufft_f64 m)
target_include_directories(finufft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(finufft_static STATIC src/utils_precindep.cpp contrib/legendre_rule_fast.c)
set_finufft_options(finufft)
target_link_libraries(finufft_static PUBLIC finufft_f32 finufft_f64 m)
target_include_directories(finufft_static PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Add tests defined in their own directory
if(BUILD_TESTING AND FINUFFT_BUILD_TESTS)
add_subdirectory(test)
add_subdirectory(perftest)
endif()

if(FINUFFT_BUILD_EXAMPLES)
add_subdirectory(examples)
endif()

if(FINUFFT_BUILD_FORTRAN)
enable_language(Fortran)
add_subdirectory(fortran)
endif()

if(FINUFFT_BUILD_MATLAB)
add_subdirectory(matlab)
endif()
