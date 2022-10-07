cmake_minimum_required(VERSION 3.19)

project(finufft VERSION 2.1.0 LANGUAGES C CXX)

set(GNU_LIKE_FRONTENDS AppleClang Clang GNU)
if (CMAKE_CXX_COMPILER_ID IN_LIST GNU_LIKE_FRONTENDS)
    # Set custom compiler flags for gcc-compatible compilers
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -funroll-loops")
endif ()

include(CTest)

option(FINUFFT_ARCH_FLAGS "Compiler flags for specifying target architecture." -march=native)
option(FINUFFT_FFTW_SUFFIX "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.)" OpenMP)
option(FINUFFT_BUILD_EXAMPLES "Whether to build the FINUFFT examples" OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_BUILD_FORTRAN "Whether to build the FINUFFT Fortran examples" OFF)
option(FINUFFT_BUILD_MATLAB "Whether to build the FINUFFT Matlab interface" OFF)
option(FINUFFT_ENABLE_SANITIZERS "Whether to enable sanitizers, only effective for Debug configuration." ON)
option(FINUFFT_USE_OPENMP "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library." ON)
option(FINUFFT_FFTW_LIBRARIES "Specify a custom FFTW library" DEFAULT)

if (FINUFFT_BUILD_MATLAB)
    # When building for matlab, we will fetch the OpenMP library used by matlab
    # instead of system default for compatibility.
    find_package(Matlab REQUIRED)
    find_library(matlab_iomp5_lib NAMES iomp5 HINTS ${Matlab_ROOT_DIR}/sys/os/ PATH_SUFFIXES glnxa64 maci64)
    find_library(pthreads_lib NAMES pthread CMAKE_FIND_ROOT_PATH_BOTH)

    # Create a "fake" imported library pointing to the matlab openmp implementation
    add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_LOCATION ${matlab_iomp5_lib})
    target_link_libraries(OpenMP::OpenMP_CXX INTERFACE ${pthreads_lib})
    # Set the OpenMP flag.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE -Xclang -fopenmp)
    else ()
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE -fopenmp)
    endif ()
else ()
    # For non-matlab builds, find system OpenMP
    if (FINUFFT_USE_OPENMP)
        find_package(OpenMP REQUIRED)
    endif ()
endif ()

# Find required packages
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (NOT FINUFFT_FFTW_LIBRARIES STREQUAL DEFAULT)
    find_package(FFTW)
    if (NOT FFTW_FOUND)
        # USING CPM TO HANDLE DEPENDENCIES
        set(CPM_DOWNLOAD_VERSION 0.34.0)
        if (CPM_SOURCE_CACHE)
            set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
        elseif (DEFINED ENV{CPM_SOURCE_CACHE})
            set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
        else ()
            set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
        endif ()

        if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
            message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
            file(DOWNLOAD
                    https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
                    ${CPM_DOWNLOAD_LOCATION}
                    )
        endif ()

        include(${CPM_DOWNLOAD_LOCATION})

        set(FFTW_VERSION 3.3.10)
        if (FINUFFT_FFTW_SUFFIX STREQUAL THREADS)
            set(FINUFFT_USE_THREADS ON)
        else ()
            set(FINUFFT_USE_THREADS OFF)
        endif ()
        CPMAddPackage(
                NAME fftw3
                OPTIONS "ENABLE_AVX2 ON" "BUILD_TESTS OFF" "BUILD_SHARED_LIBS OFF" "ENABLE_THREADS ${FINUFFT_USE_THREADS}" "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
                URL "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
                URL_HASH "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
                EXCLUDE_FROM_ALL YES
                GIT_SHALLOW YES
        )

        CPMAddPackage(
                NAME fftw3f
                OPTIONS "ENABLE_AVX2 ON" "BUILD_TESTS OFF" "BUILD_SHARED_LIBS OFF" "ENABLE_FLOAT ON" "ENABLE_THREADS ${FINUFFT_USE_THREADS}" "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
                URL "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
                URL_HASH "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
                EXCLUDE_FROM_ALL YES
                GIT_SHALLOW YES
        )
        set(FINUFFT_FFTW_LIBRARIES fftw3 fftw3f)
        if (FINUFFT_USE_THREADS)
            list(APPEND FINUFFT_FFTW_LIBRARIES fftw3_threads fftw3f_threads)
        elseif (FINUFFT_USE_OPENMP)
            list(APPEND FINUFFT_FFTW_LIBRARIES fftw3_omp fftw3f_omp)
        endif ()

        foreach (element IN LISTS FINUFFT_FFTW_LIBRARIES)
            set_property(TARGET ${element} PROPERTY POSITION_INDEPENDENT_CODE ON)
        endforeach ()

    else ()
        set(FINUFFT_FFTW_LIBRARIES "FFTW::Float" "FFTW::Double" "FFTW::Float${FINUFFT_FFTW_SUFFIX}" "FFTW::Double${FINUFFT_FFTW_SUFFIX}")
    endif ()
endif ()
# This set of sources is compiled twice, once in single precision and once in double precision
# The single precision compilation is done with -DSINGLE
set(FINUFFT_PRECISION_DEPENDENT_SOURCES src/finufft.cpp src/simpleinterfaces.cpp src/spreadinterp.cpp src/utils.cpp fortran/finufftfort.cpp)

# Set of compilers which behave like gcc
set(FINUFFT_GNU_LIKE_COMPILERS AppleClang Clang GNU)

# Utility function to enable ASAN on debug builds
function(enable_asan target)
    if (NOT FINUFFT_ENABLE_SANITIZERS)
        return()
    endif ()

    if (CMAKE_CXX_COMPILER_ID IN_LIST FINUFFT_GNU_LIKE_COMPILERS)
        # Enable only on clang / gcc compilers.
        target_compile_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=address>)
        target_link_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=address>)
    endif ()
endfunction()

# Utility function to set finufft compilation options.
function(set_finufft_options target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target} PROPERTY CMAKE_CXX_STANDARD 14)
    enable_asan(${target})

    target_compile_options(${target} PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Enable cx-limited-range on supported compilers
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-fcx-limited-range>)
    endif ()

    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if (FINUFFT_USE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    else ()
        if (CMAKE_CXX_COMPILER_ID IN_LIST FINUFFT_GNU_LIKE_COMPILERS)
            # OpenMP disabled, suppress unknown pragma warnings to avoid spam.
            target_compile_options(${target} PRIVATE "-Wno-unknown-pragmas")
        endif ()
    endif ()
    if (NOT FFTW_FOUND)
        target_include_directories(${target} PRIVATE ${fftw3_SOURCE_DIR}/api)

    endif()
endfunction()

# Main finufft libraries
add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
target_compile_definitions(finufft_f32 PRIVATE SINGLE)
set_finufft_options(finufft_f32)
target_link_libraries(finufft_f32 PUBLIC ${FINUFFT_FFTW_LIBRARIES})

add_library(finufft_f64 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
set_finufft_options(finufft_f64)
target_link_libraries(finufft_f64 PUBLIC ${FINUFFT_FFTW_LIBRARIES})

add_library(finufft SHARED src/utils_precindep.cpp contrib/legendre_rule_fast.cpp)
set_finufft_options(finufft)
target_link_libraries(finufft PUBLIC finufft_f32 finufft_f64 m)
target_include_directories(finufft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(finufft_static STATIC src/utils_precindep.cpp contrib/legendre_rule_fast.cpp)
set_finufft_options(finufft)
target_link_libraries(finufft_static PUBLIC finufft_f32 finufft_f64 m)
target_include_directories(finufft_static PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Add tests defined in their own directory
if (BUILD_TESTING AND FINUFFT_BUILD_TESTS)
    add_subdirectory(test)
    add_subdirectory(perftest)
endif ()

if (FINUFFT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (FINUFFT_BUILD_FORTRAN)
    enable_language(Fortran)
    add_subdirectory(fortran)
endif ()

if (FINUFFT_BUILD_MATLAB)
    add_subdirectory(matlab)
endif ()
