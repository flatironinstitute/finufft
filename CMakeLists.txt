cmake_minimum_required(VERSION 3.23)

project(FINUFFT VERSION 2.2.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CheckCXXCompilerFlag)

# Define the function
function(filter_supported_compiler_flags input_flags_var output_flags_var)
    # Create an empty list to store supported flags
    set(supported_flags)
    # Iterate over each flag in the input list
    foreach(flag ${${input_flags_var}})
        check_cxx_compiler_flag(${flag} FLAG_SUPPORTED)
        if(${FLAG_SUPPORTED})
            # If supported, append the flag to the list of supported flags
            message(STATUS "Flag ${flag} is supported")
            list(APPEND supported_flags ${flag})
        else()
            message(STATUS "Flag ${flag} is not supported")
        endif()
        unset(FLAG_SUPPORTED CACHE)
    endforeach()
    # Set the output variable to the list of supported flags
    set(${output_flags_var} ${supported_flags} PARENT_SCOPE)
endfunction()

# Define a list of additional flags to check for support
set(FINUFFT_CXX_FLAGS_RELEASE
        -funroll-loops
        -ffp-contract=fast
        -fno-math-errno
        -fno-signed-zeros
        -fno-trapping-math
        -fassociative-math
        -freciprocal-math
        -fmerge-all-constants
        -ftree-vectorize
        -fimplicit-constexpr
        -fcx-limited-range
        /fp:fast
        -O3
        /O2
)
filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_RELEASE FINUFFT_CXX_FLAGS_RELEASE)
message(STATUS "FINUFFT Release flags: ${FINUFFT_CXX_FLAGS_RELEASE}")
set(FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_RELEASE})

set(FINUFFT_CXX_FLAGS_DEBUG -g -g3 -ggdb -ggdb3 /Zi /ZI -Wall -Wno-sign-compare -Wno-unknown-pragmas)
filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_DEBUG FINUFFT_CXX_FLAGS_DEBUG)
message(STATUS "FINUFFT Debug flags: ${FINUFFT_CXX_FLAGS_DEBUG}")
list (APPEND FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_RELEASE} ${FINUFFT_CXX_FLAGS_DEBUG})
message(STATUS "FINUFFT RelWithDebInfo flags: ${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}")

set(FINUFFT_ARCH_FLAGS -march=native CACHE STRING "Compiler flags for specifying target architecture.")
filter_supported_compiler_flags(FINUFFT_ARCH_FLAGS FINUFFT_ARCH_FLAGS)
if (NOT FINUFFT_ARCH_FLAGS)
    message(WARNING "Architecture flags are not supported by the compiler.")
    set(FINUFFT_ARCH_FLAGS -mtune=native)
    filter_supported_compiler_flags(FINUFFT_ARCH_FLAGS FINUFFT_ARCH_FLAGS)
endif()
if (NOT FINUFFT_ARCH_FLAGS)
    message(WARNING "No architecture flags are supported by the compiler.")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Check for AVX, AVX512 and SSE support on windows
    message(STATUS "Checking for AVX, AVX512 and SSE support")
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckAVX.cpp
            COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
            RUN_OUTPUT_VARIABLE RUN_OUTPUT)
    if (RUN_OUTPUT MATCHES "AVX512")
        set(FINUFFT_ARCH_FLAGS "/arch:AVX512" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "AVX")
        set(FINUFFT_ARCH_FLAGS "/arch:AVX" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "SSE")
        set(FINUFFT_ARCH_FLAGS "/arch:SSE" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(FINUFFT_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    message(STATUS "CPU supports: ${RUN_OUTPUT}")
    message(STATUS "Using MSVC flags: ${FINUFFT_ARCH_FLAGS}")
endif ()
    message(STATUS "FINUFFT Arch flags: ${FINUFFT_ARCH_FLAGS}")

include(CTest)
# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Set the default build type to Release" FORCE)
endif()

set(FINUFFT_FFTW_SUFFIX "OpenMP" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.)")
set(FINUFFT_FFTW_LIBRARIES "DEFAULT" CACHE STRING "Specify a custom FFTW library")
set(FINUFFT_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for (e.g. 60;70;75;)")
# All options go here
# sphinx tag (don't remove): @cmake_opts_start
option(FINUFFT_BUILD_EXAMPLES "Whether to build the FINUFFT examples" OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_BUILD_FORTRAN "Whether to build the FINUFFT Fortran examples" OFF)
option(FINUFFT_BUILD_MATLAB "Whether to build the FINUFFT Matlab interface" OFF)
option(FINUFFT_BUILD_PYTHON "Whether the Python wrapper should be built." OFF)
option(FINUFFT_ENABLE_SANITIZERS "Whether to enable sanitizers, only effective for Debug configuration." ON)
option(FINUFFT_USE_OPENMP "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library." ON)
option(FINUFFT_USE_CUDA "Whether to build CUDA accelerated FINUFFT library (libcufinufft). This is completely independent of the main FINUFFT library" OFF)
option(FINUFFT_USE_CPU "Whether to build the ordinary FINUFFT library (libfinufft)." ON)
option(FINUFFT_STATIC_LINKING "Whether to build the static FINUFFT library." ON)
option(FINUFFT_BUILD_DEVEL "Whether to build development executables" OFF)
option(FINUFFT_USE_DUCC0 "Whether to use DUCC0 (instead of FFTW) for CPU FFTs" OFF)
option(FINUFFT_BUILD_DEVEL "Whether to build developement executables" OFF)
option(FINUFFT_ENABLE_INSTALL "Whether to enable installation of FINUFFT library" ON)
# sphinx tag (don't remove): @cmake_opts_end

if (NOT FINUFFT_STATIC_LINKING)
    set (FINUFFT_SHARED_LINKING ON)
else ()
    set (FINUFFT_SHARED_LINKING OFF)
endif ()

if (FINUFFT_USE_CPU)
    # suppress Windows warnings about "unsafe" functions
    if (WIN32)
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    endif ()

    # make apple with gnu use old linker, new linker breaks, see issue #360
    if ((APPLE) AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        add_link_options("-ld64")
    endif ()

    set(CPM_DOWNLOAD_VERSION 0.40.0)
    set(FFTW_VERSION 3.3.10)
    set(XTL_VERSION 0.7.7)
    set(XSIMD_VERSION 13.0.0)
    set(DUCC0_VERSION ducc0_0_34_0)
    set(FINUFFT_FFTW_LIBRARIES)
    include(cmake/setupCPM.cmake)
    include(cmake/setupXSIMD.cmake)
    include(cmake/setupSanitizers.cmake)
    if (FINUFFT_USE_DUCC0)
        include(cmake/setupDUCC.cmake)
    else ()
        include(cmake/setupFFTW.cmake)
    endif ()
endif ()

if (FINUFFT_USE_DUCC0)
    set (FINUFFT_FFTLIBS ducc0)
else ()
    set (FINUFFT_FFTLIBS ${FINUFFT_FFTW_LIBRARIES})
endif ()

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
        find_package(OpenMP COMPONENTS CXX REQUIRED)
    endif ()
endif ()

# This set of sources is compiled twice, once in single precision and once in double precision
# The single precision compilation is done with -DSINGLE
set(FINUFFT_PRECISION_DEPENDENT_SOURCES src/finufft.cpp src/fft.cpp src/simpleinterfaces.cpp src/spreadinterp.cpp src/utils.cpp fortran/finufftfort.cpp)

set(SANITIZE_ADDRESS ON)
set(SANITIZE_UNDEFINED ON)
set(SANITIZE_MEMORY OFF)
set(SANITIZE_THREAD OFF)

# Utility function to enable ASAN on debug builds
function(enable_asan target)
    if (NOT FINUFFT_ENABLE_SANITIZERS OR NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        return()
    endif ()
    find_package(Sanitizers REQUIRED)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "NVCC")
        set (SANITIZE_ADDRESS OFF)
        add_sanitizers(${target})
        set (SANITIZE_ADDRESS ON)
    else ()
        add_sanitizers(${target})
    endif ()
endfunction()

# Utility function to link static/dynamic lib
function(finufft_link_test target)
    if (FINUFFT_USE_DUCC0)
        target_compile_definitions(${target} PRIVATE FINUFFT_USE_DUCC0)
    endif ()
    target_link_libraries(${target} PRIVATE finufft ${FINUFFT_FFTLIBS})
    if(FINUFFT_USE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        if(WIN32)
            target_link_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
            if (FINUFFT_SHARED_LINKING)
                target_compile_definitions(${target} PRIVATE FINUFFT_DLL)
            endif()
        endif()
    endif()
    enable_asan(${target})
    target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

# Utility function to set finufft compilation options.
function(set_finufft_options target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:${FINUFFT_CXX_FLAGS_RELEASE}>)
    target_compile_options(${target} PRIVATE $<$<CONFIG:RelWithDebInfo>:${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}>)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:${FINUFFT_CXX_FLAGS_DEBUG}>)
    target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_include_directories(${target} SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ${FINUFFT_SHARED_LINKING})
    enable_asan(${target})
    if (FINUFFT_USE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        # there are issues on windows with OpenMP and CMake, so we need to manually add the flags
        # otherwise there are link errors
        if (WIN32)
            target_link_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
        endif ()
    endif ()
    if (FINUFFT_USE_DUCC0)
        target_compile_definitions(${target} PRIVATE FINUFFT_USE_DUCC0)
    endif ()
    target_link_libraries(${target} PRIVATE xsimd)
    target_link_libraries(${target} PRIVATE ${FINUFFT_FFTLIBS})
endfunction()

if (FINUFFT_USE_CPU)
    # Main finufft libraries
    add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
    target_compile_definitions(finufft_f32 PRIVATE SINGLE)
    set_finufft_options(finufft_f32)

    add_library(finufft_f64 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
    set_finufft_options(finufft_f64)
    if (NOT FINUFFT_STATIC_LINKING)
        add_library(finufft SHARED src/utils_precindep.cpp contrib/legendre_rule_fast.cpp)
    else ()
        add_library(finufft STATIC src/utils_precindep.cpp contrib/legendre_rule_fast.cpp)
    endif ()
    target_link_libraries(finufft PRIVATE finufft_f32 finufft_f64)
    set_finufft_options(finufft)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND FINUFFT_SHARED_LINKING)
        target_compile_definitions(finufft_f32 PRIVATE dll_EXPORTS FINUFFT_DLL)
        target_compile_definitions(finufft_f64 PRIVATE dll_EXPORTS FINUFFT_DLL)
        target_compile_definitions(finufft PRIVATE dll_EXPORTS FINUFFT_DLL)
    endif ()
    if (NOT WIN32)
        # windows does not have a math library, so we need to exclude it
        target_link_libraries(finufft PUBLIC m)
    endif ()
    target_include_directories(finufft PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_include_directories(finufft SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
    if (FINUFFT_ENABLE_INSTALL)
        file(GLOB FINUFFT_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/finufft*.h")
        set_target_properties(finufft PROPERTIES PUBLIC_HEADER "${FINUFFT_PUBLIC_HEADERS}")
    endif()
    list(APPEND INSTALL_TARGETS finufft)
endif ()

if (FINUFFT_USE_CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message("FINUFFT WARNING: No CUDA architecture supplied via '-DCMAKE_CUDA_ARCHITECTURES=...', defaulting to '60;70;75;'")
        message("See: https://developer.nvidia.com/cuda-gpus for more details on what architecture to supply.")
    endif ()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(src/cuda)
    if (BUILD_TESTING OR FINUFFT_BUILD_TESTS)
        add_subdirectory(perftest/cuda)
        add_subdirectory(test/cuda)
    endif ()

    list(APPEND INSTALL_TARGETS cufinufft)
endif ()

# Add tests defined in their own directory
if (FINUFFT_USE_CPU AND (BUILD_TESTING OR FINUFFT_BUILD_TESTS))
    add_subdirectory(test)
    add_subdirectory(perftest)
endif ()

if (FINUFFT_BUILD_EXAMPLES AND FINUFFT_USE_CPU)
    add_subdirectory(examples)
endif ()

if (FINUFFT_BUILD_EXAMPLES AND FINUFFT_USE_GPU)
    add_subdirectory(examples/cuda)
endif ()

if (FINUFFT_BUILD_FORTRAN)
    enable_language(Fortran)
    add_subdirectory(fortran)
endif ()

if (FINUFFT_BUILD_MATLAB)
    add_subdirectory(matlab)
endif ()

if (FINUFFT_BUILD_DEVEL)
    add_subdirectory(devel)
endif ()

if (FINUFFT_BUILD_PYTHON)
    add_subdirectory(python)
endif ()

message(STATUS " CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "FINUFFT configuration summary:")
message(STATUS "  FINUFFT_USE_CPU: ${FINUFFT_USE_CPU}")
message(STATUS "  FINUFFT_USE_CUDA: ${FINUFFT_USE_CUDA}")
message(STATUS "  FINUFFT_USE_OPENMP: ${FINUFFT_USE_OPENMP}")
message(STATUS "  FINUFFT_STATIC_LINKING: ${FINUFFT_STATIC_LINKING}")
message(STATUS "  FINUFFT_ENABLE_INSTALL: ${FINUFFT_ENABLE_INSTALL}")
message(STATUS "  FINUFFT_BUILD_EXAMPLES: ${FINUFFT_BUILD_EXAMPLES}")
message(STATUS "  FINUFFT_BUILD_TESTS: ${FINUFFT_BUILD_TESTS}")
message(STATUS "  FINUFFT_BUILD_FORTRAN: ${FINUFFT_BUILD_FORTRAN}")
message(STATUS "  FINUFFT_BUILD_MATLAB: ${FINUFFT_BUILD_MATLAB}")
message(STATUS "  FINUFFT_BUILD_PYTHON: ${FINUFFT_BUILD_PYTHON}")
message(STATUS "  FINUFFT_ENABLE_SANITIZERS: ${FINUFFT_ENABLE_SANITIZERS}")
message(STATUS "  FINUFFT_FFTW_SUFFIX: ${FINUFFT_FFTW_SUFFIX}")
message(STATUS "  FINUFFT_FFTW_LIBRARIES: ${FINUFFT_FFTW_LIBRARIES}")
message(STATUS "  FINUFFT_ARCH_FLAGS: ${FINUFFT_ARCH_FLAGS}")

if (FINUFFT_ENABLE_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS ${INSTALL_TARGETS} PUBLIC_HEADER)
    install(FILES ${PROJECT_SOURCE_DIR}/LICENSE
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/finufft)
    if (FINUFFT_USE_CPU)
        install(DIRECTORY ${PROJECT_SOURCE_DIR}/examples
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft
                PATTERN "CMakeLists.txt" EXCLUDE
                PATTERN "README" EXCLUDE
                PATTERN "examples/cuda" EXCLUDE
        )
        if (FINUFFT_BUILD_FORTRAN)
            install(DIRECTORY ${PROJECT_SOURCE_DIR}/fortran/examples
                    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/fortran
            )
            install(FILES ${PROJECT_SOURCE_DIR}/include/finufft.fh
                    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            )
        endif()
    endif ()
    if (FINUFFT_USE_CUDA)
        install(DIRECTORY ${PROJECT_SOURCE_DIR}/examples/cuda
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/examples
                PATTERN "README" EXCLUDE
                PATTERN "CMakeLists.txt" EXCLUDE
        )
    endif()
endif ()
