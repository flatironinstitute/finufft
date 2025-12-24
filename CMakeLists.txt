cmake_minimum_required(VERSION 3.24...3.31)
cmake_policy(VERSION 3.24)
project(FINUFFT VERSION 2.5.0 LANGUAGES C CXX)

include(CMakeDependentOption)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Set the default build type to Release" FORCE)
endif()

# gersemi: off
# All options go here sphinx tag (don't remove): @cmake_opts_start
option(FINUFFT_BUILD_DEVEL "Whether to build development executables" OFF)
option(FINUFFT_BUILD_DOCS "Whether to build the FINUFFT documentation" OFF)
option(FINUFFT_BUILD_EXAMPLES "Whether to build the FINUFFT examples" OFF)
option(FINUFFT_BUILD_FORTRAN "Whether to build the FINUFFT Fortran examples" OFF)
option(FINUFFT_BUILD_MATLAB "Whether to build the FINUFFT Matlab interface" OFF)
option(FINUFFT_BUILD_PYTHON "Whether the Python wrapper should be built." OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_INTERPROCEDURAL_OPTIMIZATION "Enable interprocedural optimization (LTO) if supported" OFF)
option(FINUFFT_POSITION_INDEPENDENT_CODE "Whether to build the finufft library with position independent code (-fPIC). This forced ON when FINUFFT_SHARED_LINKING is ON." ON)
option(FINUFFT_STATIC_ANALYSIS "Run clang-tidy/cppcheck" OFF)
option(FINUFFT_STATIC_LINKING "If ON builds the static finufft library, if OFF build a shared finufft library." ON)
option(FINUFFT_USE_CPU "Whether to build the ordinary FINUFFT library (libfinufft)." ON)
option(FINUFFT_USE_CUDA "Whether to build CUDA accelerated FINUFFT library (libcufinufft). This is completely independent of the main FINUFFT library" OFF)
option(FINUFFT_USE_DUCC0 "Whether to use DUCC0 (instead of FFTW) for CPU FFTs" OFF)
option(FINUFFT_USE_IWYU "Set CXX_INCLUDE_WHAT_YOU_USE on target (checker-only)" OFF)
option(FINUFFT_USE_OPENMP "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library." ON)
option(FINUFFT_USE_SANITIZERS "Whether to enable sanitizers, only effective for Debug configuration." OFF)
# if FINUFFT_USE_DUCC0 is ON, the following options are ignored
set(FINUFFT_FFTW_LIBRARIES "DEFAULT" CACHE STRING "Specify a custom FFTW library")
set(FINUFFT_FFTW_SUFFIX "DEFAULT" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.) defaults to empty string if OpenMP is disabled, else uses OpenMP. Ignored if DUCC0 is used.")
# if FINUFFT_USE_CPU is OFF, the following options are ignored
set(FINUFFT_ARCH_FLAGS "native" CACHE STRING "Compiler flags for specifying target architecture, defaults to -march=native")
# sphinx tag (don't remove): @cmake_opts_end
cmake_dependent_option(FINUFFT_ENABLE_INSTALL "Disable installation in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
cmake_dependent_option(FINUFFT_STATIC_LINKING "Disable static libraries in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
cmake_dependent_option(FINUFFT_SHARED_LINKING "Shared should be the opposite of static linking" ON "NOT FINUFFT_STATIC_LINKING" OFF)
cmake_dependent_option(FINUFFT_INTERPROCEDURAL_OPTIMIZATION "LTO should be the opposite of static linking" ON "NOT FINUFFT_STATIC_LINKING" OFF)
# internal development options
option(FINUFFT_IWYU_VERBOSE "Verbose IWYU tool output" ON)
# gersemi: on

# Use the folder that contains this CMakeLists.txt, not the project root
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# When building shared libraries, we need to build with -fPIC in all cases
if(FINUFFT_SHARED_LINKING)
    set(FINUFFT_POSITION_INDEPENDENT_CODE ON)
endif()

if(FINUFFT_MATLAB_INSTALL)
    set(FINUFFT_BUILD_MATLAB ON)
    set(FINUFFT_ENABLE_INSTALL OFF)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(utils) # finds cmake/utils.cmake
include(toolchain) # finds cmake/toolchain.cmake

# Versions for dependencies
set(CPM_DOWNLOAD_VERSION "0.42.0" CACHE STRING "Version of CPM.cmake to use")
set(FFTW_VERSION "3.3.10" CACHE STRING "Version of FFTW to use")
set(XSIMD_VERSION "14.0.0" CACHE STRING "Version of xsimd to use")
set(DUCC0_VERSION "ducc0_0_39_1" CACHE STRING "Version of ducc0 to use")
set(CUDA11_CCCL_VERSION "2.8.5" CACHE STRING "Version of FINUFFT-cccl for cuda 11 to use")
set(CUDA12_CCCL_VERSION "3.0.2" CACHE STRING "Version of FINUFFT-cccl for cuda 12 to use")

mark_as_advanced(
    CPM_DOWNLOAD_VERSION
    FFTW_VERSION
    XTL_VERSION
    XSIMD_VERSION
    DUCC0_VERSION
    CUDA11_CCCL_VERSION
    CUDA12_CCCL_VERSION
)

include(setupCPM)

if(FINUFFT_USE_CPU)
    add_subdirectory(src)
endif()

add_subdirectory(src/common)

if(FINUFFT_USE_CUDA)
    include(cuda_setup)
    add_subdirectory(src/cuda)
    if(BUILD_TESTING AND FINUFFT_BUILD_TESTS)
        add_subdirectory(perftest/cuda)
        add_subdirectory(test/cuda)
    endif()

    list(APPEND INSTALL_TARGETS cufinufft)
endif()

# Add tests defined in their own directory
if(BUILD_TESTING AND FINUFFT_USE_CPU AND FINUFFT_BUILD_TESTS)
    add_subdirectory(test)
    add_subdirectory(perftest)
endif()

if(FINUFFT_BUILD_EXAMPLES AND FINUFFT_USE_CPU)
    add_subdirectory(examples)
endif()

if(FINUFFT_BUILD_EXAMPLES AND FINUFFT_USE_CUDA)
    add_subdirectory(examples/cuda)
endif()

if(FINUFFT_BUILD_FORTRAN)
    enable_language(Fortran)
    add_subdirectory(fortran)
endif()

if(FINUFFT_BUILD_MATLAB)
    add_subdirectory(matlab)
endif()

if(FINUFFT_BUILD_DEVEL)
    add_subdirectory(devel)
endif()

if(FINUFFT_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# gersemi: off
message(STATUS "FINUFFT configuration summary:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  FINUFFT_USE_CPU: ${FINUFFT_USE_CPU}")
message(STATUS "  FINUFFT_USE_CUDA: ${FINUFFT_USE_CUDA}")
message(STATUS "  FINUFFT_USE_OPENMP: ${FINUFFT_USE_OPENMP}")
message(STATUS "  FINUFFT_STATIC_LINKING: ${FINUFFT_STATIC_LINKING}")
message(STATUS "  FINUFFT_POSITION_INDEPENDENT_CODE: ${FINUFFT_POSITION_INDEPENDENT_CODE}")
message(STATUS "  FINUFFT_ENABLE_INSTALL: ${FINUFFT_ENABLE_INSTALL}")
message(STATUS "  FINUFFT_BUILD_EXAMPLES: ${FINUFFT_BUILD_EXAMPLES}")
message(STATUS "  FINUFFT_BUILD_TESTS: ${FINUFFT_BUILD_TESTS}")
message(STATUS "  FINUFFT_BUILD_FORTRAN: ${FINUFFT_BUILD_FORTRAN}")
message(STATUS "  FINUFFT_BUILD_MATLAB: ${FINUFFT_BUILD_MATLAB}")
message(STATUS "  FINUFFT_BUILD_PYTHON: ${FINUFFT_BUILD_PYTHON}")
message(STATUS "  FINUFFT_USE_SANITIZERS: ${FINUFFT_USE_SANITIZERS}")
message(STATUS "  FINUFFT_FFTW_SUFFIX: ${FINUFFT_FFTW_SUFFIX}")
message(STATUS "  FINUFFT_FFTW_LIBRARIES: ${FINUFFT_FFTW_LIBRARIES}")
message(STATUS "  FINUFFT_ARCH_FLAGS: ${FINUFFT_ARCH_FLAGS}")
message(STATUS "  FINUFFT_USE_DUCC0: ${FINUFFT_USE_DUCC0}")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
# gersemi: on

if(FINUFFT_ENABLE_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    install(TARGETS ${INSTALL_TARGETS} EXPORT finufftTargets PUBLIC_HEADER)
    install(EXPORT finufftTargets NAMESPACE finufft:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/finufft)
    configure_package_config_file(
        cmake/finufftConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/finufftConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/finufft
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/finufftConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/finufftConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/finufftConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/finufft
    )
    install(FILES ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/finufft)
    if(FINUFFT_USE_CPU)
        install(
            DIRECTORY ${PROJECT_SOURCE_DIR}/examples
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft
            PATTERN "CMakeLists.txt" EXCLUDE
            PATTERN "README" EXCLUDE
            PATTERN "examples/cuda" EXCLUDE
        )
        if(FINUFFT_BUILD_FORTRAN)
            install(
                DIRECTORY ${PROJECT_SOURCE_DIR}/fortran/examples
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/fortran
            )
            install(FILES ${PROJECT_SOURCE_DIR}/include/finufft.fh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        endif()
    endif()
    if(FINUFFT_USE_CUDA)
        install(
            DIRECTORY ${PROJECT_SOURCE_DIR}/examples/cuda
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/examples
            PATTERN "README" EXCLUDE
            PATTERN "CMakeLists.txt" EXCLUDE
        )
    endif()
endif()
