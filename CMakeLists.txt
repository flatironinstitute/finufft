cmake_minimum_required(VERSION 3.25...4.4)
project(FINUFFT VERSION 2.6.0 LANGUAGES C CXX)

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
option(FINUFFT_BUILD_OCTAVE "Whether to build the FINUFFT Octave interface" OFF)
option(FINUFFT_BUILD_PYTHON "Whether the Python wrapper should be built." OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_INTERPROCEDURAL_OPTIMIZATION "Enable interprocedural optimization (LTO) if supported" OFF)
option(FINUFFT_POSITION_INDEPENDENT_CODE "Whether to build the finufft library with position independent code (-fPIC). This forced ON when FINUFFT_SHARED_LINKING is ON." ON)
option(FINUFFT_STATIC_ANALYSIS "Run clang-tidy/cppcheck" OFF)
option(FINUFFT_STATIC_ANALYSIS_WERROR "Treat clang-tidy/cppcheck findings as errors (intended for CI)." OFF)
option(FINUFFT_STATIC_LINKING "If ON builds the static finufft library, if OFF build a shared finufft library." ON)
option(FINUFFT_USE_CPU "Whether to build the ordinary FINUFFT library (libfinufft)." ON)
option(FINUFFT_USE_CUDA "Whether to build CUDA accelerated FINUFFT library (libcufinufft). This is completely independent of the main FINUFFT library" OFF)
option(FINUFFT_USE_DUCC0 "Whether to use DUCC0 (instead of FFTW) for CPU FFTs" OFF)
option(FINUFFT_USE_OPENMP "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library." ON)
option(
    FINUFFT_WARNINGS_AS_ERRORS
    "Treat compiler warnings in the finufft library sources as errors. Only Debug and RelWithDebInfo enable the warning flags, so Release builds are unaffected."
    ON
)
set(
    FINUFFT_USE_SANITIZERS
    "OFF"
    CACHE STRING
          "Sanitizer mode for Debug/RelWithDebInfo builds. Supported values: OFF, ON, MEMSAN, TSAN. ON and MEMSAN both select the default address/undefined/bounds bundle."
)
# if FINUFFT_USE_DUCC0 is ON, the following options are ignored
set(FINUFFT_FFTW_LIBRARIES "DEFAULT" CACHE STRING "Specify a custom FFTW library")
set(FINUFFT_FFTW_SUFFIX "DEFAULT" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.) defaults to empty string if OpenMP is disabled, else uses OpenMP. Ignored if DUCC0 is used.")
# if FINUFFT_USE_CPU is OFF, the following options are ignored
set(FINUFFT_ARCH_FLAGS "native" CACHE STRING "Compiler flags for specifying target architecture, defaults to -march=native")
# sphinx tag (don't remove): @cmake_opts_end
set_property(CACHE FINUFFT_USE_SANITIZERS PROPERTY STRINGS OFF ON MEMSAN TSAN)
cmake_dependent_option(FINUFFT_ENABLE_INSTALL "Disable installation in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
cmake_dependent_option(FINUFFT_STATIC_LINKING "Disable static libraries in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
cmake_dependent_option(FINUFFT_SHARED_LINKING "Shared should be the opposite of static linking" ON "NOT FINUFFT_STATIC_LINKING" OFF)
cmake_dependent_option(FINUFFT_INTERPROCEDURAL_OPTIMIZATION "LTO should be the opposite of static linking" ON "NOT FINUFFT_STATIC_LINKING" OFF)
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
set(CPM_DOWNLOAD_VERSION "0.43.1" CACHE STRING "Version of CPM.cmake to use")
set(FFTW_VERSION "3.3.10" CACHE STRING "Version of FFTW to use")
# Four commits past 14.3.0, for xsimd PR #1387. Without it any Debug build on an
# AVX2 host fails: _mm_permute_ps in xsimd_avx_128.hpp gets a non-constant
# immediate at -O0. Move to the next xsimd release once it ships that fix.
set(XSIMD_VERSION "c71a9b23258c613b6f61ae9857342dc7b774dcab" CACHE STRING "Version of xsimd to use")
set(POET_VERSION "v0.0.1" CACHE STRING "Version of POET to use")
set(DUCC0_VERSION "ducc0_0_41_1" CACHE STRING "Version of ducc0 to use")
set(CUDA11_CCCL_VERSION "2.8.5" CACHE STRING "Version of FINUFFT-cccl for cuda 11 to use")
set(CUDA12_CCCL_VERSION "3.4.2" CACHE STRING "Version of FINUFFT-cccl for cuda 12 to use")

mark_as_advanced(
    CPM_DOWNLOAD_VERSION
    FFTW_VERSION
    XTL_VERSION
    XSIMD_VERSION
    POET_VERSION
    DUCC0_VERSION
    CUDA11_CCCL_VERSION
    CUDA12_CCCL_VERSION
)

include(setupCPM)

# Sphinx documentation targets; setupSphinx needs CPM, so it cannot live in
# toolchain.cmake
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND FINUFFT_BUILD_DOCS)
    include(setupSphinx)
endif()

# POET header-only dispatcher, used by both CPU and CUDA spread/interp paths.
include(setupPOET)

if(FINUFFT_USE_CPU)
    if(FINUFFT_USE_OPENMP)
        find_package(OpenMP COMPONENTS C CXX REQUIRED)
    endif()
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

if(FINUFFT_BUILD_MATLAB OR FINUFFT_BUILD_OCTAVE)
    add_subdirectory(matlab)
endif()

if(FINUFFT_BUILD_DEVEL)
    add_subdirectory(devel)
endif()

if(FINUFFT_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Regenerate the checked-in generated docs in the source tree - the same thing
# `make docs` does, for people who never touch the makefile. Never edit
# docs/*.doc by hand: they are marked linguist-generated in .gitattributes, and
# addmhelp.sh also rewrites the help blocks in matlab/*.m from
# matlab/opts.docbit. The generators are bash, hence UNIX only; docs/makefile.doc
# is the makefile's own help text, so it stays a `make docs` job. This produces
# sphinx's inputs; finufft_sphinx (FINUFFT_BUILD_DOCS) renders them to HTML.
if(UNIX)
    add_custom_target(
        finufft_regen_docs
        COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_SOURCE_DIR}/matlab ./addmhelp.sh
        COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_SOURCE_DIR}/docs ./genmatlabhelp.sh
        COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_SOURCE_DIR}/docs ./makecdocs.sh
        COMMENT "Regenerating matlab/*.m help blocks and docs/*.doc"
        VERBATIM
    )
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
message(STATUS "  FINUFFT_BUILD_OCTAVE: ${FINUFFT_BUILD_OCTAVE}")
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
    install(
        TARGETS ${INSTALL_TARGETS}
        EXPORT finufftTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # .dll on Windows
        FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    # Static builds expose finufft's private FFT backend on the install interface
    # (see src/CMakeLists.txt), so the backend must be part of the export. DUCC0
    # and a downloaded FFTW are archives FINUFFT builds itself, so they ship
    # inside finufftTargets; a system FFTW is found again by the consumer through
    # the FindFFTW module installed beside the package config. A shared build
    # bakes the backend in and exports nothing here.
    if(FINUFFT_STATIC_LINKING AND (FINUFFT_FFT_EXPORT_TARGETS OR FINUFFT_FFT_FIND_MODULE))
        install(
            TARGETS finufft_fftlibs ${FINUFFT_FFT_EXPORT_TARGETS}
            EXPORT finufftTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
        if(FINUFFT_FFT_FIND_MODULE)
            install(FILES ${FINUFFT_FFT_FIND_MODULE} DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/finufft)
        endif()
    endif()
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
