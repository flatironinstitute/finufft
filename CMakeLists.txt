cmake_minimum_required(VERSION 3.23)

project(finufft VERSION 2.2.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GNU_LIKE_FRONTENDS AppleClang Clang GNU)
if (CMAKE_CXX_COMPILER_ID IN_LIST GNU_LIKE_FRONTENDS)
    # Set custom compiler flags for gcc-compatible compilers
    set(FINUFFT_CXX_FLAGS_RELEASE -O3 -funroll-loops -ffp-contract=fast)
    set(FINUFFT_CXX_FLAGS_RELWITHDEBINFO -g ${FINUFFT_CXX_FLAGS_RELEASE})
endif ()

include(CTest)

if (CMAKE_CXX_COMPILER_ID IN_LIST GNU_LIKE_FRONTENDS)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|powerpc|powerpc64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "ppc|ppc64"))
        # PowerPC arch does not have -march flag.
        set(FINUFFT_ARCH_FLAGS "-mtune=native" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(FINUFFT_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
endif ()
set(FINUFFT_FFTW_SUFFIX "OpenMP" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.)")
set(FINUFFT_FFTW_LIBRARIES "DEFAULT" CACHE STRING "Specify a custom FFTW library")
set(FINUFFT_CUDA_ARCHITECTURES "all-major" CACHE STRING "CUDA architectures to build for (e.g. 60;70;75;)")
# All options go here
# sphinx tag (don't remove): @cmake_opts_start
option(FINUFFT_BUILD_EXAMPLES "Whether to build the FINUFFT examples" OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_BUILD_FORTRAN "Whether to build the FINUFFT Fortran examples" OFF)
option(FINUFFT_BUILD_MATLAB "Whether to build the FINUFFT Matlab interface" OFF)
option(FINUFFT_ENABLE_SANITIZERS "Whether to enable sanitizers, only effective for Debug configuration." ON)
option(FINUFFT_USE_OPENMP "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library." ON)
option(FINUFFT_USE_CUDA "Whether to build CUDA accelerated FINUFFT library (libcufinufft). This is completely independent of the main FINUFFT library" OFF)
option(FINUFFT_USE_CPU "Whether to build the ordinary FINUFFT library (libfinufft)." ON)
option(FINUFFT_STATIC_LINKING "Whether to link the static FINUFFT library (libfinufft_static)." ON)
option(FINUFFT_BUILD_DEVEL "Whether to build development executables" OFF)
# sphinx tag (don't remove): @cmake_opts_end

if (FINUFFT_USE_CPU)
    # suppress Windows warnings about "unsafe" functions
    if (WIN32)
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    endif ()

    # make apple with gnu use old linker, new linker breaks, see issue #360
    if ((APPLE) AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        add_link_options("-ld64")
    endif ()

    set(CPM_DOWNLOAD_VERSION 0.38.0)
    set(FFTW_VERSION 3.3.10)
    set(XTL_VERSION 0.7.7)
    set(XSIMD_VERSION 13.0.0)
    include(cmake/setupCPM.cmake)
    include(cmake/setupFFTW.cmake)
    include(cmake/setupXSIMD.cmake)
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
        find_package(OpenMP REQUIRED)
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
        target_compile_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=address -fsanitize=undefined -fsanitize=bounds-strict>)
        target_link_options(${target} PRIVATE $<$<CONFIG:DEBUG>:-fsanitize=address -fsanitize=undefined -fsanitize=bounds-strict>)
    endif ()
endfunction()

# Utility function to link static/dynamic lib
function(finufft_link_test target)
    if (FINUFFT_STATIC_LINKING)
        target_link_libraries(${target} PRIVATE finufft_static)
        if (FINUFFT_USE_OPENMP)
            target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
            if (WIN32)
                target_link_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
            endif ()
        endif ()
    else ()
        target_link_libraries(${target} PRIVATE finufft)
        if (WIN32)
            target_compile_definitions(${target} PRIVATE FINUFFT_DLL)
        endif ()
    endif ()
    enable_asan(${target})
endfunction()

# Utility function to set finufft compilation options.
function(set_finufft_options target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_compile_features(${target} PRIVATE cxx_std_17)

    enable_asan(${target})

    target_compile_options(${target} PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Enable cx-limited-range on supported compilers
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-fcx-limited-range>)
    endif ()
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:${FINUFFT_CXX_FLAGS_RELEASE}>)
    target_compile_options(${target} PRIVATE $<$<CONFIG:RelWithDebInfo>:${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}>)
    target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_include_directories(${target} SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
    if (FINUFFT_USE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        # there are issues on windows with OpenMP and CMake, so we need to manually add the flags
        # otherwise there are link errors
        if (WIN32)
            target_link_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
        endif ()
    else ()
        if (CMAKE_CXX_COMPILER_ID IN_LIST FINUFFT_GNU_LIKE_COMPILERS)
            # OpenMP disabled, suppress unknown pragma warnings to avoid spam.
            target_compile_options(${target} PRIVATE "-Wno-unknown-pragmas")
        endif ()
    endif ()

    # FFTW CMAKE file includes the APIs only as an install target, so we need to manually
    # include them since we need them for build not for install
    # trying to include them directly into the fftw and fftwf targets causes issues with
    # the latest version of cmake, so we do it here instead.
    if ((NOT FFTW_FOUND) OR (FINUFFT_FFTW_LIBRARIES STREQUAL DOWNLOAD))
        list(GET FINUFFT_FFTW_LIBRARIES 0 element)
        get_property(FFTW_SOURCE_DIR TARGET ${element} PROPERTY SOURCE_DIR)
        set(FFTW_INCLUDE_DIR ${FFTW_SOURCE_DIR}/api)
        target_include_directories(${target} PUBLIC ${FFTW_INCLUDE_DIR})
    endif ()
    target_link_libraries(${target} PRIVATE xsimd)
    target_link_libraries(${target} PUBLIC ${FINUFFT_FFTW_LIBRARIES})
endfunction()

if (FINUFFT_USE_CPU)
    # Main finufft libraries
    add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
    target_compile_definitions(finufft_f32 PRIVATE SINGLE)
    set_finufft_options(finufft_f32)

    add_library(finufft_f64 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
    target_compile_definitions(finufft_f64 PRIVATE)
    set_finufft_options(finufft_f64)
    if (WIN32)
        add_library(finufft_f32_dll OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
        target_compile_definitions(finufft_f32_dll PRIVATE SINGLE dll_EXPORTS FINUFFT_DLL)
        set_finufft_options(finufft_f32_dll)

        add_library(finufft_f64_dll OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
        target_compile_definitions(finufft_f64_dll PRIVATE dll_EXPORTS FINUFFT_DLL)
        set_finufft_options(finufft_f64_dll)
    endif ()

    add_library(finufft SHARED src/utils_precindep.cpp contrib/legendre_rule_fast.cpp)
    target_compile_definitions(finufft PRIVATE dll_EXPORTS FINUFFT_DLL)
    set_finufft_options(finufft)
    if (NOT WIN32)
        target_link_libraries(finufft PUBLIC finufft_f32 finufft_f64)
    else ()
        target_link_libraries(finufft PUBLIC finufft_f32_dll finufft_f64_dll)
    endif ()
    # windows does not have a math library, so we need to exclude it
    if (NOT WIN32)
        target_link_libraries(finufft PUBLIC m)
    endif ()
    target_include_directories(finufft PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_include_directories(finufft SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

    add_library(finufft_static STATIC src/utils_precindep.cpp contrib/legendre_rule_fast.cpp)
    set_finufft_options(finufft_static)
    target_link_libraries(finufft_static PUBLIC finufft_f32 finufft_f64)
    # windows does not have a math library, so we need to exclude it
    if (NOT WIN32)
        target_link_libraries(finufft_static PUBLIC m)
    endif ()
    target_include_directories(finufft_static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_include_directories(finufft_static SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

    file(GLOB FINUFFT_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/finufft*.h")
    set_target_properties(finufft PROPERTIES PUBLIC_HEADER "${FINUFFT_PUBLIC_HEADERS}")

    list(APPEND INSTALL_TARGETS finufft finufft_static)
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
    list(APPEND INSTALL_TARGETS cufinufft cufinufft_static)
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
    endif ()
endif ()
if (FINUFFT_USE_CUDA)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/examples/cuda
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/examples
            PATTERN "README" EXCLUDE
            PATTERN "CMakeLists.txt" EXCLUDE
    )
endif ()
