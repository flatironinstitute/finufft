cmake_minimum_required(VERSION 3.19...3.30)
project(
  FINUFFT
  VERSION 2.3.0
  LANGUAGES C CXX)

# windows MSVC runtime flags policy
cmake_policy(SET CMP0091 NEW)

include(CMakeDependentOption)

# cmake-format: off
# All options go here sphinx tag (don't remove): @cmake_opts_start
option(FINUFFT_BUILD_FORTRAN "Whether to build the FINUFFT Fortran examples" OFF)
option(FINUFFT_BUILD_MATLAB "Whether to build the FINUFFT Matlab interface" OFF)
option(FINUFFT_BUILD_PYTHON "Whether the Python wrapper should be built." OFF)
option(FINUFFT_ENABLE_SANITIZERS "Whether to enable sanitizers, only effective for Debug configuration." OFF)
option(FINUFFT_USE_OPENMP "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library." ON)
option(FINUFFT_USE_CPU "Whether to build the ordinary FINUFFT library (libfinufft)." ON)
option(FINUFFT_USE_CUDA "Whether to build CUDA accelerated FINUFFT library (libcufinufft). This is completely independent of the main FINUFFT library" OFF)
option(FINUFFT_STATIC_LINKING "If ON builds the static finufft library, if OFF build a shared finufft library." ON)
option(FINUFFT_BUILD_DEVEL "Whether to build development executables" OFF)
option(FINUFFT_BUILD_EXAMPLES "Whether to build the FINUFFT examples" OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_USE_DUCC0 "Whether to use DUCC0 (instead of FFTW) for CPU FFTs" OFF)
option(FINUFFT_BUILD_DOCS "Whether to build the FINUFFT documentation" OFF)
# if FINUFFT_USE_DUCC0 is ON, the following options are ignored
set(FINUFFT_FFTW_LIBRARIES "DEFAULT" CACHE STRING "Specify a custom FFTW library")
set(FINUFFT_FFTW_SUFFIX "OpenMP" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.)")
# if FINUFFT_USE_CUDA is OFF, the following options are ignored
set(FINUFFT_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for (e.g. 60;70;75;)")
# if FINUFFT_USE_CPU is OFF, the following options are ignored
set(FINUFFT_ARCH_FLAGS "native" CACHE STRING "Compiler flags for specifying target architecture, defaults to -march=native")
# sphinx tag (don't remove): @cmake_opts_end
cmake_dependent_option(FINUFFT_ENABLE_INSTALL "Disable installation in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
cmake_dependent_option(FINUFFT_STATIC_LINKING "Disable static libraries in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
cmake_dependent_option(FINUFFT_SHARED_LINKING "Shared should be the opposite of static linking" ON "NOT FINUFFT_STATIC_LINKING" OFF)
# cmake-format: on

include(cmake/utils.cmake)

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
    -O3
    /Ox
    /fp:contract
    /fp:except-
    /GF
    /GY
    /GS-
    /Ob
    /Oi
    /Ot
    /Oy)

filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_RELEASE
                                FINUFFT_CXX_FLAGS_RELEASE)
message(STATUS "FINUFFT Release flags: ${FINUFFT_CXX_FLAGS_RELEASE}")
set(FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_RELEASE})

set(FINUFFT_CXX_FLAGS_DEBUG
    -g
    -g3
    -ggdb
    -ggdb3
    /Zi
    -Wall
    -Wno-sign-compare
    -Wno-unknown-pragmas)
filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_DEBUG FINUFFT_CXX_FLAGS_DEBUG)
message(STATUS "FINUFFT Debug flags: ${FINUFFT_CXX_FLAGS_DEBUG}")
list(APPEND FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_RELEASE}
     ${FINUFFT_CXX_FLAGS_DEBUG})
message(
  STATUS "FINUFFT RelWithDebInfo flags: ${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}")

if(FINUFFT_ARCH_FLAGS STREQUAL "native")
  set(FINUFFT_ARCH_FLAGS
      -march=native
      CACHE STRING "" FORCE)
  filter_supported_compiler_flags(FINUFFT_ARCH_FLAGS FINUFFT_ARCH_FLAGS)
  if(NOT FINUFFT_ARCH_FLAGS)
    set(FINUFFT_ARCH_FLAGS
        -mtune=native
        CACHE STRING "" FORCE)
    filter_supported_compiler_flags(FINUFFT_ARCH_FLAGS FINUFFT_ARCH_FLAGS)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # -march=native emulation for MSVC
    check_arch_support()
  endif()
  if(NOT FINUFFT_ARCH_FLAGS)
    message(WARNING "No architecture flags are supported by the compiler.")
  else()
    message(STATUS "FINUFFT Arch flags: ${FINUFFT_ARCH_FLAGS}")
  endif()
endif()

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Set the default build type to Release" FORCE)
endif()

# This set of sources is compiled twice, once in single precision and once in
# double precision The single precision compilation is done with -DSINGLE
set(FINUFFT_PRECISION_DEPENDENT_SOURCES
    src/finufft.cpp src/fft.cpp src/simpleinterfaces.cpp src/spreadinterp.cpp
    src/utils.cpp)

# If we're building for Fortran, make sure we also include the translation
# layer.
if(FINUFFT_BUILD_FORTRAN)
  list(APPEND FINUFFT_PRECISION_DEPENDENT_SOURCES fortran/finufftfort.cpp)
endif()

# set linker flags for sanitizer
set(FINUFFT_SANITIZER_FLAGS)
if(FINUFFT_ENABLE_SANITIZERS)
  set(FINUFFT_SANITIZER_FLAGS -fsanitize=address -fsanitize=undefined
                              -fsanitize=bounds-strict /fsanitize=address /RTC1)
  filter_supported_compiler_flags(FINUFFT_SANITIZER_FLAGS
                                  FINUFFT_SANITIZER_FLAGS)
  set(FINUFFT_SANITIZER_FLAGS
      $<$<CONFIG:Debug,RelWithDebInfo>:${FINUFFT_SANITIZER_FLAGS}>)
endif()
# Utility function to enable ASAN on debug builds
function(enable_asan target)
  target_compile_options(${target} PRIVATE ${FINUFFT_SANITIZER_FLAGS})
  if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    target_link_options(${target} PRIVATE ${FINUFFT_SANITIZER_FLAGS})
  endif()
endfunction()

set(CPM_DOWNLOAD_VERSION 0.40.2)
include(cmake/setupCPM.cmake)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
  if(FINUFFT_BUILD_TESTS)
    enable_testing()
  endif()
  if(FINUFFT_BUILD_DOCS)
    include(cmake/setupSphinx.cmake)
  endif()
endif()

if(FINUFFT_USE_CPU)
  # make apple with gnu use old linker, new linker breaks, see issue #360
  if((APPLE) AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    add_link_options("-ld_classic")
  endif()
  set(FFTW_VERSION 3.3.10)
  set(XTL_VERSION 0.7.7)
  set(XSIMD_VERSION 13.0.0)
  # using latest ducc0 version for now as it fixes MacOS GCC build
  set(DUCC0_VERSION b0beb85e03982344a31ebb119758d7aa3ef5d362)
  set(FINUFFT_FFTW_LIBRARIES)
  include(cmake/setupXSIMD.cmake)
  if(FINUFFT_USE_DUCC0)
    include(cmake/setupDUCC.cmake)
  else()
    include(cmake/setupFFTW.cmake)
  endif()
  if(FINUFFT_USE_DUCC0)
    set(FINUFFT_FFTLIBS ducc0)
  else()
    set(FINUFFT_FFTLIBS ${FINUFFT_FFTW_LIBRARIES})
  endif()
  if(FINUFFT_USE_OPENMP)
    find_package(
      OpenMP
      COMPONENTS C CXX
      REQUIRED)
  endif()
endif()

# check if -Wno-deprecated-declarations is supported
check_cxx_compiler_flag(-Wno-deprecated-declarations
                        FINUFFT_HAS_NO_DEPRECATED_DECLARATIONS)

# Utility function to link static/dynamic lib
function(finufft_link_test target)
  if(FINUFFT_USE_DUCC0)
    target_compile_definitions(${target} PRIVATE FINUFFT_USE_DUCC0)
  endif()
  target_link_libraries(${target} PRIVATE finufft ${FINUFFT_FFTLIBS})
  if(FINUFFT_USE_OPENMP)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
  endif()
  enable_asan(${target})
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(
    ${target} PROPERTIES MSVC_RUNTIME_LIBRARY
                         "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  # disable deprecated warnings for tests if supported
  if(FINUFFT_HAS_NO_DEPRECATED_DECLARATIONS)
    target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
  endif()
endfunction()

# Utility function to set finufft compilation options.
function(set_finufft_options target)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(
    ${target} PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>)
  target_compile_options(
    ${target} PRIVATE $<$<CONFIG:Release>:${FINUFFT_CXX_FLAGS_RELEASE}>)
  target_compile_options(
    ${target}
    PRIVATE $<$<CONFIG:RelWithDebInfo>:${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}>)
  target_compile_options(${target}
                         PRIVATE $<$<CONFIG:Debug>:${FINUFFT_CXX_FLAGS_DEBUG}>)
  target_include_directories(
    ${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_include_directories(
    ${target} SYSTEM
    INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
  set_target_properties(
    ${target}
    PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
               POSITION_INDEPENDENT_CODE ${FINUFFT_SHARED_LINKING})
  enable_asan(${target})
  if(FINUFFT_USE_OPENMP)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
  endif()
  if(FINUFFT_USE_DUCC0)
    target_compile_definitions(${target} PRIVATE FINUFFT_USE_DUCC0)
  endif()
  target_link_libraries(${target} PRIVATE xsimd)
  target_link_libraries(${target} PRIVATE ${FINUFFT_FFTLIBS})
endfunction()

if(FINUFFT_USE_CPU)
  # Main finufft libraries
  add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
  target_compile_definitions(finufft_f32 PRIVATE SINGLE)
  set_finufft_options(finufft_f32)

  add_library(finufft_f64 OBJECT ${FINUFFT_PRECISION_DEPENDENT_SOURCES})
  set_finufft_options(finufft_f64)
  if(NOT FINUFFT_STATIC_LINKING)
    add_library(finufft SHARED src/utils_precindep.cpp
                               contrib/legendre_rule_fast.cpp)
  else()
    add_library(finufft STATIC src/utils_precindep.cpp
                               contrib/legendre_rule_fast.cpp)
  endif()
  target_link_libraries(finufft PRIVATE finufft_f32 finufft_f64)
  set_finufft_options(finufft)

  if(WIN32 AND FINUFFT_SHARED_LINKING)
    target_compile_definitions(finufft_f32 PRIVATE dll_EXPORTS FINUFFT_DLL)
    target_compile_definitions(finufft_f64 PRIVATE dll_EXPORTS FINUFFT_DLL)
    target_compile_definitions(finufft PRIVATE dll_EXPORTS FINUFFT_DLL)
  endif()
  find_library(MATH_LIBRARY m)
  if(MATH_LIBRARY)
    target_link_libraries(finufft PRIVATE ${MATH_LIBRARY})
  endif()
  target_include_directories(
    finufft PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_include_directories(
    finufft SYSTEM
    INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
  if(FINUFFT_ENABLE_INSTALL)
    file(GLOB FINUFFT_PUBLIC_HEADERS
         "${CMAKE_CURRENT_SOURCE_DIR}/include/finufft*.h")
    set_target_properties(finufft PROPERTIES PUBLIC_HEADER
                                             "${FINUFFT_PUBLIC_HEADERS}")
  endif()
  list(APPEND INSTALL_TARGETS finufft)
endif()

if(FINUFFT_USE_CUDA)
  if(NOT DEFINED FINUFFT_CUDA_ARCHITECTURES)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(FINUFFT_CUDA_ARCHITECTURES "{$CMAKE_CUDA_ARCHITECTURES}")
    else()
      message(
        "FINUFFT WARNING: No CUDA architecture supplied via '-DFINUFFT_CUDA_ARCHITECTURES=...', defaulting to 'native'"
      )
      message(
        "See: https://developer.nvidia.com/cuda-gpus for more details on what architecture to supply."
      )
    endif()
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(src/cuda)
  if(BUILD_TESTING AND FINUFFT_BUILD_TESTS)
    add_subdirectory(perftest/cuda)
    add_subdirectory(test/cuda)
  endif()

  list(APPEND INSTALL_TARGETS cufinufft)
endif()

# Add tests defined in their own directory
if(BUILD_TESTING
   AND FINUFFT_USE_CPU
   AND FINUFFT_BUILD_TESTS)
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
message(STATUS "  FINUFFT_USE_DUCC0: ${FINUFFT_USE_DUCC0}")

if(FINUFFT_ENABLE_INSTALL)
  include(GNUInstallDirs)
  install(TARGETS ${INSTALL_TARGETS} PUBLIC_HEADER)
  install(FILES ${PROJECT_SOURCE_DIR}/LICENSE
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/finufft)
  if(FINUFFT_USE_CPU)
    install(
      DIRECTORY ${PROJECT_SOURCE_DIR}/examples
      DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft
      PATTERN "CMakeLists.txt" EXCLUDE
      PATTERN "README" EXCLUDE
      PATTERN "examples/cuda" EXCLUDE)
    if(FINUFFT_BUILD_FORTRAN)
      install(DIRECTORY ${PROJECT_SOURCE_DIR}/fortran/examples
              DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/fortran)
      install(FILES ${PROJECT_SOURCE_DIR}/include/finufft.fh
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif()
  endif()
  if(FINUFFT_USE_CUDA)
    install(
      DIRECTORY ${PROJECT_SOURCE_DIR}/examples/cuda
      DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/examples
      PATTERN "README" EXCLUDE
      PATTERN "CMakeLists.txt" EXCLUDE)
  endif()
endif()
