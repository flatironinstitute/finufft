# cmake/toolchain.cmake
include_guard(GLOBAL)

# Assumes cmake/utils.cmake has already been included by the top-level CMakeLists
# for: filter_supported_compiler_flags(), check_arch_support(), detect_cuda_architecture()

# ---- Install targets container ------------------------------------------------
# Keep this accessible globally
set(INSTALL_TARGETS "" CACHE INTERNAL "FINUFFT install targets list")

# ---- C++ flags (Release / Debug / RelWithDebInfo) ----------------------------
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
    -fno-semantic-interposition
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
    /Oy
)
filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_RELEASE FINUFFT_CXX_FLAGS_RELEASE)
message(STATUS "FINUFFT Release flags: ${FINUFFT_CXX_FLAGS_RELEASE}")
set(FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_RELEASE})

set(FINUFFT_CXX_FLAGS_DEBUG
    -g
    -g3
    -ggdb
    -ggdb3
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unknown-pragmas
    /W4
    /permissive-
    /wd4068
)
filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_DEBUG FINUFFT_CXX_FLAGS_DEBUG)
message(STATUS "FINUFFT Debug flags: ${FINUFFT_CXX_FLAGS_DEBUG}")

list(APPEND FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_RELEASE} ${FINUFFT_CXX_FLAGS_DEBUG})
message(STATUS "FINUFFT RelWithDebInfo flags: ${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}")

# ---- Architecture flags -------------------------------------------------------
if(FINUFFT_ARCH_FLAGS STREQUAL "native")
    set(FINUFFT_ARCH_FLAGS -march=native CACHE STRING "" FORCE)
    filter_supported_compiler_flags(FINUFFT_ARCH_FLAGS FINUFFT_ARCH_FLAGS)
    if(NOT FINUFFT_ARCH_FLAGS)
        set(FINUFFT_ARCH_FLAGS -mtune=native CACHE STRING "" FORCE)
        filter_supported_compiler_flags(FINUFFT_ARCH_FLAGS FINUFFT_ARCH_FLAGS)
    endif()
    if(MSVC)
        # -march=native emulation for MSVC
        check_msvc_arch_support()
    endif()
    if(NOT FINUFFT_ARCH_FLAGS)
        message(WARNING "No architecture flags are supported by the compiler.")
    else()
        message(STATUS "FINUFFT Arch flags: ${FINUFFT_ARCH_FLAGS}")
    endif()
endif()

# ---- Default build type -------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Set the default build type to Release" FORCE)
endif()

# ---- Precision-dependent sources ---------------------------------------------
set(FINUFFT_PRECISION_DEPENDENT_SOURCES)

# Fortran translation layer when enabled
if(FINUFFT_BUILD_FORTRAN)
    list(APPEND FINUFFT_PRECISION_DEPENDENT_SOURCES fortran/finufftfort.cpp)
endif()

# ---- Sanitizers ---------------------------------------------------------------
set(FINUFFT_SANITIZER_FLAGS)
if(FINUFFT_USE_SANITIZERS)
    set(FINUFFT_SANITIZER_FLAGS
        -fsanitize=address
        -fsanitize=undefined
        -fsanitize=bounds-strict
        /fsanitize=address
        /RTC1
    )
    filter_supported_compiler_flags(FINUFFT_SANITIZER_FLAGS FINUFFT_SANITIZER_FLAGS)
    set(FINUFFT_SANITIZER_FLAGS $<$<CONFIG:Debug,RelWithDebInfo>:${FINUFFT_SANITIZER_FLAGS}>)
endif()

# ---- Top-project features (CTest, Sphinx) ------------------------------------
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(FINUFFT_BUILD_TESTS)
        enable_testing()
    endif()
    if(FINUFFT_BUILD_DOCS)
        include(setupSphinx)
    endif()
endif()
