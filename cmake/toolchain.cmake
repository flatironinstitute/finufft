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
# Symbol interposition is an ELF concept. clang accepts the flag on Mach-O and COFF, then
# warns that it went unused, which the try_compile filter below cannot see and warnings as
# errors turns into a build failure.
if(NOT APPLE AND NOT WIN32)
    list(APPEND FINUFFT_CXX_FLAGS_RELEASE -fno-semantic-interposition)
endif()
filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_RELEASE FINUFFT_CXX_FLAGS_RELEASE)
message(STATUS "FINUFFT Release flags: ${FINUFFT_CXX_FLAGS_RELEASE}")
set(FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_RELEASE})

set(FINUFFT_CXX_FLAGS_DEBUG
    -g
    -g3
    -ggdb
    -ggdb3
    -Wextra
    -Wpedantic
    -Wno-unknown-pragmas
    /W4
    /permissive-
    /wd4068
)
# cl.exe accepts -Wall as a synonym for /Wall (every warning, including the purely
# informational C4710/C4820/C4514), so the try_compile filter below cannot reject it.
# /W4 above is the MSVC equivalent of -Wall -Wextra.
if(NOT MSVC)
    list(APPEND FINUFFT_CXX_FLAGS_DEBUG -Wall)
endif()
filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_DEBUG FINUFFT_CXX_FLAGS_DEBUG)
message(STATUS "FINUFFT Debug flags: ${FINUFFT_CXX_FLAGS_DEBUG}")

# MSVC reports what GCC and clang leave to opt-in flags this project does not ask for.
# Every configuration carries these: /W3 is CMake's MSVC default, so Release warns too,
# whereas the /W4 above is Debug and RelWithDebInfo only.
set(FINUFFT_CXX_FLAGS_WARNINGS
    /wd4244 # narrowing conversion, i.e. -Wconversion
    /wd4267 # size_t narrowing, i.e. -Wconversion
    /wd4305 # truncation to a narrower floating type, i.e. -Wconversion
    /wd4324 # padding inserted for an alignas member, i.e. -Wpadded
    /wd4456 # a local hides an outer local, i.e. -Wshadow
    /wd4458 # a local hides a class member, i.e. -Wshadow
    /wd4702 # unreachable code, i.e. -Wunreachable-code
    /wd4849 # MSVC implements OpenMP 2.0, which has no collapse clause
)
# clang 18 reports a lone -fcx-limited-range as overriding the empty option it compares
# against; clang 19 fixed that comparison. GCC accepts the unknown -Wno- silently, which
# then annotates every later diagnostic, so ask for it on clang alone.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND FINUFFT_CXX_FLAGS_WARNINGS -Wno-overriding-option)
endif()
filter_supported_compiler_flags(FINUFFT_CXX_FLAGS_WARNINGS FINUFFT_CXX_FLAGS_WARNINGS)
message(STATUS "FINUFFT warning flags: ${FINUFFT_CXX_FLAGS_WARNINGS}")

list(APPEND FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_DEBUG})
message(STATUS "FINUFFT RelWithDebInfo flags: ${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}")

# Microsoft's CRT deprecates portable C (sscanf, getenv, ...) in favour of its _s
# variants. Applies to any compiler using those headers, MSVC and clang alike.
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

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
# -O1, not Debug's -O0: an instrumented build is slow enough already.
set(FINUFFT_SANITIZER_FLAGS)
string(TOUPPER "${FINUFFT_USE_SANITIZERS}" FINUFFT_USE_SANITIZERS_MODE)
if(FINUFFT_USE_SANITIZERS_MODE STREQUAL "OFF")
elseif(FINUFFT_USE_SANITIZERS_MODE STREQUAL "ON" OR FINUFFT_USE_SANITIZERS_MODE STREQUAL "MEMSAN")
    set(FINUFFT_SANITIZER_FLAGS
        -fsanitize=address
        -fsanitize=undefined
        -fsanitize=bounds-strict
        -O1
        -fno-omit-frame-pointer
        /fsanitize=address
        /RTC1
    )
elseif(FINUFFT_USE_SANITIZERS_MODE STREQUAL "TSAN")
    set(FINUFFT_SANITIZER_FLAGS -fsanitize=thread -O1 -fno-omit-frame-pointer)
else()
    message(
        FATAL_ERROR
        "Unsupported FINUFFT_USE_SANITIZERS value '${FINUFFT_USE_SANITIZERS}'. Use one of: OFF, ON, MEMSAN, TSAN."
    )
endif()

if(FINUFFT_SANITIZER_FLAGS)
    filter_supported_compiler_flags(FINUFFT_SANITIZER_FLAGS FINUFFT_SANITIZER_FLAGS)
    set(FINUFFT_SANITIZER_FLAGS $<$<CONFIG:Debug,RelWithDebInfo>:${FINUFFT_SANITIZER_FLAGS}>)
endif()

# ---- Top-project features (CTest) --------------------------------------------
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(FINUFFT_BUILD_TESTS)
        enable_testing()
    endif()
endif()
