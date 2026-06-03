cmake_minimum_required(VERSION 3.24)

# Build a small static helper library for CPU-only kernel parameter helpers
# Source is in ../kernel.cpp as requested
set(FINUFFT_COMMON_SOURCES kernel.cpp utils.cpp pswf.cpp)

add_library(finufft_common STATIC ${FINUFFT_COMMON_SOURCES})

# The public include directory is the top-level include/
target_include_directories(
    finufft_common
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
set_target_properties(finufft_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Visibility and compile options consistent with finufft
if(FINUFFT_SHARED_LINKING)
    target_compile_definitions(finufft_common PRIVATE FINUFFT_DLL)
    if(WIN32)
        # Mirror src/cuda/CMakeLists.txt: dll_EXPORTS selects __declspec(dllexport)
        # so FINUFFT_EXPORT symbols compiled here are exported, not imported.
        target_compile_definitions(finufft_common PRIVATE dll_EXPORTS)
    endif()
endif()

target_compile_features(finufft_common PRIVATE cxx_std_17)

# If you need -lm etc. Wrap in BUILD_INTERFACE so the absolute libm path is not
# propagated into the installed package interface (finufft_common is a static
# archive and is exported for static consumers).
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(finufft_common PRIVATE $<BUILD_INTERFACE:${MATH_LIBRARY}>)
endif()

set_target_properties(finufft_common PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

# finufft links finufft_common privately. For a SHARED build its objects are
# baked into the library, but a STATIC finufft.a leaves them as unresolved
# references, so a static consumer must also link finufft_common. Export it
# (handled in the top-level install block) and propagate it on the install
# interface in that case.
if(FINUFFT_ENABLE_INSTALL AND FINUFFT_STATIC_LINKING)
    set(_targets ${INSTALL_TARGETS})
    list(APPEND _targets finufft_common)
    set(INSTALL_TARGETS "${_targets}" PARENT_SCOPE)
endif()
