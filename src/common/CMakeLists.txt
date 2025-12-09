cmake_minimum_required(VERSION 3.24)

# Build a small static helper library for CPU-only kernel parameter helpers
# Source is in ../kernel.cpp as requested
set(FINUFFT_COMMON_SOURCES kernel.cpp utils.cpp)

add_library(finufft_common STATIC ${FINUFFT_COMMON_SOURCES})
add_library(finufft::common ALIAS finufft_common)

# The public include directory is the top-level include/
target_include_directories(
    finufft_common
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)

# Visibility and compile options consistent with finufft
if(FINUFFT_SHARED_LINKING)
    target_compile_definitions(finufft_common PRIVATE FINUFFT_DLL)
endif()

target_compile_features(finufft_common PRIVATE cxx_std_17)

# If you need -lm etc.
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(finufft_common PRIVATE ${MATH_LIBRARY})
endif()

set_target_properties(finufft_common PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
