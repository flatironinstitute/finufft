# Second GPU consumption route: pull FINUFFT in via FetchContent (the
# add_subdirectory path) instead of an installed find_package. CI points
# FINUFFT_SOURCE_DIR at the checked-out repo so no network access is needed;
# drop that define to fetch from GitHub instead.
cmake_minimum_required(VERSION 3.25)
# CXX only: FINUFFT runs enable_language(CUDA) itself when FINUFFT_USE_CUDA is
# ON, so the consumer never has to.
project(cufinufft_consume_fetchcontent LANGUAGES CXX)

set(FINUFFT_SOURCE_DIR "" CACHE PATH "Local FINUFFT checkout to consume")
set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)

include(FetchContent)
if(FINUFFT_SOURCE_DIR)
    FetchContent_Declare(finufft SOURCE_DIR ${FINUFFT_SOURCE_DIR})
else()
    FetchContent_Declare(finufft GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git GIT_TAG master)
endif()
FetchContent_MakeAvailable(finufft)

add_executable(app ../main.cpp)
target_link_libraries(app PRIVATE finufft::cufinufft)
target_compile_features(app PRIVATE cxx_std_17)
