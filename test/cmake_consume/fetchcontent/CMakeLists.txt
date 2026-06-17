# Second consume variant: pull FINUFFT in via FetchContent (the add_subdirectory
# path) instead of an installed find_package. CI points FINUFFT_SOURCE_DIR at the
# checked-out repo so no network access is needed; drop that define to fetch from
# GitHub instead.
cmake_minimum_required(VERSION 3.25)
project(finufft_consume_fetchcontent LANGUAGES C CXX)

set(FINUFFT_SOURCE_DIR "" CACHE PATH "Local FINUFFT checkout to consume")

include(FetchContent)
if(FINUFFT_SOURCE_DIR)
    FetchContent_Declare(finufft SOURCE_DIR ${FINUFFT_SOURCE_DIR})
else()
    FetchContent_Declare(finufft GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git GIT_TAG master)
endif()
FetchContent_MakeAvailable(finufft)

add_executable(app ../main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
target_compile_features(app PRIVATE cxx_std_17)
