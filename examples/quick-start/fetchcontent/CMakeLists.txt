cmake_minimum_required(VERSION 3.23)
project(finufft_consume_fetchcontent LANGUAGES CXX)

# docs-start: fetchcontent
include(FetchContent)
FetchContent_Declare(finufft GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git GIT_TAG v2.5.1)
FetchContent_MakeAvailable(finufft)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
# docs-end: fetchcontent
