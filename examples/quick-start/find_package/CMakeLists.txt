cmake_minimum_required(VERSION 3.23)
project(find_package_consumer LANGUAGES CXX)

# docs-start: find_package
find_package(finufft REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
# docs-end: find_package
