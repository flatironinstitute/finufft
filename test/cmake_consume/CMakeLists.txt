# Standalone downstream project: verifies an *installed* FINUFFT can be consumed
# via find_package(finufft) + target_link_libraries(finufft::finufft).
#
# This is intentionally NOT part of the main FINUFFT build. CI installs FINUFFT
# to a staging prefix and then configures this project against it with
#   -DCMAKE_PREFIX_PATH=<stage>   (or -Dfinufft_DIR=<stage>/lib/cmake/finufft)
#
# It is the regression guard for: missing transitive headers, missing
# find_dependency() in the config, and the Windows DLL export/runtime path.
cmake_minimum_required(VERSION 3.19)
project(finufft_consume LANGUAGES C CXX)

find_package(finufft REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
target_compile_features(app PRIVATE cxx_std_17)

# On Windows the finufft DLL must sit next to the executable at runtime.
if(WIN32 AND TARGET finufft::finufft)
    get_target_property(_finufft_type finufft::finufft TYPE)
    if(_finufft_type STREQUAL "SHARED_LIBRARY")
        add_custom_command(
            TARGET app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:finufft::finufft> $<TARGET_FILE_DIR:app>
        )
    endif()
endif()
