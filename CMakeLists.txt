cmake_minimum_required(VERSION 3.12.4)
project(finufft LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)

add_definitions(-DFFTW_PLAN_SAFE)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    add_compile_options(-fexceptions -fstack-protector-strong)
    add_compile_options(-Wall -Wextra -pedantic)
endif ()
if (CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    add_compile_options(-ffp-contract=fast -ffast-math -freciprocal-math -fmerge-all-constants)
    add_compile_options(-march=native -funroll-loops -finline-functions -ftree-vectorize)
endif ()
if (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    add_compile_options(-g -fno-omit-frame-pointer)
    add_compile_options(-ffp-contract=fast -ffast-math -freciprocal-math -fmerge-all-constants)
    add_compile_options(-march=native -funroll-loops -finline-functions -ftree-vectorize)
endif ()

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif ()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/contrib)

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
list(APPEND SOURCES ${CMAKE_SOURCE_DIR}/contrib/legendre_rule_fast.c)

find_package(FFTW REQUIRED FLOAT_THREADS_LIB FLOAT_THREADS_LIB)

message("FFTW_FOUND            ${FFTW_FOUND}")
message("FFTW_INCLUDE_DIRS     ${FFTW_INCLUDE_DIRS}")
message("FFTW_LIBRARIES        ${FFTW_LIBRARIES}")

list(APPEND LIBRARIES ${FFTW_LIBRARIES})
include_directories(${FFTW_INCLUDE_DIRS})

find_package(Threads REQUIRED)
list(APPEND LIBRARIES ${CMAKE_THREAD_LIBS_INIT})


add_library(finufft_static STATIC ${SOURCES})
target_link_libraries(finufft_static ${LIBRARIES})
set_target_properties(finufft_static PROPERTIES OUTPUT_NAME finufft)
install(TARGETS finufft_static DESTINATION ${CMAKE_SOURCE_DIR}/lib-static)


add_library(finufft_shared SHARED ${SOURCES})
target_link_libraries(finufft_shared ${LIBRARIES})
set_target_properties(finufft_shared PROPERTIES OUTPUT_NAME finufft)

install(TARGETS finufft_shared DESTINATION ${CMAKE_SOURCE_DIR}/lib)

list(APPEND LIBRARIES finufft_shared)

add_executable(basicpassfail test/basicpassfail.cpp)
target_link_libraries(basicpassfail ${LIBRARIES})

add_executable(dumbinputs test/dumbinputs.cpp)
target_link_libraries(dumbinputs ${LIBRARIES})

add_executable(finufft1d_test test/finufft1d_test.cpp)
target_link_libraries(finufft1d_test ${LIBRARIES})