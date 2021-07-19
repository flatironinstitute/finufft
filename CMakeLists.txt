cmake_minimum_required(VERSION 3.12.4)
project(finufft LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


add_definitions(-DFFTW_PLAN_SAFE)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    if (WIN32)
        add_compile_options(/Wall /WL /Z7 /Gs /Od /fp:strict)
    else ()
        add_compile_options(-fexceptions -fstack-protector-strong)
        add_compile_options(-Wall -Wextra -pedantic)
    endif ()
endif ()
if (CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    if (WIN32)
        add_compile_options(/O2 /fp:fast /arch:AVX2)
    else ()
        add_compile_options(-ffp-contract=fast -ffast-math -freciprocal-math -fmerge-all-constants)
        add_compile_options(-march=native -funroll-loops -finline-functions -ftree-vectorize)
    endif ()
endif ()
if (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    if (WIN32)
        add_compile_options(/O2 /fp:fast /Z7 /arch:AVX2 /arch:AVX /arch:AVX512)
    else ()
        add_compile_options(-g -fno-omit-frame-pointer)
        add_compile_options(-ffp-contract=fast -ffast-math -freciprocal-math -fmerge-all-constants)
        add_compile_options(-march=native -funroll-loops -finline-functions -ftree-vectorize)
    endif ()
endif ()

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif ()

if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif ()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/contrib)

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")


find_package(FFTW REQUIRED FLOAT_THREADS_LIB DOUBLE_THREADS_LIB)

message("FFTW_FOUND            ${FFTW_FOUND}")
message("FFTW_INCLUDE_DIRS     ${FFTW_INCLUDE_DIRS}")
message("FFTW_LIBRARIES        ${FFTW_LIBRARIES}")

list(APPEND LIBRARIES ${FFTW_LIBRARIES})
include_directories(${FFTW_INCLUDE_DIRS})

find_package(Threads REQUIRED)
list(APPEND LIBRARIES ${CMAKE_THREAD_LIBS_INIT})


add_library(finufft32 OBJECT src/finufft.cpp src/simpleinterfaces.cpp src/spreadinterp.cpp src/utils.cpp ${HEADERS})
set_target_properties(finufft32 PROPERTIES COMPILE_DEFINITIONS SINGLE)


list(APPEND SOURCES ${CMAKE_SOURCE_DIR}/contrib/legendre_rule_fast.c)
add_library(finufft64 OBJECT ${SOURCES} ${HEADERS})


add_library(finufft_static STATIC $<TARGET_OBJECTS:finufft32> $<TARGET_OBJECTS:finufft64> ${HEADERS})
target_link_libraries(finufft_static PUBLIC ${LIBRARIES})


add_library(finufft_shared SHARED $<TARGET_OBJECTS:finufft32> $<TARGET_OBJECTS:finufft64> ${HEADERS})
target_link_libraries(finufft_shared PUBLIC ${LIBRARIES})

install(TARGETS finufft_static DESTINATION ${CMAKE_SOURCE_DIR}/lib-static)
install(TARGETS finufft_shared DESTINATION ${CMAKE_SOURCE_DIR}/lib)

if (MSVC)
    include(GenerateExportHeader)
    generate_export_header(finufft_shared)
    generate_export_header(finufft_static)
    if (MSVC AND NOT (CMAKE_C_COMPILER_ID STREQUAL "Intel"))
        target_compile_definitions(finufft PRIVATE /bigobj)
    endif ()
    install(FILES ${PROJECT_BINARY_DIR}/finufft_shared_export.h DESTINATION ${CMAKE_SOURCE_DIR}/lib)
    install(FILES ${PROJECT_BINARY_DIR}/finufft_static_export.h DESTINATION ${CMAKE_SOURCE_DIR}/lib-static)
endif ()


set(LIBRARIES finufft32 finufft64)
message("TEST LIBRARIES: ${LIBRARIES}")

add_executable(basicpassfail test/basicpassfail.cpp)
target_link_libraries(basicpassfail finufft_static)

add_executable(dumbinputs test/dumbinputs.cpp)
target_link_libraries(dumbinputs finufft_static)

add_executable(finufft1d_test test/finufft1d_test.cpp)
target_link_libraries(finufft1d_test finufft_static)

add_executable(simple1d1f examples/simple1d1f.cpp)
target_link_libraries(simple1d1f finufft_static)