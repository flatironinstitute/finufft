include(staticAnalysis)
include(setupIWYU)

if((APPLE) AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    add_link_options("-ld_classic")
endif()

set(FINUFFT_FFTW_LIBRARIES)
include(setupXSIMD)
if(FINUFFT_USE_DUCC0)
    include(setupDUCC)
else()
    include(setupFFTW)
endif()

if(FINUFFT_USE_OPENMP)
    find_package(OpenMP COMPONENTS C CXX REQUIRED)
endif()

set(FINUFFT_SOURCES
    spreadinterp.cpp
    fft.cpp
    finufft_core.cpp
    c_interface.cpp
    finufft_utils.cpp
    utils.cpp
)

if(FINUFFT_BUILD_FORTRAN)
    list(APPEND FINUFFT_SOURCES ${PROJECT_SOURCE_DIR}/fortran/finufftfort.cpp)
endif()

if(FINUFFT_STATIC_LINKING)
    add_library(finufft STATIC ${FINUFFT_SOURCES})
else()
    add_library(finufft SHARED ${FINUFFT_SOURCES})
endif()

if(FINUFFT_BUILD_TESTS)
    target_compile_definitions(finufft PRIVATE FINUFFT_BUILD_TESTS)
endif()

enable_static_analysis(finufft)
add_iwyu_fix_target(finufft)
add_library(finufft::finufft ALIAS finufft)

if(FINUFFT_USE_DUCC0)
    target_compile_definitions(finufft PRIVATE FINUFFT_USE_DUCC0)
endif()

target_link_libraries(finufft PRIVATE $<BUILD_INTERFACE:finufft_fftlibs xsimd>)
if(FINUFFT_USE_OPENMP)
    target_link_libraries(finufft PRIVATE OpenMP::OpenMP_CXX)
    if(NOT FINUFFT_STATIC_LINKING)
        target_link_libraries(finufft INTERFACE OpenMP::OpenMP_CXX)
    endif()
    target_link_options(finufft PRIVATE ${OpenMP_CXX_FLAGS})
    target_include_directories(finufft PRIVATE $<BUILD_INTERFACE:${OpenMP_CXX_INCLUDE_DIR}>)
endif()

if(FINUFFT_SHARED_LINKING)
    target_compile_definitions(finufft PRIVATE FINUFFT_DLL)
    if(WIN32)
        target_compile_definitions(finufft PRIVATE dll_EXPORTS)
    endif()
endif()

set_target_properties(finufft PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(finufft PRIVATE ${MATH_LIBRARY})
endif()

target_include_directories(finufft PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_compile_features(finufft PRIVATE cxx_std_17)
target_compile_options(finufft PRIVATE ${FINUFFT_ARCH_FLAGS})
target_compile_options(finufft PRIVATE $<$<CONFIG:Release>:${FINUFFT_CXX_FLAGS_RELEASE}>)
target_compile_options(finufft PRIVATE $<$<CONFIG:RelWithDebInfo>:${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}>)
target_compile_options(finufft PRIVATE $<$<CONFIG:Debug>:${FINUFFT_CXX_FLAGS_DEBUG}>)
target_include_directories(finufft PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_include_directories(finufft SYSTEM INTERFACE $<INSTALL_INTERFACE:include>)
set_target_properties(
    finufft
    PROPERTIES
        POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
        INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION}
)
enable_asan(finufft)

if(FINUFFT_ENABLE_INSTALL)
    file(GLOB FINUFFT_PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/include/finufft*.h")
    set_target_properties(finufft PROPERTIES PUBLIC_HEADER "${FINUFFT_PUBLIC_HEADERS}")
endif()

set(_targets ${INSTALL_TARGETS})
list(APPEND _targets finufft)
set(INSTALL_TARGETS "${_targets}" PARENT_SCOPE)
