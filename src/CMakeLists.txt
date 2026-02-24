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

# Sources compiled per-precision (once for float, once for double)
set(FINUFFT_PRECISION_SOURCES makeplan.cpp setpts.cpp execute.cpp)

# Sources compiled once (handle both precisions internally)
set(FINUFFT_COMMON_SOURCES spreadinterp.cpp fft.cpp c_interface.cpp finufft_utils.cpp)

if(FINUFFT_BUILD_FORTRAN)
    list(APPEND FINUFFT_COMMON_SOURCES ${PROJECT_SOURCE_DIR}/fortran/finufftfort.cpp)
endif()

# Helper: apply common compile settings to a target
function(finufft_apply_compile_settings target)
    target_link_libraries(${target} PRIVATE $<BUILD_INTERFACE:finufft_fftlibs xsimd finufft_common>)
    target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE ${FINUFFT_ARCH_FLAGS})
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:${FINUFFT_CXX_FLAGS_RELEASE}>)
    target_compile_options(${target} PRIVATE $<$<CONFIG:RelWithDebInfo>:${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}>)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:${FINUFFT_CXX_FLAGS_DEBUG}>)
    set_target_properties(
        ${target}
        PROPERTIES
            POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
            INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION}
    )
    if(FINUFFT_USE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        target_link_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
        target_include_directories(${target} PRIVATE $<BUILD_INTERFACE:${OpenMP_CXX_INCLUDE_DIR}>)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /bigobj)
    endif()
endfunction()

# Single-precision object library: compile precision sources with FINUFFT_SINGLE
add_library(finufft_f32 OBJECT ${FINUFFT_PRECISION_SOURCES})
target_compile_definitions(finufft_f32 PRIVATE FINUFFT_SINGLE)
finufft_apply_compile_settings(finufft_f32)

# Main library: double-precision precision sources + common sources + f32 objects
set(FINUFFT_SOURCES ${FINUFFT_PRECISION_SOURCES} ${FINUFFT_COMMON_SOURCES})

if(FINUFFT_STATIC_LINKING)
    add_library(finufft STATIC ${FINUFFT_SOURCES})
else()
    add_library(finufft SHARED ${FINUFFT_SOURCES})
endif()
target_sources(finufft PRIVATE $<TARGET_OBJECTS:finufft_f32>)

if(FINUFFT_BUILD_TESTS)
    target_compile_definitions(finufft PRIVATE FINUFFT_BUILD_TESTS)
    target_compile_definitions(finufft_f32 PRIVATE FINUFFT_BUILD_TESTS)
endif()

enable_static_analysis(finufft)
add_iwyu_fix_target(finufft)
add_library(finufft::finufft ALIAS finufft)

if(FINUFFT_USE_DUCC0)
    target_compile_definitions(finufft PRIVATE FINUFFT_USE_DUCC0)
    target_compile_definitions(finufft_f32 PRIVATE FINUFFT_USE_DUCC0)
endif()

finufft_apply_compile_settings(finufft)

if(FINUFFT_USE_OPENMP)
    if(NOT FINUFFT_STATIC_LINKING)
        target_link_libraries(finufft INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

if(FINUFFT_SHARED_LINKING)
    target_compile_definitions(finufft PRIVATE FINUFFT_DLL)
    target_compile_definitions(finufft_f32 PRIVATE FINUFFT_DLL)
    if(WIN32)
        target_compile_definitions(finufft PRIVATE dll_EXPORTS)
        target_compile_definitions(finufft_f32 PRIVATE dll_EXPORTS)
    endif()
endif()

set_target_properties(finufft PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
set_target_properties(finufft_f32 PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(finufft PRIVATE ${MATH_LIBRARY})
endif()

target_include_directories(finufft SYSTEM INTERFACE $<INSTALL_INTERFACE:include>)

enable_asan(finufft)

if(FINUFFT_ENABLE_INSTALL)
    file(GLOB FINUFFT_PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/include/finufft*.h")
    set_target_properties(finufft PROPERTIES PUBLIC_HEADER "${FINUFFT_PUBLIC_HEADERS}")
endif()

set(_targets ${INSTALL_TARGETS})
list(APPEND _targets finufft)
set(INSTALL_TARGETS "${_targets}" PARENT_SCOPE)
