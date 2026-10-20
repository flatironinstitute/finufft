find_package(Threads REQUIRED)
file(GLOB example_src "*.cu" "*.c")

# MSVC's C compiler has no C99 complex, so skip the C example there (same check as examples/CMakeLists.txt)
include(CheckCSourceCompiles)
# gersemi: off
check_c_source_compiles("
        #include <complex.h>
        int main() {
            float complex z = 1.0f + 2.0f*I;
            float x = crealf(z);
            return 0;
        }
    " HAS_C_COMPLEX)
# gersemi: on
if(NOT HAS_C_COMPLEX)
    list(FILTER example_src EXCLUDE REGEX "\\.c$")
endif()

# colliding names get a cuda_ prefix
foreach(srcfile ${example_src})
    get_filename_component(name ${srcfile} NAME_WE)
    set(executable ${name})
    if(TARGET ${executable})
        set(executable cuda_${name})
    endif()
    add_executable(${executable} ${srcfile})
    set_target_properties(${executable} PROPERTIES OUTPUT_NAME ${name})
    target_include_directories(${executable} PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
    target_link_libraries(${executable} PRIVATE cufinufft CUDA::cufft CUDA::cudart $<BUILD_INTERFACE:finufft_common>)
    if(srcfile MATCHES "\.cu$")
        target_compile_features(${executable} PRIVATE cxx_std_17 cuda_std_17)
    else()
        set_target_properties(${executable} PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
    endif()
    if(name MATCHES "^threadsafe")
        target_link_libraries(${executable} PRIVATE Threads::Threads)
    endif()
    add_test(NAME example_${executable} COMMAND ${executable})
endforeach()
