# include cpm cmake, downloading it
CPMAddPackage(
    NAME
    benchmark
    GITHUB_REPOSITORY
    google/benchmark
    VERSION
    1.8.3
    OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
    "BENCHMARK_ENABLE_WERROR OFF"
)

if(benchmark_ADDED)
    # patch benchmark target
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 17)
    # benchmark compiles itself with -pedantic-errors, so any new pedantic
    # diagnostic in its own headers breaks our build (clang 22: __COUNTER__ is
    # flagged as a C2y extension). Not our code, not our warning policy.
    if(NOT MSVC)
        target_compile_options(benchmark PRIVATE -Wno-pedantic)
        target_compile_options(benchmark_main PRIVATE -Wno-pedantic)
    endif()
endif()

# These link exactly like test/ and perftest/ do: finufft_link_test() carries the
# private includes (xsimd, poet), the DUCC0 define and the sanitizer flags.
add_executable(foldrescale foldrescale.cpp)
finufft_link_test(foldrescale)
target_link_libraries(foldrescale PRIVATE benchmark)

add_executable(padding padding.cpp)
finufft_link_test(padding)
target_link_libraries(padding PRIVATE xsimd)
target_compile_options(padding PRIVATE ${FINUFFT_ARCH_FLAGS})

add_executable(guru_benchmark guru_benchmark.cpp)
finufft_link_test(guru_benchmark)
target_link_libraries(guru_benchmark PRIVATE benchmark)

# reports the detected thread count for the current affinity mask; run it under taskset
add_executable(thread_count_affinity thread_count_affinity.cpp)
target_compile_features(thread_count_affinity PRIVATE cxx_std_17)
target_link_libraries(thread_count_affinity finufft)

add_executable(binsort_gbench binsort_bench.cpp)
finufft_link_test(binsort_gbench)
target_link_libraries(binsort_gbench PRIVATE benchmark xsimd)
target_compile_options(binsort_gbench PRIVATE ${FINUFFT_ARCH_FLAGS})
add_executable(binsort_gbenchf binsort_bench.cpp)
target_compile_definitions(binsort_gbenchf PRIVATE -DSINGLE)
finufft_link_test(binsort_gbenchf)
target_link_libraries(binsort_gbenchf PRIVATE benchmark xsimd)
target_compile_options(binsort_gbenchf PRIVATE ${FINUFFT_ARCH_FLAGS})

include(CheckIncludeFile)
check_include_file("getopt.h" HAVE_GETOPT_H)
if(HAVE_GETOPT_H)
    CPMAddPackage("gh:andviane/polynomial_regression#master")
    target_include_directories(polynomial_regression PUBLIC "${polynomial_regression_SOURCE_DIR}")
    add_executable(find_sigma_bound find_sigma_bound.cpp)
    finufft_link_test(find_sigma_bound)
    target_link_libraries(find_sigma_bound PRIVATE polynomial_regression)
    target_include_directories(find_sigma_bound PRIVATE "../test")
    add_executable(calibrate_upsampfac calibrate_upsampfac.cpp)
    finufft_link_test(calibrate_upsampfac)
    target_link_libraries(calibrate_upsampfac PRIVATE xsimd poet::poet)
    target_compile_options(calibrate_upsampfac PRIVATE ${FINUFFT_ARCH_FLAGS})
endif()
