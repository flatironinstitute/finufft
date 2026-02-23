add_executable(cuperftest cuperftest.cu)
target_link_libraries(cuperftest PRIVATE cufinufft)
target_compile_features(cuperftest PRIVATE cxx_std_17)
set_target_properties(
    cuperftest
    PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

add_executable(binsize_sweep binsize_sweep.cu)
target_link_libraries(binsize_sweep PRIVATE cufinufft)
target_compile_features(binsize_sweep PRIVATE cxx_std_17)
set_target_properties(
    binsize_sweep
    PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)
