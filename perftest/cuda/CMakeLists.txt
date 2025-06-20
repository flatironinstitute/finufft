add_executable(cuperftest cuperftest.cu)
target_include_directories(cuperftest PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
target_link_libraries(cuperftest cufinufft CUDA::cufft CUDA::cudart)
target_compile_features(cuperftest PRIVATE cxx_std_17)
target_compile_options(cuperftest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
set_target_properties(
    cuperftest
    PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)
