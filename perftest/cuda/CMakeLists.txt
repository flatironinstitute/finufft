add_executable(cuperftest cuperftest.cu)
target_include_directories(cuperftest PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
target_link_libraries(cuperftest PUBLIC cufinufft)
set_target_properties(cuperftest PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES ${FINUFFT_CUDA_ARCHITECTURES}
)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bench.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
