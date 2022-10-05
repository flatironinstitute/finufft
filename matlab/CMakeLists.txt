find_package(Matlab REQUIRED)
matlab_add_mex(NAME finufft_mex SRC finufft.cpp LINK_TO finufft OUTPUT_NAME finufft R2018a)
target_compile_definitions(finufft_mex PRIVATE -DR2008OO)

file(GLOB FINUFFT_MATLAB_M_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.m)

add_custom_command(
    TARGET finufft_mex
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${FINUFFT_MATLAB_M_SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
    VERBATIM)
