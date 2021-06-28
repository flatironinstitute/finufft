include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
link_libraries(finufft)

add_executable(basicpassfail basicpassfail.cpp)
add_executable(finufft1d_test finufft1d_test.cpp)
add_executable(finufft2d_test finufft2d_test.cpp)
add_executable(finufft3d_test finufft3d_test.cpp)

add_executable(basicpassfailf basicpassfail.cpp)
add_executable(finufft1df_test finufft1d_test.cpp)
add_executable(finufft2df_test finufft2d_test.cpp)
add_executable(finufft3df_test finufft3d_test.cpp)

foreach(test_double basicpassfail finufft1d_test finufft2d_test finufft3d_test)
	target_compile_definitions(${test_double} PRIVATE DOUBLE)
endforeach()

foreach(test_float basicpassfailf finufft1df_test finufft2df_test finufft3df_test)
	target_compile_definitions(${test_float} PRIVATE SINGLE)
endforeach()
