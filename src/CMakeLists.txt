include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(finufft_source
  spreadinterp.cpp  utils.cpp
  finufft.cpp simpleinterfaces.cpp)

add_library(finufft_float OBJECT ${finufft_source})
target_compile_definitions(finufft_float PRIVATE SINGLE)
target_link_libraries(finufft_float PRIVATE FFTW3::fftw3f)

add_library(finufft_double OBJECT ${finufft_source})
target_compile_definitions(finufft_double PRIVATE DOUBLE)
target_link_libraries(finufft_double PRIVATE FFTW3::fftw3)

add_library(finufft 
	$<TARGET_OBJECTS:finufft_float> 
	$<TARGET_OBJECTS:finufft_double> 
	utils_precindep.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../contrib/legendre_rule_fast.c)

target_link_libraries(finufft PUBLIC FFTW3::fftw3)
target_link_libraries(finufft PUBLIC FFTW3::fftw3f)

target_compile_options(finufft PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries(finufft PRIVATE ${OpenMP_C_LIBRARIES})
