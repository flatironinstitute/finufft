set(EXAMPLES
    guru1d1
    guru1d1f
    guru2d1
    many1d1
    simple1d1
    simple1d1f
    simulplans1d1)
set(EXAMPLES_OPENMP threadsafe1d1 threadsafe2d2f)
set(EXAMPLES_C guru1d1c simple1d1c simple1d1cf)

find_library(MATH_LIBRARY m)

foreach(EXAMPLE ${EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_compile_features(${EXAMPLE} PRIVATE cxx_std_14)
  target_link_libraries(${EXAMPLE} PRIVATE finufft)
  if(CMAKE_PROJECT_NAME STREQUAL "FINUFFT")
    enable_asan(${EXAMPLE})
  endif()
endforeach()

foreach(EXAMPLE ${EXAMPLES_C})
  add_executable(${EXAMPLE} ${EXAMPLE}.c)
  target_link_libraries(${EXAMPLE} PRIVATE finufft)
  if(CMAKE_PROJECT_NAME STREQUAL "FINUFFT")
    enable_asan(${EXAMPLE})
  endif()
  if(MATH_LIBRARY)
    target_link_libraries(${EXAMPLE} PRIVATE ${MATH_LIBRARY})
  endif()
endforeach()

if(FINUFFT_USE_OPENMP)
  foreach(EXAMPLE ${EXAMPLES_OPENMP})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} PRIVATE finufft OpenMP::OpenMP_CXX)
    target_compile_features(${EXAMPLE} PRIVATE cxx_std_11)
    if(CMAKE_PROJECT_NAME STREQUAL "FINUFFT")
      enable_asan(${EXAMPLE})
    endif()
  endforeach()
endif()
