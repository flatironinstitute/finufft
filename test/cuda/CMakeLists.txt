
file(GLOB test_src "*.cu")

foreach(srcfile ${test_src})
  string(REPLACE ".cu" "" executable ${srcfile})
  get_filename_component(executable ${executable} NAME)
  add_executable(${executable} ${srcfile})
  target_include_directories(${executable} PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
  target_link_libraries(${executable} PUBLIC cufinufft)
endforeach()

foreach(basename spread1d_test spread2d_test spread3d_test interp1d_test interp2d_test interp3d_test fseries_kernel_test)
  set(executable "${basename}_32")
  set(srcfile "${basename}.cu")
  add_executable(${executable} ${srcfile})
  target_include_directories(${executable} PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
  target_link_libraries(${executable} PUBLIC cufinufft)
  target_compile_definitions(${executable} PUBLIC CUFINUFFT_SINGLE)
endforeach()

configure_file(spreadperf.sh ${CMAKE_BINARY_DIR}/test/cuda/spreadperf.sh COPYONLY)
configure_file(fseriesperf.sh ${CMAKE_BINARY_DIR}/test/cuda/fseriesperf.sh COPYONLY)

function(add_tests REQ_TOL CHECK_TOL)
  add_test(
    NAME run_cufinufft1d1_test_nupts
    COMMAND cufinufft1d1_test 1 1e2 2e2 ${REQ_TOL} ${CHECK_TOL})

  add_test(
    NAME run_cufinufft1d1_test_sub_problem
    COMMAND cufinufft1d1_test 2 1e2 2e2 ${REQ_TOL} ${CHECK_TOL})

  add_test(
    NAME run_cufinufft1d2_test_nupts
    COMMAND cufinufft1d2_test 1 1e2 2e2 ${REQ_TOL} ${CHECK_TOL})
endfunction()

add_tests(1e-5 2e-4)
