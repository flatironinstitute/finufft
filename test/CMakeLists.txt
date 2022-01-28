# Each source test file is instantiated in single and double precision
set(TESTS basicpassfail dumbinputs finufft1d_test finufft1dmany_test finufft2d_test finufft2dmany_test finufft3d_test finufft3dmany_test)

foreach(TEST ${TESTS})
  add_executable(${TEST}_f64 ${TEST}.cpp)

  add_executable(${TEST}_f32 ${TEST}.cpp)
  target_compile_definitions(${TEST}_f32 PRIVATE -DSINGLE)

  target_link_libraries(${TEST}_f64 PRIVATE finufft)
  target_link_libraries(${TEST}_f32 PRIVATE finufft)
endforeach()


add_executable(testutils testutils.cpp)
target_link_libraries(testutils PRIVATE finufft)
