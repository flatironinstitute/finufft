# Each source test file is instantiated in single and double precision
set(PERFTESTS guru_timing_test manysmallprobs spreadtestnd)

foreach(TEST ${PERFTESTS})
  add_executable(${TEST} ${TEST}.cpp)
  finufft_link_test(${TEST})

  add_executable(${TEST}f ${TEST}.cpp)
  target_compile_definitions(${TEST}f PRIVATE -DSINGLE)
  finufft_link_test(${TEST}f)
endforeach()
