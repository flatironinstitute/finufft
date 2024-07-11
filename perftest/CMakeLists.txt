# Each source test file is instantiated in single and double precision
set(PERFTESTS guru_timing_test manysmallprobs spreadtestnd spreadtestndall)

foreach(TEST ${PERFTESTS})
  add_executable(${TEST} ${TEST}.cpp)
  if (FINUFFT_USE_DUCC0)
    target_compile_definitions(${TEST} PRIVATE -DFINUFFT_USE_DUCC0)
  endif ()
  finufft_link_test(${TEST})

  add_executable(${TEST}f ${TEST}.cpp)
  target_compile_definitions(${TEST}f PRIVATE -DSINGLE)
  if (FINUFFT_USE_DUCC0)
    target_compile_definitions(${TEST}f PRIVATE -DFINUFFT_USE_DUCC0)
  endif ()
  finufft_link_test(${TEST}f)
endforeach()
