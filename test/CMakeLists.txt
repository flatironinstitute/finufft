# Each of these source test files is instantiated in single and double precision
set(TESTS basicpassfail dumbinputs finufft1d_test finufft1dmany_test finufft2d_test finufft2dmany_test finufft3d_test finufft3dmany_test)

foreach(TEST ${TESTS})
  add_executable(${TEST} ${TEST}.cpp)
  add_executable(${TEST}f ${TEST}.cpp)

  target_compile_definitions(${TEST}f PRIVATE -DSINGLE)

  target_link_libraries(${TEST} PRIVATE finufft)
  target_link_libraries(${TEST}f PRIVATE finufft)
  enable_asan(${TEST})
  enable_asan(${TEST}f)
endforeach()

# Ctest definitions not for both precisions...
add_executable(testutils testutils.cpp)
target_link_libraries(testutils PRIVATE finufft)
enable_asan(testutils)
add_test(NAME run_testutils COMMAND testutils)

# Ctest definitions that run at both precisions...
function(add_tests_with_prec PREC REQ_TOL CHECK_TOL SUFFIX)

  # all the following should be run at OMP_NUM_THREADS=4 or something small, as in makefile
  #... but I don't know platform-indep way to do that:
add_test(
  NAME run_basic_pass_fail_${PREC}
  COMMAND basicpassfail${SUFFIX})

add_test(
  NAME run_finufft1d_test_${PREC}
  COMMAND finufft1d_test${SUFFIX} 1e2 2e2 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft1dmany_test_${PREC}
  COMMAND finufft1dmany_test${SUFFIX} 3 1e2 2e2 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft2d_test_${PREC}
  COMMAND finufft2d_test${SUFFIX} 1e2 1e1 1e3 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft2dmany_test_${PREC}
  COMMAND finufft2dmany_test${SUFFIX} 3 1e2 1e1 1e3 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft3d_test_${PREC}
  COMMAND finufft3d_test${SUFFIX} 5 10 20 1e2 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL})

add_test(
  NAME run_finufft3dmany_test_${PREC}
  COMMAND finufft3dmany_test${SUFFIX} 2 5 10 20 1e2 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL})

# Here's where to save output of dumbinputs for later diff
set(tempout
  "${CMAKE_CURRENT_BINARY_DIR}/dumbinputs${SUFFIX}.out"
  )
# follow https://stackoverflow.com/questions/36304289/how-to-use-redirection-in-cmake-add-test
# https://stackoverflow.com/questions/39960173/run-custom-shell-script-with-cmake
# Is pipe here platform-indep? very hard to find out. sh or bash are not cross-platform :(
add_custom_target(
  run_dumbinputs_${PREC} ALL
  COMMENT "Custom target run of dumbinputs${SUFFIX}..."
  # this still spews stuff to stderr which looks broken to the user; it is not...
  COMMAND OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/dumbinputs${SUFFIX} > ${tempout}
  # I don't know if correctly using dependency: it needs the current executable...
  DEPENDS dumbinputs${SUFFIX}
  )
# I considered add_test() as another way (fails since can't handle the pipe >)
# and add_custom_command(), which fails to run unless add_custom_target used anyway.
# annoyingly, add_test COMMAND is quite lame: does not allow piping, as above,
# or multiple cmds
# See: https://stackoverflow.com/questions/3065220/ctest-with-multiple-commands?rq=4
add_test(
  NAME check_dumbinputs_${PREC}
  COMMAND ${CMAKE_COMMAND} -E compare_files ${tempout} ${CMAKE_CURRENT_SOURCE_DIR}/results/dumbinputs${SUFFIX}.refout
  )

endfunction()

# use above function to actually add the tests with certain requested and check tols
add_tests_with_prec(float 1e-5 2e-4 f)
add_tests_with_prec(double 1e-12 1e-11 "")

if (FINUFFT_USE_CUDA)
  add_subdirectory(cuda)
endif()
