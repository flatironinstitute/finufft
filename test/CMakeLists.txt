# Each of these source test files is instantiated in single and double precision
set(TESTS
    basicpassfail
    dumbinputs
    finufft1d_test
    finufft1dmany_test
    finufft2d_test
    finufft2dmany_test
    finufft3d_test
    finufft3dmany_test
    finufft3dkernel_test)

foreach(TEST ${TESTS})
  add_executable(${TEST} ${TEST}.cpp)
  target_compile_features(${TEST} PRIVATE cxx_std_17)
  finufft_link_test(${TEST})

  add_executable(${TEST}f ${TEST}.cpp)
  target_compile_definitions(${TEST}f PRIVATE -DSINGLE)
  target_compile_features(${TEST}f PRIVATE cxx_std_17)
  finufft_link_test(${TEST}f)
endforeach()

# copy the DLLs to the build directory so that the tests can find them there fix
# for windows
copy_dll(finufft basicpassfail)

# Add ctest definitions not for both precisions...
add_executable(testutils testutils.cpp)
if(FINUFFT_USE_DUCC0)
  target_compile_definitions(testutils PRIVATE -DFINUFFT_USE_DUCC0)
endif()
finufft_link_test(testutils)
add_test(
  NAME run_testutils
  COMMAND testutils
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Add ctest definitions that run at both precisions...
function(add_tests_with_prec PREC REQ_TOL CHECK_TOL SUFFIX)
  # All of the following should be run at OMP_NUM_THREADS=4 or something small,
  # as in makefile. This prevents them taking a huge time on a, say, 128-core
  # Rome node. ... but I don't know platform-indep way to do that! Does anyone?

  add_test(
    NAME run_basic_pass_fail_${PREC}
    COMMAND basicpassfail${SUFFIX}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME run_finufft1d_test_${PREC}
    COMMAND finufft1d_test${SUFFIX} 1e2 2e2 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME run_finufft1dmany_test_${PREC}
    COMMAND finufft1dmany_test${SUFFIX} 3 1e2 2e2 ${REQ_TOL} 0 0 0 2 0.0
            ${CHECK_TOL}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME run_finufft2d_test_${PREC}
    COMMAND finufft2d_test${SUFFIX} 1e2 1e1 1e3 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME run_finufft2dmany_test_${PREC}
    COMMAND finufft2dmany_test${SUFFIX} 3 1e2 1e1 1e3 ${REQ_TOL} 0 0 0 2 0.0
            ${CHECK_TOL}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME run_finufft3d_test_${PREC}
    COMMAND finufft3d_test${SUFFIX} 5 10 20 1e2 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME run_finufft3dmany_test_${PREC}
    COMMAND finufft3dmany_test${SUFFIX} 2 5 10 20 1e2 ${REQ_TOL} 0 0 0 2 0.0
            ${CHECK_TOL}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME run_dumbinputs_${PREC}
    COMMAND dumbinputs${SUFFIX}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

endfunction()

# use above function to actually add the tests, with certain requested and check
# tols
add_tests_with_prec(float 1e-5 2e-4 f)
add_tests_with_prec(double 1e-12 1e-11 "")
