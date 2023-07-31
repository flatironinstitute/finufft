# Each of these source test files is instantiated in single and double precision
set(TESTS basicpassfail dumbinputs finufft1d_test finufft1dmany_test finufft2d_test finufft2dmany_test finufft3d_test finufft3dmany_test)

foreach(TEST ${TESTS})
  add_executable(${TEST} ${TEST}.cpp)
  add_executable(${TEST}f ${TEST}.cpp)

  target_compile_definitions(${TEST}f PRIVATE -DSINGLE)

  if(FINUFFT_STATIC_LINKING)
      target_link_libraries(${TEST} PRIVATE finufft_static)
      target_link_libraries(${TEST}f PRIVATE finufft_static)
      if(FINUFFT_USE_OPENMP)
          target_link_libraries(${TEST} PRIVATE OpenMP::OpenMP_CXX)
          target_link_libraries(${TEST}f PRIVATE OpenMP::OpenMP_CXX)
          if(WIN32)
              target_link_options(${TEST} PRIVATE ${OpenMP_CXX_FLAGS})
              target_link_options(${TEST}f PRIVATE ${OpenMP_CXX_FLAGS})
          endif()
      endif()
  else()
      target_link_libraries(${TEST} PRIVATE finufft)
      target_link_libraries(${TEST}f PRIVATE finufft)
      if(WIN32)
          target_compile_definitions(${TEST} PRIVATE FINUFFT_DLL)
          target_compile_definitions(${TEST}f PRIVATE FINUFFT_DLL)
      endif()
  endif()
  enable_asan(${TEST})
  enable_asan(${TEST}f)
endforeach()

# Add ctest definitions not for both precisions...
add_executable(testutils testutils.cpp)
if(FINUFFT_STATIC_LINKING)
    target_link_libraries(testutils PRIVATE finufft_static)
    if(FINUFFT_USE_OPENMP)
        target_link_libraries(testutils PRIVATE OpenMP::OpenMP_CXX)
        if(WIN32)
            target_link_options(testutils PRIVATE ${OpenMP_CXX_FLAGS})
        endif()
    endif()
else()
    target_link_libraries(testutils PRIVATE finufft)
    if(WIN32)
        target_compile_definitions(testutils PRIVATE FINUFFT_DLL)
    endif()
endif()
enable_asan(testutils)
add_test(NAME run_testutils COMMAND testutils WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Add ctest definitions that run at both precisions...
function(add_tests_with_prec PREC REQ_TOL CHECK_TOL SUFFIX)
  # All of the following should be run at OMP_NUM_THREADS=4 or something small, as in makefile.
  # This prevents them taking a huge time on a, say, 128-core Rome node.
  #... but I don't know platform-indep way to do that! Does anyone?
  
add_test(
  NAME run_basic_pass_fail_${PREC}
  COMMAND basicpassfail${SUFFIX} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(
  NAME run_finufft1d_test_${PREC}
  COMMAND finufft1d_test${SUFFIX} 1e2 2e2 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(
  NAME run_finufft1dmany_test_${PREC}
  COMMAND finufft1dmany_test${SUFFIX} 3 1e2 2e2 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(
  NAME run_finufft2d_test_${PREC}
  COMMAND finufft2d_test${SUFFIX} 1e2 1e1 1e3 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(
  NAME run_finufft2dmany_test_${PREC}
  COMMAND finufft2dmany_test${SUFFIX} 3 1e2 1e1 1e3 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(
  NAME run_finufft3d_test_${PREC}
  COMMAND finufft3d_test${SUFFIX} 5 10 20 1e2 ${REQ_TOL} 0 2 0.0 ${CHECK_TOL} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(
  NAME run_finufft3dmany_test_${PREC}
  COMMAND finufft3dmany_test${SUFFIX} 2 5 10 20 1e2 ${REQ_TOL} 0 0 0 2 0.0 ${CHECK_TOL} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(
  NAME run_dumbinputs_${PREC}
  COMMAND dumbinputs${SUFFIX} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

endfunction()

# use above function to actually add the tests, with certain requested and check tols
add_tests_with_prec(float 1e-5 2e-4 f)
add_tests_with_prec(double 1e-12 1e-11 "")
