file(GLOB test_src "*.c*")

foreach(srcfile ${test_src})
  string(REGEX REPLACE "\\.c.?" "" executable ${srcfile})
  get_filename_component(executable ${executable} NAME)
  add_executable(${executable} ${srcfile})
  target_include_directories(${executable} PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
  target_compile_options(${executable}
                         PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  find_library(MathLib m)
  if(MathLib)
    target_link_libraries(${executable} PUBLIC cufinufft ${MathLib})
  endif()
  target_compile_features(${executable} PUBLIC cxx_std_17)
  set_target_properties(
    ${executable} PROPERTIES LINKER_LANGUAGE CUDA
                             CUDA_ARCHITECTURES "${FINUFFT_CUDA_ARCHITECTURES}")
  message(STATUS "Adding test ${executable}"
                 " with CUDA_ARCHITECTURES=${FINUFFT_CUDA_ARCHITECTURES}"
                 " and INCLUDE=${CUFINUFFT_INCLUDE_DIRS}")
endforeach()

function(add_tests PREC REQ_TOL CHECK_TOL UPSAMP)
  add_test(NAME cufinufft1d1_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft1d_test 1 1 1e2 2e2 ${REQ_TOL} ${CHECK_TOL} ${PREC}
                   ${UPSAMP})

  add_test(NAME cufinufft1d1_test_SM_${PREC}_${UPSAMP}
           COMMAND cufinufft1d_test 2 1 1e2 2e2 ${REQ_TOL} ${CHECK_TOL} ${PREC}
                   ${UPSAMP})

  add_test(NAME cufinufft1d2_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft1d_test 1 2 1e2 2e2 ${REQ_TOL} ${CHECK_TOL} ${PREC}
                   ${UPSAMP})
  add_test(NAME cufinufft1d3_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft1d_test 1 3 1e2 2e2 ${REQ_TOL} ${CHECK_TOL} ${PREC}
                   ${UPSAMP})

  add_test(NAME cufinufft2d1_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft2d_test 1 1 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL}
                   ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d1_test_SM_${PREC}_${UPSAMP}
           COMMAND cufinufft2d_test 2 1 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL}
                   ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d2_test_SM_${PREC}_${UPSAMP}
           COMMAND cufinufft2d_test 2 2 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL}
                   ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d3_test_SM_${PREC}_${UPSAMP}
           COMMAND cufinufft2d_test 2 3 1e2 2e2 2e4 ${REQ_TOL} ${CHECK_TOL}
                   ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d1many_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft2dmany_test 1 1 1e2 2e2 5 0 2e4 ${REQ_TOL}
                   ${CHECK_TOL} ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d1many_test_SM_${PREC}_${UPSAMP}
           COMMAND cufinufft2dmany_test 2 1 1e2 2e2 5 0 2e4 ${REQ_TOL}
                   ${CHECK_TOL} ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d2many_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft2dmany_test 1 2 1e2 2e2 5 0 2e4 ${REQ_TOL}
                   ${CHECK_TOL} ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d2many_test_SM_${PREC}_${UPSAMP}
           COMMAND cufinufft2dmany_test 2 2 1e2 2e2 5 0 2e4 ${REQ_TOL}
                   ${CHECK_TOL} ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d3many_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft2dmany_test 1 3 1e2 2e2 5 0 2e4 ${REQ_TOL}
                   ${CHECK_TOL} ${PREC} ${UPSAMP})

  add_test(NAME cufinufft2d3many_test_SM_${PREC}_${UPSAMP}
           COMMAND cufinufft2dmany_test 2 3 1e2 2e2 5 0 2e4 ${REQ_TOL}
                   ${CHECK_TOL} ${PREC} ${UPSAMP})

  add_test(NAME cufinufft3d1_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft3d_test 1 1 2 5 10 20 ${REQ_TOL} ${CHECK_TOL}
                   ${PREC} ${UPSAMP})

  if(${PREC} STREQUAL "float")
    add_test(NAME cufinufft3d1_test_SM_${PREC}_${UPSAMP}
             COMMAND cufinufft3d_test 2 1 2 5 10 20 ${REQ_TOL} ${CHECK_TOL}
                     ${PREC} ${UPSAMP})

    add_test(NAME cufinufft3d1_test_block_${PREC}_${UPSAMP}
             COMMAND cufinufft3d_test 4 1 2 5 10 20 ${REQ_TOL} ${CHECK_TOL}
                     ${PREC} ${UPSAMP})

    add_test(NAME cufinufft3d2_test_SM_${PREC}_${UPSAMP}
             COMMAND cufinufft3d_test 2 2 2 5 10 20 ${REQ_TOL} ${CHECK_TOL}
                     ${PREC} ${UPSAMP})

    add_test(NAME cufinufft3d3_test_SM_${PREC}_${UPSAMP}
             COMMAND cufinufft3d_test 2 3 2 5 10 30 ${REQ_TOL} ${CHECK_TOL}
                     ${PREC} ${UPSAMP})
  endif()

  add_test(NAME cufinufft3d2_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft3d_test 1 2 2 5 10 20 ${REQ_TOL} ${CHECK_TOL}
                   ${PREC} ${UPSAMP})

  add_test(NAME cufinufft3d3_test_GM_${PREC}_${UPSAMP}
           COMMAND cufinufft3d_test 1 3 2 3 7 20 ${REQ_TOL} ${CHECK_TOL}*100
                   ${PREC} ${UPSAMP})
endfunction()

add_test(NAME cufinufft_public_api COMMAND public_api_test)
add_test(NAME cufinufft_makeplan COMMAND test_makeplan)
add_test(NAME cufinufft_math_test COMMAND cufinufft_math_test)

add_tests(float 1e-5 2e-4 2.0)
add_tests(double 1e-12 1e-11 2.0)
add_tests(float 1e-5 2e-4 1.25)
add_tests(double 1e-8 1e-7 1.25)
# the upsamp is appended to the testname, ctest does not allows multiple tests
# to share the same testname hence we use the trick 0. and 0.f to differentiate
# the tests and allow them to run in the future we should add the precision to
# the test (f
add_tests(float 1e-5 2e-4 0.f)
add_tests(double 1e-12 1e-11 0.f)
add_tests(float 1e-5 2e-4 0.)
add_tests(double 1e-8 1e-7 0.)
