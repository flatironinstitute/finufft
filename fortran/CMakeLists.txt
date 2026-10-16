add_library(
    directft
    OBJECT
    directft/dirft1d.f
    directft/dirft1df.f
    directft/dirft2d.f
    directft/dirft2df.f
    directft/dirft3d.f
    directft/dirft3df.f
)

set(FORTRAN_EXAMPLES
    nufft1d_demo
    nufft2d_demo
    nufft2dmany_demo
    nufft3d_demo
    simple1d1
    guru1d1_adjoint
    guru1d2_adjoint
)
# Notes: all these need single-prec versions.
# we still don't build simple1d1_f90

# see end for reason for this
if(NOT FINUFFT_USE_DUCC0)
    list(APPEND FORTRAN_EXAMPLES guru1d1)
endif()

foreach(EXAMPLE ${FORTRAN_EXAMPLES})
    if(FINUFFT_BUILD_EXAMPLES OR (BUILD_TESTING AND FINUFFT_BUILD_TESTS))
        add_executable(fort_${EXAMPLE} examples/${EXAMPLE}.f)
        add_executable(fort_${EXAMPLE}f examples/${EXAMPLE}f.f)

        target_link_libraries(fort_${EXAMPLE} PRIVATE directft finufft)
        target_link_libraries(fort_${EXAMPLE}f PRIVATE directft finufft)
    endif()

    if(BUILD_TESTING AND FINUFFT_BUILD_TESTS)
        add_test(NAME fort_${EXAMPLE} COMMAND fort_${EXAMPLE})
        add_test(NAME fort_${EXAMPLE}f COMMAND fort_${EXAMPLE}f)
    endif()
endforeach()

# guru1d1/guru1d1f include fftw3.f for the FFTW plan-mode constants, so a DUCC0 build
# (no FFTW dependency) omits them from FORTRAN_EXAMPLES above instead of failing to compile.
if(TARGET fort_guru1d1)
    target_link_libraries(fort_guru1d1 PRIVATE finufft_fftlibs)
    target_link_libraries(fort_guru1d1f PRIVATE finufft_fftlibs)
endif()

# simple1d1_f90 needs finufft_mod.f90, so it does not fit the EXAMPLE/EXAMPLEf pairing above.
if(FINUFFT_BUILD_EXAMPLES OR (BUILD_TESTING AND FINUFFT_BUILD_TESTS))
    add_executable(fort_simple1d1_f90 ../include/finufft_mod.f90 examples/simple1d1.f90)
    target_link_libraries(fort_simple1d1_f90 PRIVATE finufft)
endif()

if(BUILD_TESTING AND FINUFFT_BUILD_TESTS)
    add_test(NAME fort_simple1d1_f90 COMMAND fort_simple1d1_f90)
endif()
