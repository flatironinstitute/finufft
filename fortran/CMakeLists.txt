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
)

if(NOT FINUFFT_USE_DUCC0)
    list(APPEND FORTRAN_EXAMPLES guru1d1)
endif()
foreach(EXAMPLE ${FORTRAN_EXAMPLES})
    add_executable(fort_${EXAMPLE} examples/${EXAMPLE}.f)
    add_executable(fort_${EXAMPLE}f examples/${EXAMPLE}f.f)

    target_link_libraries(fort_${EXAMPLE} PRIVATE directft finufft)
    target_link_libraries(fort_${EXAMPLE}f PRIVATE directft finufft)

    if(BUILD_TESTING AND FINUFFT_BUILD_TESTS)
        add_test(NAME fort_${EXAMPLE} COMMAND fort_${EXAMPLE})
        add_test(NAME fort_${EXAMPLE}f COMMAND fort_${EXAMPLE}f)
    endif()
endforeach()
