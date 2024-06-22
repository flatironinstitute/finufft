CPMAddPackage(
        NAME ducc0
        GIT_REPOSITORY https://gitlab.mpcdf.mpg.de/mtr/ducc.git
        GIT_TAG ${DUCC0_VERSION}
        DOWNLOAD_ONLY YES
)

if(ducc0_ADDED)
    add_library(ducc0 OBJECT ${ducc0_SOURCE_DIR}/src/ducc0/infra/string_utils.cc ${ducc0_SOURCE_DIR}/src/ducc0/infra/threading.cc ${ducc0_SOURCE_DIR}/src/ducc0/infra/mav.cc ${ducc0_SOURCE_DIR}/src/ducc0/math/gridding_kernel.cc ${ducc0_SOURCE_DIR}/src/ducc0/math/gl_integrator.cc)
    target_include_directories(ducc0 PUBLIC ${ducc0_SOURCE_DIR}/src/)
endif ()

list(APPEND FINUFFT_FFTW_LIBRARIES ducc0)
