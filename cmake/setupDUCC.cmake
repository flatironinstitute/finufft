CPMAddPackage(
        NAME ducc0
        GIT_REPOSITORY https://gitlab.mpcdf.mpg.de/mtr/ducc.git
        GIT_TAG ${DUCC0_VERSION}
        DOWNLOAD_ONLY YES
)

if(ducc0_ADDED)
    add_library(ducc0 INTERFACE)
    target_include_directories(ducc0 INTERFACE ${ducc0_SOURCE_DIR}/src/)
endif ()

list(APPEND FINUFFT_FFTW_LIBRARIES ducc0)