# use cpm to dowload mdspan
CPMAddPackage(
    NAME
    mdspan
    GITHUB_REPOSITORY
    kokkos/mdspan
    GIT_TAG
    mdspan-${FINUFFT_MDSPAN_VERSION}
    OPTIONS
    "MDSPAN_ENABLE_CUDA On"
)

CPMAddPackage(
    NAME
    span
    GITHUB_REPOSITORY
    DiamonDinoia/cuda-span
    GIT_TAG
    ac4f156
)
