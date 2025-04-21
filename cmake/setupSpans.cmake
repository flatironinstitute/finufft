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
    DiamonDinoia/span
    GIT_TAG
    e53094544c66c5156e21be19a521cf0ab2962058
)
