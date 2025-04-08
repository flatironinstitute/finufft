# use cpm to dowload mdspan
cpmaddpackage(
  NAME mdspan
  GITHUB_REPOSITORY kokkos/mdspan
  GIT_TAG mdspan-${FINUFFT_MDSPAN_VERSION}
  OPTIONS "MDSPAN_ENABLE_CUDA On"
)
