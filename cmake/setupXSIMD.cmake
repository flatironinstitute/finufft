cpmaddpackage(
  NAME
  xtl
  GIT_REPOSITORY
  "https://github.com/xtensor-stack/xtl.git"
  GIT_TAG
  ${XTL_VERSION}
  EXCLUDE_FROM_ALL
  YES
  GIT_SHALLOW
  YES
  OPTIONS
  "XTL_DISABLE_EXCEPTIONS YES")

cpmaddpackage(
  NAME
  xsimd
  GIT_REPOSITORY
  "https://github.com/xtensor-stack/xsimd.git"
  GIT_TAG
  ${XSIMD_VERSION}
  EXCLUDE_FROM_ALL
  YES
  GIT_SHALLOW
  YES
  OPTIONS
  "XSIMD_SKIP_INSTALL YES"
  "XSIMD_ENABLE_XTL_COMPLEX YES")
