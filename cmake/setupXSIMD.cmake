CPMAddPackage(
    NAME
    xsimd
    GIT_REPOSITORY
    "https://github.com/xtensor-stack/xsimd.git"
    GIT_TAG
    ${XSIMD_VERSION}
    EXCLUDE_FROM_ALL
    YES
    GIT_SHALLOW
    NO
    OPTIONS
    "XSIMD_SKIP_INSTALL YES"
    "CMAKE_POLICY_VERSION_MINIMUM 3.10"
)
