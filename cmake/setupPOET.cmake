CPMAddPackage(
    NAME
    POET
    GIT_REPOSITORY
    "https://github.com/flatironinstitute/poet.git"
    GIT_TAG
    ${POET_VERSION}
    EXCLUDE_FROM_ALL
    YES
    GIT_SHALLOW
    YES
    OPTIONS
    "POET_BUILD_TESTS OFF"
)
