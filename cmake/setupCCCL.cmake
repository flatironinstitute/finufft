CPMAddPackage(
    NAME
    CCCL
    GIT_REPOSITORY
    https://github.com/NVIDIA/cccl.git
    VERSION
    ${FINUFFT_CCCL_VERSION}
    #        OPTIONS
    #            "CCCL_ENABLE_LIBCUDACXX ON"
    #            "LIBCUDACXX_ENABLE_LIBCUDACXX_TESTS OFF"
)
