string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)

if(CUDA_VERSION_MAJOR LESS 12)
    message(STATUS "CUDA 11 detected")
    CPMAddPackage(
        NAME
        CCCL
        GIT_REPOSITORY
        https://github.com/NVIDIA/cccl.git
        GIT_TAG
        v${FINUFFT_CUDA11_CCCL_VERSION}
    )
else()
    message(STATUS "CUDA 12 detected")
    CPMAddPackage(
        NAME
        CCCL
        GIT_REPOSITORY
        https://github.com/NVIDIA/cccl.git
        GIT_TAG
        v${FINUFFT_CUDA12_CCCL_VERSION}
    )
endif()
