string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
message(STATUS "CUDA ${CUDA_VERSION_MAJOR} detected")
if(CUDA_VERSION_MAJOR LESS 12)
    CPMAddPackage(
        NAME
        CCCL
        GIT_REPOSITORY
        https://github.com/NVIDIA/cccl.git
        GIT_TAG
        v${CUDA11_CCCL_VERSION}
    )
else()
    CPMAddPackage(
        NAME
        CCCL
        GIT_REPOSITORY
        https://github.com/NVIDIA/cccl.git
        GIT_TAG
        v${CUDA12_CCCL_VERSION}
    )
endif()
