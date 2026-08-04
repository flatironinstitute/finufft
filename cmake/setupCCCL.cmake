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
        SYSTEM
        YES
    )
else()
    # Any CCCL 3.x already on the machine will do, so take the one the CUDA
    # toolkit ships (13.x does; its config lives in <libdir>/cmake/cccl, which
    # plain prefix search does not reach) and only fetch when there is none.
    # Preferring it is not just to save a download: the toolkit's
    # <toolkit>/include/cccl lands ahead of any CPM include dir, so a fetched
    # CCCL would have the toolkit's thrust/cub compiled against its libcudacxx
    # - two incompatible trees, which fails to build.
    cpmfindpackage(
        NAME
        CCCL
        VERSION
        3
        GIT_REPOSITORY
        https://github.com/NVIDIA/cccl.git
        GIT_TAG
        v${CUDA12_CCCL_VERSION}
        SYSTEM
        YES
        FIND_PACKAGE_ARGUMENTS
        "CONFIG PATHS ${CUDAToolkit_LIBRARY_DIR}/cmake ${CUDAToolkit_LIBRARY_ROOT}/lib64/cmake"
    )
endif()
