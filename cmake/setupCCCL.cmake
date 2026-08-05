string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
message(STATUS "CUDA ${CUDA_VERSION_MAJOR} detected")
if(CUDA_VERSION_MAJOR LESS 12)
    # CUDA 11 ships libcudacxx 1.x, too old for us (no structured bindings on
    # cuda::std::tuple). SYSTEM NO is required, not cosmetic: nvcc puts its own
    # include dir ahead of every -isystem path but after -I, so a SYSTEM package
    # would give cuda/std/* from the toolkit and thrust/cub from here.
    CPMAddPackage(
        NAME
        CCCL
        GIT_REPOSITORY
        https://github.com/NVIDIA/cccl.git
        GIT_TAG
        v${CUDA11_CCCL_VERSION}
        SYSTEM
        NO
    )
else()
    # Prefer the CCCL the toolkit ships (13.x does; its config lives in
    # <libdir>/cmake/cccl, which plain prefix search does not reach) - its
    # <toolkit>/include/cccl wins over any CPM include dir anyway, so fetching
    # over it just mixes two trees. 12.x ships thrust/cub but no config, so it
    # fetches, and non-SYSTEM for the same reason as the CUDA 11 branch.
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
        NO
        FIND_PACKAGE_ARGUMENTS
        "CONFIG PATHS ${CUDAToolkit_LIBRARY_DIR}/cmake ${CUDAToolkit_LIBRARY_ROOT}/lib64/cmake"
    )
endif()
