include(CheckIncludeFileCXX)
# Try to find libcudacxx from the installed CUDA toolkit
check_include_file_cxx("cuda/std/mdspan" HAS_CUDA_STD_MDSPAN)

if(HAS_CUDA_STD_MDSPAN)
    message(STATUS "Found system cuda/std/mdspan no need to download")
else()
    message(STATUS "cuda/std/mdspan not found, downloading via CPM")
    CPMAddPackage(
        NAME
        CCCL
        GIT_REPOSITORY
        https://github.com/NVIDIA/cccl.git
        VERSION
        ${FINUFFT_CCCL_VERSION}
    )
    set(CCCL_LIB CCCL::CCCL)
endif()
