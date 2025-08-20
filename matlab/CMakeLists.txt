# Ensure MATLAB is on PATH if provided by setup-matlab
find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER MX_LIBRARY)

# MEX extension and common defines
set(MEX_EXT "${Matlab_MEX_EXTENSION}")
set(FINUFFT_MEX_DEFS R2008OO)

# Make MEX self-contained where we can
function(make_mex_self_contained tgt)
    if(UNIX AND NOT APPLE)
        # Avoid MATLAB's older libstdc++ at runtime on Linux
        target_link_options(${tgt} PRIVATE -static-libstdc++ -static-libgcc)
    endif()
endfunction()

# ------------------------------------------------------------
# --- CPU MEX (uses 'mex' via matlab_add_mex) ---
if(FINUFFT_USE_CPU)
    matlab_add_mex(
        NAME finufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/finufft.cpp"
        OUTPUT_NAME "finufft"
        R2018a
        LINK_TO finufft
    )
    target_compile_definitions(finufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(finufft_mex PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    make_mex_self_contained(finufft_mex)
endif()

# ------------------------------------------------------------
# --- CUDA MEX (uses 'mexcuda' via matlab_add_mex) ---
if(FINUFFT_USE_CUDA)
    find_path(
        Matlab_GPU_INCLUDE_DIR
        NAMES gpu/mxGPUArray.h
        HINTS
            "${Matlab_ROOT_DIR}/extern/include"
            "${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include"
            "${Matlab_ROOT_DIR}/toolbox/distcomp/gpu/extern/include"
    )
    if(NOT Matlab_GPU_INCLUDE_DIR)
        message(
            FATAL_ERROR
            "MATLAB GPU header mxGPUArray.h not found. \
            Ensure Parallel Computing Toolbox is installed, or add its include path."
        )
    endif()
    matlab_add_mex(
        NAME cufinufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/cufinufft.cu"
        OUTPUT_NAME "cufinufft"
        R2018a
        LINK_TO cufinufft Matlab::mex Matlab::mx CUDA::cudart
    )
    target_compile_definitions(cufinufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(cufinufft_mex PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    target_include_directories(cufinufft_mex PRIVATE $<BUILD_INTERFACE:${Matlab_GPU_INCLUDE_DIR}>)
    make_mex_self_contained(cufinufft_mex)
endif()

# ------------------------------------------------------------
# MATLAB assets (.m and .docsrc files)
file(GLOB MATLAB_MFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.m" "${CMAKE_CURRENT_SOURCE_DIR}/*.docsrc")

# Copy .m files next to built MEX(es) for in-tree testing
if(TARGET finufft_mex)
    add_custom_command(
        TARGET finufft_mex
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MATLAB_MFILES} $<TARGET_FILE_DIR:finufft_mex>
    )
endif()
if(TARGET cufinufft_mex)
    add_custom_command(
        TARGET cufinufft_mex
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MATLAB_MFILES} $<TARGET_FILE_DIR:cufinufft_mex>
    )
endif()

# ------------------------------------------------------------
# RPATH for installed artifacts
# Option A (macOS): don't overwrite MATLAB's rpaths; let them propagate.
foreach(tgt IN ITEMS finufft_mex cufinufft_mex)
    if(TARGET ${tgt})
        if(APPLE)
            set_target_properties(
                ${tgt}
                PROPERTIES
                    BUILD_WITH_INSTALL_RPATH OFF
                    INSTALL_RPATH
                        "" # do not clobber MATLAB rpaths
                    INSTALL_RPATH_USE_LINK_PATH
                        TRUE # propagate link-line rpaths
            )
        elseif(UNIX AND NOT APPLE)
            # Side-by-side loading on Linux (your own shared libs next to the MEX)
            set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "\$ORIGIN" INSTALL_RPATH_USE_LINK_PATH TRUE)
        endif()
    endif()
endforeach()

# ------------------------------------------------------------
# Install & CPack (optional)
if(FINUFFT_MATLAB_INSTALL)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CPACK_STRIP_FILES TRUE)
    endif()

    # Install MATLAB assets
    if(MATLAB_MFILES)
        install(FILES ${MATLAB_MFILES} DESTINATION matlab)
    endif()

    # Install class folder (if present)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/@gpuArray")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/@gpuArray" DESTINATION matlab)
    endif()

    # Install the MEX targets into matlab/
    if(TARGET finufft_mex)
        install(TARGETS finufft_mex RUNTIME DESTINATION matlab LIBRARY DESTINATION matlab ARCHIVE DESTINATION matlab)
    endif()
    if(TARGET cufinufft_mex)
        install(TARGETS cufinufft_mex RUNTIME DESTINATION matlab LIBRARY DESTINATION matlab ARCHIVE DESTINATION matlab)
    endif()

    # CPack: produce a clean archive with ONLY top-level "matlab/"
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    if(WIN32)
        set(CPACK_GENERATOR "ZIP")
        set(CPACK_PACKAGE_FILE_NAME "finufft-matlab-mex-windows")
    elseif(APPLE)
        set(CPACK_GENERATOR "TGZ")
        set(CPACK_PACKAGE_FILE_NAME "finufft-matlab-mex-macos-${CMAKE_OSX_DEPLOYMENT_TARGET}")
    else()
        set(CPACK_GENERATOR "TGZ")
        set(CPACK_PACKAGE_FILE_NAME "finufft-matlab-mex-linux")
    endif()

    include(CPack)
endif()
