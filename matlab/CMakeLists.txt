set(FINUFFT_MEX_DEFS R2008OO)

if(FINUFFT_BUILD_MATLAB)
    # Ensure MATLAB is on PATH if provided by setup-matlab
    find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER MX_LIBRARY)
endif()

# Make MEX self-contained where we can
function(make_mex_self_contained tgt)
    if(UNIX AND NOT APPLE)
        # Avoid MATLAB's older libstdc++ at runtime on Linux
        target_link_options(${tgt} PRIVATE -static-libstdc++ -static-libgcc)
    endif()
endfunction()

# ------------------------------------------------------------
# --- CPU MEX (uses 'mex' via matlab_add_mex) ---
if(FINUFFT_BUILD_MATLAB AND FINUFFT_USE_CPU)
    matlab_add_mex(
        NAME finufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/finufft.cpp"
        OUTPUT_NAME "finufft"
        R2018a
        LINK_TO finufft
    )
    target_compile_definitions(finufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(finufft_mex PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    if(APPLE)
        # needed to prevent new Apple XCode linker which can break mex...
        target_link_options(finufft_mex PRIVATE -ld_classic)
    endif()
    make_mex_self_contained(finufft_mex)
endif()

# ------------------------------------------------------------
# --- CUDA MEX (uses 'mexcuda' via matlab_add_mex) ---
if(FINUFFT_BUILD_MATLAB AND FINUFFT_USE_CUDA)
    # GPU headers from Parallel Computing Toolbox (mxGPUArray.h)
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
            "MATLAB GPU header mxGPUArray.h not found. "
            "Ensure Parallel Computing Toolbox is installed, or add its include path."
        )
    endif()

    # GPU MEX binder (provides mxInitGPU / mxGPU* symbols)
    if(WIN32)
        find_library(Matlab_GPUMEXBINDER_LIB NAMES gpumexbinder HINTS "${Matlab_ROOT_DIR}/extern/lib/win64/microsoft")
    else() # Linux
        find_library(
            Matlab_GPUMEXBINDER_LIB
            NAMES mwgpumexbinder
            HINTS "${Matlab_ROOT_DIR}/bin/glnxa64" "${Matlab_ROOT_DIR}/extern/bin/glnxa64"
        )
    endif()
    if(NOT Matlab_GPUMEXBINDER_LIB)
        message(FATAL_ERROR "MATLAB GPU binder library not found (gpumexbinder/mwgpumexbinder).")
    endif()

    matlab_add_mex(
        NAME cufinufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/cufinufft.cu"
        OUTPUT_NAME "cufinufft"
        R2018a
        LINK_TO cufinufft Matlab::mex Matlab::mx CUDA::cudart CUDA::cufft ${Matlab_GPUMEXBINDER_LIB}
    )

    target_compile_definitions(cufinufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(
        cufinufft_mex
        PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${Matlab_GPU_INCLUDE_DIR}>
    )
    # Windows: export the standard MEX entry points (mexcuda also does this)
    if(WIN32)
        target_link_options(cufinufft_mex PRIVATE /EXPORT:mexFunction /EXPORT:mexfilerequiredapiversion)
        # Assist library resolution for MATLAB libs if needed
        target_link_directories(cufinufft_mex PRIVATE "${Matlab_ROOT_DIR}/extern/lib/win64/microsoft")
    endif()

    make_mex_self_contained(cufinufft_mex)
endif()

# ------------------------------------------------------------
# MATLAB assets (.m and .docsrc files)
file(GLOB MATLAB_MFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.m" "${CMAKE_CURRENT_SOURCE_DIR}/*.docsrc")

# ------------------------------------------------------------
# --- Octave MEX (native CMake module library; uses Octave headers/libs) ---
if(FINUFFT_BUILD_OCTAVE)
    find_program(Octave_MKOCTFILE_EXECUTABLE mkoctfile)
    find_program(Octave_EXECUTABLE octave)
    find_program(Octave_CONFIG_EXECUTABLE octave-config)
    if(NOT Octave_MKOCTFILE_EXECUTABLE OR NOT Octave_EXECUTABLE OR NOT Octave_CONFIG_EXECUTABLE)
        message(
            FATAL_ERROR
            "FINUFFT_BUILD_OCTAVE=ON but mkoctfile / octave / octave-config not "
            "found on PATH. Install octave + liboctave-dev, or set "
            "Octave_MKOCTFILE_EXECUTABLE / Octave_EXECUTABLE / Octave_CONFIG_EXECUTABLE."
        )
    endif()

    # Query Octave install layout via mkoctfile -p (canonical, matches what
    # 'mkoctfile --mex' would use internally).
    execute_process(
        COMMAND ${Octave_MKOCTFILE_EXECUTABLE} -p OCTINCLUDEDIR
        OUTPUT_VARIABLE Octave_INCLUDEDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND ${Octave_MKOCTFILE_EXECUTABLE} -p OCTLIBDIR
        OUTPUT_VARIABLE Octave_LIBDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND ${Octave_CONFIG_EXECUTABLE} --version
        OUTPUT_VARIABLE Octave_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Found Octave ${Octave_VERSION} (include: ${Octave_INCLUDEDIR}, lib: ${Octave_LIBDIR})")

    set(_octave_mex_dir "${CMAKE_CURRENT_BINARY_DIR}/octave")

    # Octave's mex loader checks for an integer symbol __octave_mex_soversion__
    # exported by the .mex; mkoctfile injects this via a tiny auto-generated TU.
    # We replicate that here so the .mex loads without "No SOVERSION found" errors.
    set(_octave_soversion_src "${CMAKE_CURRENT_BINARY_DIR}/octave_mex_soversion.c")
    file(WRITE ${_octave_soversion_src} "const int __octave_mex_soversion__ = 1;\n")

    add_library(octave_mex MODULE "${CMAKE_CURRENT_SOURCE_DIR}/finufft.cpp" ${_octave_soversion_src})
    set_target_properties(
        octave_mex
        PROPERTIES OUTPUT_NAME finufft PREFIX "" SUFFIX ".mex" LIBRARY_OUTPUT_DIRECTORY ${_octave_mex_dir}
    )
    # Octave headers expect the parent dir on the include path too (octave/octave.h vs
    # bare octave.h, depending on Octave version).
    target_include_directories(octave_mex SYSTEM PRIVATE ${Octave_INCLUDEDIR} ${Octave_INCLUDEDIR}/..)
    target_compile_definitions(octave_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_compile_features(octave_mex PRIVATE cxx_std_17)
    # finufft propagates its FFT backend, OpenMP and finufft_common deps via
    # CMake's static-lib LINK_ONLY interface — no manual plumbing needed.
    target_link_libraries(octave_mex PRIVATE finufft)
    target_link_directories(octave_mex PRIVATE ${Octave_LIBDIR})
    # liboctmex exists only in Octave >= 7; on older releases (e.g. Ubuntu 22.04
    # ships Octave 6.4) mkoctfile links -loctinterp -loctave instead.
    find_library(Octave_OCTMEX_LIBRARY octmex HINTS ${Octave_LIBDIR})
    if(Octave_OCTMEX_LIBRARY)
        target_link_libraries(octave_mex PRIVATE octmex)
    else()
        target_link_libraries(octave_mex PRIVATE octinterp octave)
    endif()
    # Side-by-side libfinufft.so when finufft is built shared.
    if(NOT FINUFFT_STATIC_LINKING AND UNIX AND NOT APPLE)
        set_target_properties(octave_mex PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "\$ORIGIN")
        add_custom_command(
            TARGET octave_mex
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:finufft> ${_octave_mex_dir}
        )
    endif()

    # Copy .m / .docsrc assets beside the mex so addpath() resolves wrappers + mex.
    add_custom_command(
        TARGET octave_mex
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MATLAB_MFILES} ${_octave_mex_dir}
    )

    if(FINUFFT_BUILD_TESTS)
        set(_octave_tests
            test/fullmathtest.m
            test/check_finufft.m
            test/check_finufft_single.m
            examples/guru1d1.m
            examples/guru1d1_single.m
            examples/guru1d1_adjoint.m
        )

        # Fixture: build the mex before any octave_* test runs.
        add_test(
            NAME octave_mex_build
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target octave_mex
        )
        set_tests_properties(octave_mex_build PROPERTIES FIXTURES_SETUP octave_mex_built LABELS octave)

        foreach(_t IN LISTS _octave_tests)
            get_filename_component(_n ${_t} NAME_WE)
            add_test(
                NAME octave_${_n}
                COMMAND
                    ${Octave_EXECUTABLE} --no-gui --norc -q --eval
                    "addpath('${_octave_mex_dir}'); run('${CMAKE_CURRENT_SOURCE_DIR}/${_t}')"
            )
            set_tests_properties(octave_${_n} PROPERTIES LABELS octave FIXTURES_REQUIRED octave_mex_built)
        endforeach()
    endif()
endif()

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
