include(CheckCXXCompilerFlag)

# Define the function
function(filter_supported_compiler_flags input_flags_var output_flags_var)
    # Create an empty list to store supported flags
    set(supported_flags)
    # Iterate over each flag in the input list
    set(ORIGINAL_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
    foreach(flag ${${input_flags_var}})
        string(REPLACE "=" "_" flag_var ${flag}) # Convert flag to a valid variable
        # name
        string(REPLACE "-" "" flag_var ${flag_var}) # Remove '-' for the variable
        # name Append the test linker flag to the existing flags
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}")
        check_cxx_compiler_flag(${flag} ${flag_var})
        if(${flag_var})
            # If supported, append the flag to the list of supported flags
            list(APPEND supported_flags ${flag})
        else()
            message(STATUS "Flag ${flag} is not supported")
        endif()
        unset(${flag_var} CACHE)
        # remove last flag from CMAKE_EXE_LINKER_FLAGS using substring
        set(CMAKE_EXE_LINKER_FLAGS ${ORIGINAL_LINKER_FLAGS})
    endforeach()
    # Set the output variable to the list of supported flags
    set(${output_flags_var} ${supported_flags} PARENT_SCOPE)
endfunction()

function(check_arch_support)
    message(STATUS "Checking for AVX, AVX512 and SSE support")
    try_run(
        RUN_RESULT_VAR
        COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckAVX.cpp
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
        RUN_OUTPUT_VARIABLE RUN_OUTPUT
    )
    if(RUN_OUTPUT MATCHES "AVX512")
        set(FINUFFT_ARCH_FLAGS "/arch:AVX512" CACHE STRING "" FORCE)
    elseif(RUN_OUTPUT MATCHES "AVX2")
        set(FINUFFT_ARCH_FLAGS "/arch:AVX2" CACHE STRING "" FORCE)
    elseif(RUN_OUTPUT MATCHES "AVX")
        set(FINUFFT_ARCH_FLAGS "/arch:AVX" CACHE STRING "" FORCE)
    elseif(RUN_OUTPUT MATCHES "SSE")
        set(FINUFFT_ARCH_FLAGS "/arch:SSE" CACHE STRING "" FORCE)
    else()
        set(FINUFFT_ARCH_FLAGS "" CACHE STRING "" FORCE)
    endif()
    message(STATUS "CPU supports: ${RUN_OUTPUT}")
    message(STATUS "Using MSVC flags: ${FINUFFT_ARCH_FLAGS}")
endfunction()

function(copy_dll source_target destination_target)
    if(NOT WIN32)
        return()
    endif()
    # Get the binary directory of the destination target
    get_target_property(DESTINATION_DIR ${destination_target} BINARY_DIR)
    set(DESTINATION_FILE ${DESTINATION_DIR}/$<TARGET_FILE_NAME:${source_target}>)
    if(NOT EXISTS ${DESTINATION_FILE})
        message(STATUS "Copying ${source_target} to ${DESTINATION_DIR} directory for ${destination_target}")
        # Define the custom command to copy the source target to the destination
        # directory
        add_custom_command(
            TARGET ${destination_target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${source_target}> ${DESTINATION_FILE}
            COMMENT "Copying ${source_target} to ${destination_target} directory"
        )
    endif()
    # Unset the variables to leave a clean state
    unset(DESTINATION_DIR)
    unset(SOURCE_FILE)
    unset(DESTINATION_FILE)
endfunction()

if(FINUFFT_INTERPROCEDURAL_OPTIMIZATION)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
    if(NOT LTO_SUPPORTED)
        message(WARNING "IPO is not supported: ${LTO_ERROR}")
        set(FINUFFT_INTERPROCEDURAL_OPTIMIZATION OFF)
    else()
        message(STATUS "IPO is supported, enabling interprocedural optimization")
    endif()
endif()

function(enable_asan target)
    target_compile_options(${target} PRIVATE ${FINUFFT_SANITIZER_FLAGS})
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
        target_link_options(${target} PRIVATE ${FINUFFT_SANITIZER_FLAGS})
    endif()
endfunction()

function(finufft_link_test target)
    target_link_libraries(${target} PRIVATE finufft::finufft)
    if(FINUFFT_USE_DUCC0)
        target_compile_definitions(${target} PRIVATE FINUFFT_USE_DUCC0)
    endif()
    enable_asan(${target})
    target_compile_features(${target} PRIVATE cxx_std_17)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE})
    if(FINUFFT_HAS_NO_DEPRECATED_DECLARATIONS)
        target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
    endif()
endfunction()

# Requires CMake >= 3.13 (for target_link_options / target_compile_options)
# Usage:
#   enable_lto(<target>)        # Clang: ThinLTO, GCC/MSVC/NVCC: full LTO (Clang defaults to Thin)
#   enable_lto(<target> FULL)   # Force full LTO on Clang
function(enable_lto target)
    if(NOT FINUFFT_INTERPROCEDURAL_OPTIMIZATION)
        return()
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "enable_lto(): '${target}' is not a target")
    endif()

    # Optional mode: THIN (default for Clang) or FULL
    set(_mode "${ARGN}")
    string(TOLOWER "${_mode}" _mode)
    if(_mode STREQUAL "full")
        set(_clang_lto "full")
    else()
        set(_clang_lto "thin")
    endif()

    # Turn on IPO for the target itself (lets CMake add the right flags for that target)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)

    # Figure out target type (to decide whether to propagate to consumers)
    get_target_property(_ttype ${target} TYPE)
    set(_is_lib FALSE)
    if(
        _ttype STREQUAL "STATIC_LIBRARY"
        OR _ttype STREQUAL "SHARED_LIBRARY"
        OR _ttype STREQUAL "MODULE_LIBRARY"
        OR _ttype STREQUAL "OBJECT_LIBRARY"
        OR _ttype STREQUAL "INTERFACE_LIBRARY"
    )
        set(_is_lib TRUE)
    endif()

    # Helper generator expressions
    set(_C_or_CXX "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:")
    set(_CUDA "$<$<COMPILE_LANGUAGE:CUDA>:")
    set(_cfg "$<$<NOT:$<CONFIG:Debug>>:")

    if(MSVC)
        # MSVC: /GL for compile, /LTCG for link; disable incremental linking with LTCG
        if(NOT _ttype STREQUAL "INTERFACE_LIBRARY")
            target_compile_options(${target} PRIVATE "${_C_or_CXX}${_cfg}/GL>>")
            target_link_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/LTCG /INCREMENTAL:NO>)
        endif()
        if(_is_lib)
            target_compile_options(${target} INTERFACE "${_C_or_CXX}${_cfg}/GL>>")
            target_link_options(${target} INTERFACE $<$<NOT:$<CONFIG:Debug>>:/LTCG /INCREMENTAL:NO>)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Clang/AppleClang: prefer ThinLTO by default; lld on non-Apple
        set(_clang_flag "-flto=${_clang_lto}")

        if(NOT _ttype STREQUAL "INTERFACE_LIBRARY")
            target_compile_options(${target} PRIVATE "${_C_or_CXX}${_cfg}${_clang_flag}>>")
            if(APPLE)
                target_link_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:${_clang_flag}>)
            else()
                target_link_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:${_clang_flag} -fuse-ld=lld>)
            endif()
        endif()

        if(_is_lib)
            target_compile_options(${target} INTERFACE "${_C_or_CXX}${_cfg}${_clang_flag}>>")
            if(APPLE)
                target_link_options(${target} INTERFACE $<$<NOT:$<CONFIG:Debug>>:${_clang_flag}>)
            else()
                target_link_options(${target} INTERFACE $<$<NOT:$<CONFIG:Debug>>:${_clang_flag} -fuse-ld=lld>)
            endif()
        endif()

        # Prefer llvm-ar/ranlib if available
        find_program(LLVM_AR NAMES llvm-ar)
        find_program(LLVM_RANLIB NAMES llvm-ranlib)
        if(LLVM_AR AND LLVM_RANLIB)
            set(CMAKE_AR "${LLVM_AR}" PARENT_SCOPE)
            set(CMAKE_RANLIB "${LLVM_RANLIB}" PARENT_SCOPE)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # GCC: -flto; relies on binutils LTO plugin
        if(NOT _ttype STREQUAL "INTERFACE_LIBRARY")
            target_compile_options(${target} PRIVATE "${_C_or_CXX}${_cfg}-flto>>")
            target_link_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-flto>)
        endif()
        if(_is_lib)
            target_compile_options(${target} INTERFACE "${_C_or_CXX}${_cfg}-flto>>")
            target_link_options(${target} INTERFACE $<$<NOT:$<CONFIG:Debug>>:-flto>)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVIDIA")
        # NVCC: use -dlto for device link-time optimization (CUDA >=11.2)
        if(NOT _ttype STREQUAL "INTERFACE_LIBRARY")
            target_compile_options(${target} PRIVATE "${_CUDA}${_cfg}-dlto>>")
            target_link_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-dlto>)
        endif()
        if(_is_lib)
            target_compile_options(${target} INTERFACE "${_CUDA}${_cfg}-dlto>>")
            target_link_options(${target} INTERFACE $<$<NOT:$<CONFIG:Debug>>:-dlto>)
        endif()
    else()
        message(
            WARNING
            "enable_lto(): unknown compiler '${CMAKE_CXX_COMPILER_ID}'. IPO enabled for '${target}', but flags may not propagate."
        )
    endif()
endfunction()
