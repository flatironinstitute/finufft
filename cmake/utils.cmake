include(CheckCXXCompilerFlag)
# Define the function
function(filter_supported_compiler_flags input_flags_var output_flags_var)
  # Create an empty list to store supported flags
  set(supported_flags)

  # Iterate over each flag in the input list
  foreach(flag ${${input_flags_var}})
    string(REPLACE "=" "_" flag_var ${flag}) # Convert flag to a valid variable
                                             # name
    string(REPLACE "-" "" flag_var ${flag_var}) # Remove '-' for the variable
                                                # name

    # Append the test linker flag to the existing flags
    list(APPEND CMAKE_EXE_LINKER_FLAGS ${flag})
    check_cxx_compiler_flag(${flag} ${flag_var})
    if(${flag_var})
      # If supported, append the flag to the list of supported flags
      list(APPEND supported_flags ${flag})
    else()
      message(STATUS "Flag ${flag} is not supported")
    endif()
    unset(${flag_var} CACHE)
    # remove last flag from linker flags
    list(REMOVE_ITEM CMAKE_EXE_LINKER_FLAGS ${flag})
  endforeach()
  # Set the output variable to the list of supported flags
  set(${output_flags_var}
      ${supported_flags}
      PARENT_SCOPE)
endfunction()

function(check_arch_support)
  message(STATUS "Checking for AVX, AVX512 and SSE support")
  try_run(
    RUN_RESULT_VAR COMPILE_RESULT_VAR ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckAVX.cpp
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE RUN_OUTPUT)
  if(RUN_OUTPUT MATCHES "AVX512")
    set(FINUFFT_ARCH_FLAGS
        "/arch:AVX512"
        CACHE STRING "Compiler flags for specifying target architecture.")
  elseif(RUN_OUTPUT MATCHES "AVX")
    set(FINUFFT_ARCH_FLAGS
        "/arch:AVX"
        CACHE STRING "Compiler flags for specifying target architecture.")
  elseif(RUN_OUTPUT MATCHES "SSE")
    set(FINUFFT_ARCH_FLAGS
        "/arch:SSE"
        CACHE STRING "Compiler flags for specifying target architecture.")
  else()
    set(FINUFFT_ARCH_FLAGS
        ""
        CACHE STRING "Compiler flags for specifying target architecture.")
  endif()
  message(STATUS "CPU supports: ${RUN_OUTPUT}")
  message(STATUS "Using MSVC flags: ${FINUFFT_ARCH_FLAGS}")
endfunction()
