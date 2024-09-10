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
        CACHE STRING "" FORCE)
  elseif(RUN_OUTPUT MATCHES "AVX2")
    set(FINUFFT_ARCH_FLAGS
        "/arch:AVX2"
        CACHE STRING "" FORCE)
  elseif(RUN_OUTPUT MATCHES "AVX")
    set(FINUFFT_ARCH_FLAGS
        "/arch:AVX"
        CACHE STRING "" FORCE)
  elseif(RUN_OUTPUT MATCHES "SSE")
    set(FINUFFT_ARCH_FLAGS
        "/arch:SSE"
        CACHE STRING "" FORCE)
  else()
    set(FINUFFT_ARCH_FLAGS
        ""
        CACHE STRING "" FORCE)
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
    message(
      STATUS
        "Copying ${source_target} to ${DESTINATION_DIR} directory for ${destination_target}"
    )
    # Define the custom command to copy the source target to the destination
    # directory
    add_custom_command(
      TARGET ${destination_target}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${source_target}>
              ${DESTINATION_FILE}
      COMMENT "Copying ${source_target} to ${destination_target} directory")
  endif()
  # Unset the variables to leave a clean state
  unset(DESTINATION_DIR)
  unset(SOURCE_FILE)
  unset(DESTINATION_FILE)
endfunction()
