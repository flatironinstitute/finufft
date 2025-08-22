# cmake/cuda_setup

include_guard(GLOBAL)

function(detect_cuda_architecture)
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE compute_caps_raw
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(compute_caps_raw)
            string(REPLACE "\n" ";" compute_caps "${compute_caps_raw}")
            set(max_arch_num 0)
            foreach(cc ${compute_caps})
                string(STRIP "${cc}" cc_s)
                if(cc_s MATCHES "^[0-9]+\\.[0-9]+$")
                    string(REPLACE "." "" arch_digit "${cc_s}") # 8.9 -> 89
                    if(arch_digit GREATER max_arch_num)
                        set(max_arch_num ${arch_digit})
                    endif()
                endif()
            endforeach()
            if(max_arch_num GREATER 0)
                message(STATUS "Detected CUDA compute capability: sm_${max_arch_num}")
                # Write to CACHE so it "sticks" for this configure and future runs
                set(CMAKE_CUDA_ARCHITECTURES "${max_arch_num}" CACHE STRING "CUDA SMs" FORCE)
                return()
            endif()
        endif()
        message(
            WARNING
            "Could not parse compute capability from nvidia-smi output '${compute_caps_raw}'. Using 'native'."
        )
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA SMs" FORCE)
    else()
        message(WARNING "nvidia-smi not found. Using 'native'.")
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA SMs" FORCE)
    endif()
endfunction()

# Only detect if user didn't supply it on the command line/preset
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    detect_cuda_architecture()
else()
    message(STATUS "Using user-specified CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Now it's safe to enable CUDA / find the toolkit
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

include(setupCCCL)
