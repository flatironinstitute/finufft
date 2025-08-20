# IWYU integration helpers
# - Adds a custom target `iwyu-<tgt>` that:
#   1) Runs iwyu_tool.py over <tgt>'s sources using compile_commands.json
#   2) Optionally pipes into fix_includes.py to auto-apply fixes
#   3) Applies fixes ONLY under src/, include/, test/ (default)
#
# Extras:
# - IWYU_EXCLUDE_FILES: skip certain files entirely (analysis + auto-fix)
#   (Default excludes: src/fft.cpp and include/finufft/fft.h)
# - IWYU_MAPPING_FILES: semicolon-separated .imp files (auto-added if they exist)
#
# Usage in CMakeLists.txt:
#   add_iwyu_fix_target(my_target)

# ----------------------------------------------------------------------------- #
# User-tunable cache variables
# ----------------------------------------------------------------------------- #
set(IWYU_ONLY_REGEX
    "^${PROJECT_SOURCE_DIR}/(src|include|test|examples)/.*"
    CACHE STRING
    "Regex of files to APPLY FIXES to (fix_includes.py --only_re)."
)

set(IWYU_IGNORE_REGEX
    "(external/|third_party/|3rdparty/|vendor/|_deps/|build/|cmake%-build)"
    CACHE STRING
    "Regex of paths to ignore when applying fixes (fix_includes.py --ignore_re)."
)

set(IWYU_EXTRA_ARGS "" CACHE STRING "Extra arguments forwarded to include-what-you-use (space-separated).")

# Files to exclude entirely from IWYU analysis AND auto-fix (semicolon-separated).
# Paths may be absolute or relative to ${PROJECT_SOURCE_DIR}.
set(IWYU_EXCLUDE_FILES
    "src/fft.cpp;include/finufft/fft.h;src/finufft_utils.cpp"
    CACHE STRING
    "Files to exclude from IWYU analysis and auto-fix (semicolon-separated)."
)

# Optional: semicolon-separated list of .imp mapping files to load.
# Example: "${PROJECT_SOURCE_DIR}/iwyu.imp"
set(IWYU_MAPPING_FILES
    "${PROJECT_SOURCE_DIR}/iwyu.imp"
    CACHE STRING
    "Semicolon-separated IWYU mapping files (.imp). Non-existent paths are ignored."
)

mark_as_advanced(
    IWYU_ONLY_REGEX
    IWYU_IGNORE_REGEX
    IWYU_EXTRA_ARGS
    IWYU_EXCLUDE_FILES
    IWYU_MAPPING_FILES
    IWYU_BIN
    IWYU_TOOL
    IWYU_FIX
)

# ----------------------------------------------------------------------------- #
# Find tools (soft — never fail the configure step)
# ----------------------------------------------------------------------------- #
find_program(IWYU_BIN NAMES include-what-you-use iwyu)
find_program(IWYU_TOOL NAMES iwyu_tool.py iwyu_tool)
find_program(IWYU_FIX NAMES fix_includes.py fix_includes)
find_package(Python3 COMPONENTS Interpreter) # no REQUIRED, soft fail with message

# If scripts not found, try beside IWYU_BIN
if(IWYU_BIN AND (NOT IWYU_TOOL OR NOT IWYU_FIX))
    get_filename_component(_IWYU_DIR "${IWYU_BIN}" DIRECTORY)
    if(NOT IWYU_TOOL AND EXISTS "${_IWYU_DIR}/iwyu_tool.py")
        set(IWYU_TOOL "${_IWYU_DIR}/iwyu_tool.py")
    endif()
    if(NOT IWYU_FIX AND EXISTS "${_IWYU_DIR}/fix_includes.py")
        set(IWYU_FIX "${_IWYU_DIR}/fix_includes.py")
    endif()
endif()

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
# Collect C/C++ sources for a target (skip headers and non-C/C++)
function(_iwyu_collect_sources tgt out_var)
    get_target_property(_srcs ${tgt} SOURCES)
    set(_files "")
    foreach(s IN LISTS _srcs)
        get_source_file_property(_lang "${s}" LANGUAGE)
        if(_lang STREQUAL "C" OR _lang STREQUAL "CXX")
            list(APPEND _files "${s}")
        elseif(NOT _lang OR _lang STREQUAL "NONE")
            if(s MATCHES "\\.(c|cc|cp|cxx|cpp|c\\+\\+|C)$")
                list(APPEND _files "${s}")
            endif()
        endif()
    endforeach()
    set(${out_var} "${_files}" PARENT_SCOPE)
endfunction()

# Detect whether fix_includes.py supports --quiet; sets IWYU_FIX_QUIET_ARG.
function(_iwyu_detect_fix_quiet)
    set(IWYU_FIX_QUIET_ARG "" PARENT_SCOPE)
    if(NOT IWYU_FIX OR NOT Python3_Interpreter_FOUND OR NOT Python3_EXECUTABLE)
        return()
    endif()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${IWYU_FIX}" -h
        OUTPUT_VARIABLE _fix_help_out
        ERROR_VARIABLE _fix_help_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    set(_help_text "${_fix_help_out}\n${_fix_help_err}")
    if(_help_text MATCHES "--quiet")
        set(IWYU_FIX_QUIET_ARG "--quiet" PARENT_SCOPE)
    endif()
endfunction()

# Strong health check: try an actual (headerless) compile to trigger ABI issues.
function(_iwyu_check_binary out_ok)
    set(${out_ok} FALSE PARENT_SCOPE)
    if(NOT IWYU_BIN)
        return()
    endif()

    execute_process(COMMAND "${IWYU_BIN}" --help RESULT_VARIABLE _rc_help OUTPUT_QUIET ERROR_QUIET)

    # Probe compile: C++ file with no headers, syntax-only.
    set(_probe_dir "${CMAKE_BINARY_DIR}/_iwyu")
    file(MAKE_DIRECTORY "${_probe_dir}")
    set(_probe_src "${_probe_dir}/__iwyu_probe__.cc")
    file(WRITE "${_probe_src}" "int main(){return 0;}")

    execute_process(
        COMMAND "${IWYU_BIN}" -fsyntax-only -x c++ "${_probe_src}"
        RESULT_VARIABLE _rc_probe
        OUTPUT_VARIABLE _o
        ERROR_VARIABLE _e
    )

    if(_rc_probe EQUAL 0)
        set(${out_ok} TRUE PARENT_SCOPE)
    else()
        string(REPLACE "\n" "\n    " _e_indented "${_e}")
        message(STATUS "[IWYU] '${IWYU_BIN}' failed probe compile (rc=${_rc_probe}).\n    ${_e_indented}")
        set(${out_ok} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Create a shim dir that provides 'include-what-you-use' which execs IWYU_BIN.
# iwyu_tool.py looks up 'include-what-you-use' on PATH; this forces it to use ours.
function(_iwyu_prepare_shim out_dir)
    set(_shim_dir "${CMAKE_BINARY_DIR}/_iwyu/shim")
    if(UNIX AND IWYU_BIN)
        file(MAKE_DIRECTORY "${_shim_dir}")
        set(_shim "${_shim_dir}/include-what-you-use")
        if(NOT EXISTS "${_shim}")
            file(WRITE "${_shim}" "#!/bin/sh\nexec \"${IWYU_BIN}\" \"$@\"\n")
            execute_process(COMMAND /bin/chmod +x "${_shim}")
        endif()
        set(${out_dir} "${_shim_dir}" PARENT_SCOPE)
    else()
        if(IWYU_BIN)
            get_filename_component(_d "${IWYU_BIN}" DIRECTORY)
            set(${out_dir} "${_d}" PARENT_SCOPE)
        else()
            set(${out_dir} "" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ----------------------------------------------------------------------------- #
# Public: add_iwyu_fix_target(<tgt>)
# ----------------------------------------------------------------------------- #
function(add_iwyu_fix_target tgt)
    if(NOT TARGET ${tgt})
        message(STATUS "[IWYU] IWYU not enabled: target '${tgt}' not found")
        return()
    endif()

    get_target_property(_type ${tgt} TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY")
        message(STATUS "[IWYU] Skipping INTERFACE target '${tgt}'")
        return()
    endif()

    # Ensure compile_commands.json exists (iwyu_tool.py requires it)
    set(_CC_JSON "${CMAKE_BINARY_DIR}/compile_commands.json")
    if(NOT EXISTS "${_CC_JSON}")
        message(
            STATUS
            "[IWYU] IWYU not enabled: compile_commands.json not found (enable with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)"
        )
        return()
    endif()

    # Gather and filter sources
    _iwyu_collect_sources(${tgt} _SRC_FILES)

    # Normalize IWYU_EXCLUDE_FILES to absolute paths
    set(_EXCL_ABS "")
    if(IWYU_EXCLUDE_FILES)
        foreach(ex IN LISTS IWYU_EXCLUDE_FILES)
            if(IS_ABSOLUTE "${ex}")
                list(APPEND _EXCL_ABS "${ex}")
            else()
                list(APPEND _EXCL_ABS "${PROJECT_SOURCE_DIR}/${ex}")
            endif()
        endforeach()
    endif()

    # Remove excluded files from analysis
    if(_EXCL_ABS)
        set(_SRC_FILES_ABS "")
        foreach(f IN LISTS _SRC_FILES)
            if(IS_ABSOLUTE "${f}")
                list(APPEND _SRC_FILES_ABS "${f}")
            else()
                list(APPEND _SRC_FILES_ABS "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
            endif()
        endforeach()
        foreach(exabs IN LISTS _EXCL_ABS)
            list(FILTER _SRC_FILES_ABS EXCLUDE REGEX "^${exabs}$")
        endforeach()
        set(_SRC_FILES "${_SRC_FILES_ABS}")
    endif()

    if(_SRC_FILES STREQUAL "")
        message(STATUS "[IWYU] No C/C++ sources for '${tgt}', skipping")
        return()
    endif()

    # Tools availability and health (degrade gracefully, never fail)
    # We need Python for iwyu_tool.py and fix_includes.py.
    if(NOT Python3_Interpreter_FOUND OR NOT Python3_EXECUTABLE)
        message(STATUS "[IWYU] IWYU not enabled: Python3 interpreter not found")
        return()
    endif()

    if(NOT IWYU_BIN)
        message(STATUS "[IWYU] IWYU not enabled: include-what-you-use binary not found")
        return()
    endif()

    if(NOT IWYU_TOOL)
        message(STATUS "[IWYU] IWYU not enabled: iwyu_tool.py not found")
        return()
    endif()

    # Health check the IWYU binary
    _iwyu_check_binary(_ok)
    if(NOT _ok)
        message(STATUS "[IWYU] IWYU not enabled: probe compile failed")
        return()
    endif()

    # We can still run analysis if fix_includes.py is missing.
    set(_have_fix TRUE)
    if(NOT IWYU_FIX)
        set(_have_fix FALSE)
        message(STATUS "[IWYU] fix_includes.py not found: will run analysis only (no auto-fix)")
    endif()

    # Normalize to absolute paths for iwyu_tool.py
    set(_SRC_ARGS "")
    foreach(f IN LISTS _SRC_FILES)
        if(IS_ABSOLUTE "${f}")
            list(APPEND _SRC_ARGS "${f}")
        else()
            list(APPEND _SRC_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
        endif()
    endforeach()

    # Compose extra args (local copy so we can safely append)
    set(_EXTRA_LOCAL "${IWYU_EXTRA_ARGS}")

    # Add any mapping files that exist
    if(IWYU_MAPPING_FILES)
        foreach(_mf IN LISTS IWYU_MAPPING_FILES)
            if(NOT IS_ABSOLUTE "${_mf}")
                set(_mf "${PROJECT_SOURCE_DIR}/${_mf}")
            endif()
            if(EXISTS "${_mf}")
                # Use -Xiwyu so flags reach IWYU proper
                set(_EXTRA_LOCAL "${_EXTRA_LOCAL} -Xiwyu --mapping_file=${_mf}")
            else()
                message(STATUS "[IWYU] Mapping file not found (skipped): ${_mf}")
            endif()
        endforeach()
    endif()

    # Verbosity & split
    set(_VERBOSE "")
    if(FINUFFT_IWYU_VERBOSE)
        set(_VERBOSE "-v")
    endif()
    separate_arguments(_EXTRA NATIVE_COMMAND "${_EXTRA_LOCAL}")
    if(_EXTRA)
        set(_EXTRA_PART " -- $<JOIN:${_EXTRA}, >")
    else()
        set(_EXTRA_PART "")
    endif()

    _iwyu_detect_fix_quiet()
    set(_FIX_QUIET_ARG "${IWYU_FIX_QUIET_ARG}")

    # Logging
    set(_LOG_DIR "${CMAKE_BINARY_DIR}/_iwyu")
    file(MAKE_DIRECTORY "${_LOG_DIR}")
    set(_LOG_FILE "${_LOG_DIR}/iwyu-${tgt}.out")

    # Ensure iwyu_tool.py resolves the correct IWYU via PATH
    _iwyu_prepare_shim(_PATH_FRONT)

    # Base command (force IWYU output format!)
    set(_CMD1
        "${Python3_EXECUTABLE} \"${IWYU_TOOL}\" -p \"${CMAKE_BINARY_DIR}\" -o iwyu $<JOIN:${_SRC_ARGS}, > ${_VERBOSE}${_EXTRA_PART}"
    )

    # Build an ignore regex that also blocks the excluded files from auto-fix
    set(_IWYU_IGNORE_RX "${IWYU_IGNORE_REGEX}")
    if(_EXCL_ABS)
        foreach(ex IN LISTS _EXCL_ABS)
            string(REPLACE "." "\\." _ex_esc "${ex}") # escape '.'
            set(_IWYU_IGNORE_RX "${_IWYU_IGNORE_RX}|^${_ex_esc}$")
        endforeach()
    endif()

    if(WIN32)
        # Escape % for cmd.exe
        set(_IGNORE_ESC "${_IWYU_IGNORE_RX}")
        set(_ONLY_ESC "${IWYU_ONLY_REGEX}")
        string(REPLACE "%" "%%" _IGNORE_ESC "${_IGNORE_ESC}")
        string(REPLACE "%" "%%" _ONLY_ESC "${_ONLY_ESC}")

        if(_have_fix)
            set(_PIPE_CMD
                "${_CMD1} 2>&1 > \"${_LOG_FILE}\" & type \"${_LOG_FILE}\" & (findstr /r \"should .* these lines\" \"${_LOG_FILE}\" >NUL) && (${Python3_EXECUTABLE} \"${IWYU_FIX}\" --nocomments ${_FIX_QUIET_ARG} --ignore_re=${_IGNORE_ESC} --only_re=${_ONLY_ESC} < \"${_LOG_FILE}\") || echo [IWYU] No actionable suggestions (see ${_LOG_FILE})"
            )
        else()
            set(_PIPE_CMD "${_CMD1} 2>&1 > \"${_LOG_FILE}\" & type \"${_LOG_FILE}\"")
        endif()
        set(_SHELL cmd /C)
        set(_ENV_PATH "PATH=${_PATH_FRONT};$ENV{PATH}")
    else()
        if(_have_fix)
            set(_PIPE_CMD
                "${_CMD1} 2>&1 | tee \"${_LOG_FILE}\"; if grep -E -q \"should (add|remove) these lines\" \"${_LOG_FILE}\"; then ${Python3_EXECUTABLE} \"${IWYU_FIX}\" --nocomments ${_FIX_QUIET_ARG} --ignore_re='${_IWYU_IGNORE_RX}' --only_re='${IWYU_ONLY_REGEX}' < \"${_LOG_FILE}\"; else echo \"[IWYU] No actionable suggestions (see ${_LOG_FILE})\"; fi"
            )
        else()
            set(_PIPE_CMD "${_CMD1} 2>&1 | tee \"${_LOG_FILE}\"")
        endif()
        set(_SHELL /bin/sh -lc)
        if(_PATH_FRONT)
            set(_ENV_PATH "PATH=${_PATH_FRONT}:$ENV{PATH}")
        else()
            set(_ENV_PATH "PATH=$ENV{PATH}")
        endif()
    endif()

    # Comment
    set(_COMMENT "[IWYU] include-what-you-use")
    if(_have_fix)
        set(_COMMENT "${_COMMENT} + auto-fix")
    endif()
    set(_COMMENT "${_COMMENT} (only src/include/test) for '${tgt}'")

    add_custom_target(
        iwyu-${tgt}
        COMMAND ${CMAKE_COMMAND} -E env "${_ENV_PATH}" ${_SHELL} "${_PIPE_CMD}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        USES_TERMINAL
        VERBATIM
        COMMENT "${_COMMENT}"
    )

    if(FINUFFT_ENABLE_IWYU AND IWYU_BIN)
        set_property(TARGET ${tgt} PROPERTY CXX_INCLUDE_WHAT_YOU_USE "${IWYU_BIN};${IWYU_EXTRA_ARGS}")
    endif()

    message(STATUS "[IWYU] Added target: iwyu-${tgt}")
endfunction()

mark_as_advanced(
    IWYU_ONLY_REGEX
    IWYU_IGNORE_REGEX
    IWYU_EXTRA_ARGS
    IWYU_EXCLUDE_FILES
    IWYU_MAPPING_FILES
    IWYU_BIN
    IWYU_TOOL
    IWYU_FIX
)
