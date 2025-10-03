# ---- Optional: tweak defaults without editing code (cache variables) -------
set(ANALYSIS_HEADER_FILTER
    "^${PROJECT_SOURCE_DIR}/(src|include)/.*"
    CACHE STRING
    "Regex for clang-tidy -header-filter (project files)"
)
set(ANALYSIS_CPPCHECK_SUPPRESS_DIRS
    "${PROJECT_SOURCE_DIR}/external;${PROJECT_SOURCE_DIR}/third_party"
    CACHE STRING
    "Semicolon-separated dirs to suppress in cppcheck"
)

# ---- Find newest clang-tidy ------------------------------------------------
set(_CLANG_TIDY_CANDIDATES)
foreach(ver RANGE 22 10 -1)
    list(APPEND _CLANG_TIDY_CANDIDATES clang-tidy-${ver})
endforeach()
list(APPEND _CLANG_TIDY_CANDIDATES clang-tidy)
find_program(CLANG_TIDY_EXE NAMES ${_CLANG_TIDY_CANDIDATES})

# ---- Find cppcheck (unversioned only) --------------------------------------
find_program(CPPCHECK_EXE NAMES cppcheck)

# ---- Internal helper: detect if target has any C++ sources -----------------
function(_target_has_cxx out_var tgt)
    get_target_property(_srcs ${tgt} SOURCES)
    set(_has FALSE)
    foreach(s IN LISTS _srcs)
        get_source_file_property(_lang "${s}" LANGUAGE)
        if(_lang STREQUAL "CXX")
            set(_has TRUE)
            break()
        endif()
        if(NOT _lang OR _lang STREQUAL "NONE")
            if(s MATCHES "\\.(cc|cpp|cxx|C)($|\\.)")
                set(_has TRUE)
                break()
            endif()
        endif()
    endforeach()
    set(${out_var} ${_has} PARENT_SCOPE)
endfunction()

# ---- Internal helper: compute C++ standard for target ----------------------
function(_detect_cxx_standard out_var tgt)
    get_target_property(_std ${tgt} CXX_STANDARD)
    if(NOT _std)
        if(DEFINED CMAKE_CXX_STANDARD)
            set(_std ${CMAKE_CXX_STANDARD})
        else()
            set(_std 17)
        endif()
    endif()
    set(${out_var} ${_std} PARENT_SCOPE)
endfunction()

# ---- Public: enable analysis for a single target ---------------------------
function(enable_static_analysis tgt)
    if(NOT FINUFFT_STATIC_ANALYSIS)
        return()
    endif()

    if(NOT TARGET ${tgt})
        message(FATAL_ERROR "FINUFFT_STATIC_ANALYSIS_for_target: target '${tgt}' not found")
    endif()

    # Skip header-only libs
    get_target_property(_type ${tgt} TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY")
        return()
    endif()

    # Skip targets without C++ sources
    _target_has_cxx(_has ${tgt})
    if(NOT _has)
        return()
    endif()

    # Determine target C++ standard
    _detect_cxx_standard(_std ${tgt})

    # ----- clang-tidy per-target -------------------------------------------
    if(CLANG_TIDY_EXE)
        set(_TIDY_ARGS -checks=*, -header-filter=${ANALYSIS_HEADER_FILTER}, --extra-arg=-std=c++${_std})
        set_property(TARGET ${tgt} PROPERTY CXX_CLANG_TIDY "${CLANG_TIDY_EXE};${_TIDY_ARGS}")
    endif()

    # ----- cppcheck per-target ---------------------------------------------
    if(CPPCHECK_EXE)
        set(_CPPCHECK_ARGS
            --enable=warning,style,performance,portability
            --inconclusive
            --inline-suppr
            --suppress=missingIncludeSystem
            --std=c++${_std}
        )

        foreach(d IN LISTS ANALYSIS_CPPCHECK_SUPPRESS_DIRS)
            if(EXISTS "${d}")
                list(APPEND _CPPCHECK_ARGS "--suppress=*:${d}/*")
            endif()
        endforeach()

        set_property(TARGET ${tgt} PROPERTY CXX_CPPCHECK "${CPPCHECK_EXE};${_CPPCHECK_ARGS}")
    endif()
endfunction()

# ---- Public: enable analysis for all targets in this directory -------------
function(FINUFFT_STATIC_ANALYSIS_here)
    if(NOT FINUFFT_STATIC_ANALYSIS)
        return()
    endif()
    get_property(_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(t IN LISTS _targets)
        finufft_static_analysis_for_target(${t})
    endforeach()
endfunction()
