# Static analysis wiring (per-target, opt-in).
#
# Usage from any CMakeLists.txt:
#
#     enable_static_analysis(<target>)
#
# Sets the target's CXX_CLANG_TIDY and CXX_CPPCHECK properties when
# FINUFFT_STATIC_ANALYSIS=ON and the corresponding tools are found.
# Per-target wiring keeps analysis off CPM/FetchContent dependency
# targets regardless of include/subdirectory order.
#
# Check selection, header-filter, and per-check options live in the
# project-root `.clang-tidy` so they are the single source of truth
# for both the compile-time checker and any direct `clang-tidy`
# invocation. cppcheck has no equivalent config file, so its options
# are passed inline here.

if(NOT FINUFFT_STATIC_ANALYSIS)
    function(enable_static_analysis)
    endfunction()
    return()
endif()

set(_CLANG_TIDY_CANDIDATES)
foreach(ver RANGE 22 16 -1)
    list(APPEND _CLANG_TIDY_CANDIDATES clang-tidy-${ver})
endforeach()
list(APPEND _CLANG_TIDY_CANDIDATES clang-tidy)
find_program(CLANG_TIDY_EXE NAMES ${_CLANG_TIDY_CANDIDATES})
find_program(CPPCHECK_EXE NAMES cppcheck)
mark_as_advanced(CLANG_TIDY_EXE CPPCHECK_EXE)

set(_CPPCHECK_ARGS
    # `style` is intentionally not enabled: it's high-noise (no-explicit-ctor,
    # known-condition, redundant-init etc.) and not what we want to gate CI on.
    # Re-enable locally with -DFINUFFT_STATIC_ANALYSIS_STYLE=ON if desired.
    --enable=warning,performance,portability
    --inline-suppr
    --suppress=missingIncludeSystem
    # cppcheck has no header-filter; mute findings from third-party
    # headers reached transitively through includes. Cover both the
    # default user-cache path (`~/.cpm/`) and the in-tree CPM cache
    # (`<repo>/cpm/`) used by some CI workflows.
    --suppress=*:*/.cpm/*
    --suppress=*:*/cpm/*
    --suppress=*:*/_deps/*
    --suppress=*:*/external/*
    --suppress=*:*/third_party/*
)

set(_CLANG_TIDY_INVOCATION "${CLANG_TIDY_EXE}")
if(FINUFFT_STATIC_ANALYSIS_WERROR)
    # Gate clang-tidy on the include-cleaner check only. `*` would also
    # escalate `clang-diagnostic-*` (compiler diagnostics surfaced through
    # tidy), which collides with the project's own deprecated finufft_opts
    # fields. Cppcheck stays informational — its existing findings are
    # mostly pre-existing portability/style nits and not part of the
    # include-hygiene gate.
    list(APPEND _CLANG_TIDY_INVOCATION --warnings-as-errors=misc-include-cleaner)
endif()

function(enable_static_analysis tgt)
    if(NOT TARGET ${tgt})
        message(FATAL_ERROR "enable_static_analysis: target '${tgt}' not found")
    endif()
    get_target_property(_type ${tgt} TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY")
        return()
    endif()
    if(CLANG_TIDY_EXE)
        set_property(TARGET ${tgt} PROPERTY CXX_CLANG_TIDY "${_CLANG_TIDY_INVOCATION}")
    endif()
    if(CPPCHECK_EXE)
        set_property(TARGET ${tgt} PROPERTY CXX_CPPCHECK "${CPPCHECK_EXE};${_CPPCHECK_ARGS}")
    endif()
endfunction()
