# Build and install finufft from this checkout, then build and run each
# quick-start recipe against it: the find_package recipe consumes the staged
# install, the FetchContent recipe the checkout itself, and with WITH_CUDA=ON
# the CUDA recipe the staged cuFINUFFT. Runs on Linux, macOS and Windows.
# Invoke from the checkout root:
#   cmake -P tools/ci/install-test.cmake
#
# Env inputs and defaults: LINKING=Static BACKEND=ducc OPENMP=ON
# BUILD_TYPE=Release WITH_CUDA=OFF CUDA_ARCH=native. CUDA_ARCH sets the
# library's CMAKE_CUDA_ARCHITECTURES; the CUDA recipe always builds for all
# architectures, and its run needs a card.

set(checkout "${CMAKE_CURRENT_SOURCE_DIR}")
set(stage "${checkout}/_stage")

foreach(
    d
    _build
    _stage
    _consume
    _fetch
    _consume_cuda
    _leak
)
    file(REMOVE_RECURSE "${checkout}/${d}")
endforeach()

macro(env_default var default)
    if("$ENV{${var}}" STREQUAL "")
        set(${var} "${default}")
    else()
        set(${var} "$ENV{${var}}")
    endif()
endmacro()
env_default(LINKING Static)
env_default(BACKEND ducc)
env_default(OPENMP ON)
env_default(BUILD_TYPE Release)
env_default(WITH_CUDA OFF)

set(static ON)
if(LINKING STREQUAL "Shared")
    set(static OFF)
endif()
set(ducc ON)
if(BACKEND MATCHES "^fftw")
    set(ducc OFF)
endif()
set(install_flags -DFINUFFT_USE_DUCC0=${ducc} -DFINUFFT_STATIC_LINKING=${static} -DFINUFFT_USE_OPENMP=${OPENMP})
# clang-cl warns about the GNU -O3 in the shared Release flag list, and
# warnings-as-errors makes that fatal, so this arm runs without it.
if(CMAKE_HOST_WIN32 AND "$ENV{CC}" MATCHES "clang-cl")
    list(APPEND install_flags -DFINUFFT_WARNINGS_AS_ERRORS=OFF)
endif()
set(fftw_download_opt -DFINUFFT_FFTW_LIBRARIES=DOWNLOAD)
# fftw-dl forces the downloaded FFTW: exercises the static-bundle install route.
if(BACKEND STREQUAL "fftw-dl")
    list(APPEND install_flags ${fftw_download_opt})
endif()

set(cuda_flags)
if(WITH_CUDA STREQUAL "ON")
    env_default(CUDA_ARCH native)
    set(cuda_flags -DFINUFFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH})
endif()

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
        string(REPLACE ";" " " cmd "${ARGN}")
        message(FATAL_ERROR "FAILED (${rc}): ${cmd}")
    endif()
endfunction()

function(run_app build_dir)
    foreach(app app app.exe ${BUILD_TYPE}/app ${BUILD_TYPE}/app.exe)
        if(EXISTS "${checkout}/${build_dir}/${app}")
            # Windows has no RPATH: the app finds only its own dir and PATH,
            # while the recipe's own DLL sits elsewhere in its build tree.
            set(saved_loader "$ENV{PATH}")
            if(CMAKE_HOST_WIN32)
                file(GLOB_RECURSE app_dlls "${checkout}/${build_dir}/*.dll")
                set(dll_dirs "")
                foreach(dll IN LISTS app_dlls)
                    get_filename_component(dll_dir "${dll}" DIRECTORY)
                    list(APPEND dll_dirs "${dll_dir}")
                endforeach()
                list(REMOVE_DUPLICATES dll_dirs)
                list(JOIN dll_dirs ";" dll_path)
                if(dll_path)
                    set(ENV{PATH} "${dll_path};$ENV{PATH}")
                endif()
            endif()
            run("${checkout}/${build_dir}/${app}")
            if(CMAKE_HOST_WIN32)
                set(ENV{PATH} "${saved_loader}")
            endif()
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "ERROR: ${build_dir} built no executable")
endfunction()

# Run a consumer with the staged install on the library search path, scoped to
# this run only: the FetchContent recipe builds its own library and must not
# resolve symbols against the staged one.
macro(run_staged build_dir)
    if(CMAKE_HOST_WIN32)
        set(saved_loader "$ENV{PATH}")
        set(ENV{PATH} "${stage}/bin;$ENV{PATH}")
    else()
        set(saved_loader "$ENV{LD_LIBRARY_PATH}")
        set(saved_dyld "$ENV{DYLD_LIBRARY_PATH}")
        set(ENV{LD_LIBRARY_PATH} "${stage}/lib:${stage}/lib64:$ENV{LD_LIBRARY_PATH}")
        set(ENV{DYLD_LIBRARY_PATH} "${stage}/lib:$ENV{DYLD_LIBRARY_PATH}")
    endif()
    run_app("${build_dir}")
    if(CMAKE_HOST_WIN32)
        set(ENV{PATH} "${saved_loader}")
    else()
        set(ENV{LD_LIBRARY_PATH} "${saved_loader}")
        set(ENV{DYLD_LIBRARY_PATH} "${saved_dyld}")
    endif()
endmacro()

# The exported interface must not bake in build-machine absolute paths. The
# patterns cover POSIX roots and the checkout; the regex below covers Windows
# paths: drive-qualified (C:/ or C:\), or UNC (// or \\) at a path start.
set(leak_patterns
    "${checkout}"
    "/usr/"
    "/opt/"
    "/home/"
    "/Users/"
)
set(leak_regex "[A-Za-z]:[/\\\\]|[\";]//|[\";]\\\\\\\\")

# Collect every exported-targets line that names such a path.
function(build_path_hits root out_var)
    file(GLOB targets "${root}/lib*/cmake/finufft/finufftTargets*.cmake")
    if(NOT targets)
        message(FATAL_ERROR "ERROR: no exported targets file under ${root}, so the check proves nothing")
    endif()
    set(hits "")
    foreach(target IN LISTS targets)
        file(STRINGS "${target}" lines)
        foreach(line IN LISTS lines)
            set(hit 0)
            foreach(pattern IN LISTS leak_patterns)
                string(FIND "${line}" "${pattern}" pos)
                if(NOT pos EQUAL -1)
                    set(hit 1)
                    break()
                endif()
            endforeach()
            if(NOT hit AND line MATCHES "${leak_regex}")
                set(hit 1)
            endif()
            if(hit)
                string(APPEND hits "${line}\n")
            endif()
        endforeach()
    endforeach()
    set(${out_var} "${hits}" PARENT_SCOPE)
endfunction()

run(${CMAKE_COMMAND} -S "${checkout}" -B "${checkout}/_build"
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DFINUFFT_ENABLE_INSTALL=ON
    -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded ${install_flags} ${cuda_flags}
)
run(${CMAKE_COMMAND} --build "${checkout}/_build" --config ${BUILD_TYPE})
run(${CMAKE_COMMAND} --install "${checkout}/_build" --prefix "${stage}"
    --config ${BUILD_TYPE}
)

build_path_hits("${stage}" hits)
if(hits)
    message(FATAL_ERROR "ERROR: a build-machine path leaked into exported finufftTargets:\n${hits}")
endif()

# An injected leak must make the check fire, so a green run above means the
# check ran, not that the paths happen to be absent.
run(${CMAKE_COMMAND} -E copy_directory "${stage}" "${checkout}/_leak")
file(GLOB leak_files "${checkout}/_leak/lib*/cmake/finufft/finufftTargets.cmake")
list(GET leak_files 0 leak_file)
# Windows negative controls: one drive-qualified, one UNC.
set(markers
    "/usr/lib/libfftw3.so"
    "${checkout}/libfinufft.a"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64/cudart.lib"
    "//ci-share/cuda/v12.8/lib/x64/cudart.lib"
)
foreach(marker IN LISTS markers)
    file(APPEND "${leak_file}" "set(FINUFFT_LEAK_CONTROL \"${marker}\")\n")
endforeach()
build_path_hits("${checkout}/_leak" hits)
if(NOT hits)
    message(FATAL_ERROR "ERROR: the build-machine path check does not fire on an injected leak")
endif()
foreach(marker IN LISTS markers)
    string(FIND "${hits}" "${marker}" pos)
    if(pos EQUAL -1)
        message(FATAL_ERROR "ERROR: the build-machine path check does not fire on the injected ${marker}")
    endif()
endforeach()
file(REMOVE_RECURSE "${checkout}/_leak")

set(exported OFF)
file(GLOB export_files "${stage}/lib*/cmake/finufft/finufftTargets.cmake")
foreach(export_file IN LISTS export_files)
    file(READ "${export_file}" content)
    string(FIND "${content}" "OpenMP::" pos)
    if(NOT pos EQUAL -1)
        set(exported ON)
    endif()
endforeach()
if(NOT exported STREQUAL OPENMP)
    message(FATAL_ERROR "ERROR: built with FINUFFT_USE_OPENMP=${OPENMP}, but the export says ${exported}")
endif()

# The documented recipes double as consume tests: the find_package one against
# the staged install, the FetchContent one from source.
run(${CMAKE_COMMAND} -S "${checkout}/examples/quick-start/find_package"
    -B "${checkout}/_consume" -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
    -DCMAKE_PREFIX_PATH=${stage} -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
)
run(${CMAKE_COMMAND} --build "${checkout}/_consume" --config ${BUILD_TYPE})
run_staged(_consume)

# The _build pass above covers the FFTW download route, so this saves a
# second FFTW build. On branch jenkins-fix the fftw-dl arm falls from
# 11:00 (builds 4 and 6) to 7:12 (build 9).
list(FIND install_flags ${fftw_download_opt} fftw_download_flag)
set(fetch_fftw_default)
if(NOT fftw_download_flag EQUAL -1)
    set(fetch_fftw_default -DFINUFFT_FFTW_LIBRARIES=DEFAULT)
endif()

run(${CMAKE_COMMAND} -S "${checkout}/examples/quick-start/fetchcontent"
    -B "${checkout}/_fetch" -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
    -DFETCHCONTENT_SOURCE_DIR_FINUFFT=${checkout}
    -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded ${install_flags}
    ${fetch_fftw_default}
)
run(${CMAKE_COMMAND} --build "${checkout}/_fetch" --config ${BUILD_TYPE})
run_app(_fetch)

# The GPU recipe consumes the staged cuFINUFFT with find_package and runs on
# the card.
if(WITH_CUDA STREQUAL "ON")
    run(${CMAKE_COMMAND} -S "${checkout}/examples/quick-start/cuda"
        -B "${checkout}/_consume_cuda" -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
      -DCMAKE_PREFIX_PATH=${stage}
      -DCMAKE_CUDA_ARCHITECTURES=all
      -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
    )
    run(${CMAKE_COMMAND} --build "${checkout}/_consume_cuda" --config ${BUILD_TYPE})
    run_staged(_consume_cuda)
endif()

message(STATUS "install-test.cmake: every recipe built and ran")
