#!/bin/bash

# Install FINUFFT to a staging prefix and consume it the three ways a user does:
# find_package against the install, FetchContent against the sources, and a bare
# compiler line against the installed headers and library.
#
# One script for GitHub and for Jenkins. GitHub runs it on Windows, where no
# Jenkins agent carries a toolchain; Jenkins runs the Linux arms, and is the only
# place the CUDA consumer can be run rather than only linked, because its pods
# have a device.
#
# Environment:
#   LINKING   Static (default) or Shared
#   BACKEND   ducc (default) or fftw
#   CUDA      1 to install and consume cufinufft instead of the CPU library
#   CUDA_ARCH the pod's compute capability, required when CUDA=1
#   CONTROLS  1 to also run the two FFTW positive controls (Linux, fftw only)
set -euo pipefail

linking=${LINKING:-Static}
backend=${BACKEND:-ducc}
cuda=${CUDA:-0}
stage="$PWD/_stage"

static=ON
[[ "$linking" == "Shared" ]] && static=OFF
ducc=ON
[[ "$backend" == "fftw" ]] && ducc=OFF

if [[ "$cuda" == "1" ]]; then
	: "${CUDA_ARCH:?CUDA=1 needs CUDA_ARCH; a default would silently build for the wrong card}"
	consumer=test/cmake_consume/cuda
	# CPU off on purpose: this is the CUDA-only install layout that used to ship
	# cufinufft.h without the headers it includes.
	install_flags=(-DFINUFFT_USE_CUDA=ON -DFINUFFT_USE_CPU=OFF
		-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
else
	consumer=test/cmake_consume
	install_flags=(-DFINUFFT_USE_DUCC0=$ducc -DFINUFFT_STATIC_LINKING=$static)
fi

cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release \
	-DFINUFFT_ENABLE_INSTALL=ON \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
	"${install_flags[@]}"
cmake --build _build --config Release
cmake --install _build --prefix "$stage" --config Release

# The exported interface must not bake in build-machine absolute paths.
if grep -R "/usr/" "$stage"/lib*/cmake/finufft/finufftTargets*.cmake; then
	echo "ERROR: absolute path leaked into exported finufftTargets"
	exit 1
fi

cmake -S "$consumer" -B _consume -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_PREFIX_PATH="$stage" \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
cmake --build _consume --config Release

# A shared install leaves the loader to find the library; a Windows build puts
# the executable in a per-config subdirectory.
if [[ "${RUNNER_OS:-}" == "Windows" ]]; then
	app=_consume/app.exe
	[[ -x $app ]] || app=_consume/Release/app.exe
else
	app=_consume/app
	export LD_LIBRARY_PATH="$stage/lib:$stage/lib64:${LD_LIBRARY_PATH:-}"
	export DYLD_LIBRARY_PATH="$stage/lib:${DYLD_LIBRARY_PATH:-}"
fi
"$app"

# Second route: FetchContent/add_subdirectory, which builds FINUFFT as a
# subproject rather than against an install. It is what the CPM and FetchContent
# recipes in docs/install.rst do, and the path that breaks when a
# top-level-only guard (CTest, docs targets, install rules) is missing. The
# route depends on neither linkage nor backend, so one CPU arm covers it.
if [[ "$cuda" == "1" || ("$linking" == "Static" && "$backend" == "ducc") ]]; then
	cmake -S "$consumer/fetchcontent" -B _fetch -DCMAKE_BUILD_TYPE=Release \
		-DFINUFFT_SOURCE_DIR="$PWD" \
		-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
		"${install_flags[@]}"
	cmake --build _fetch --config Release
	if [[ "${RUNNER_OS:-}" == "Windows" ]]; then
		app=_fetch/app.exe
		[[ -x $app ]] || app=_fetch/Release/app.exe
	else
		app=_fetch/app
	fi
	"$app"
fi

# Third route: no CMake at all. A shared install has to be usable from a plain
# compiler line, which is what a hand-written Makefile, a ctypes load or a Julia
# ccall ends up doing, and it is the only route that reads the installed headers
# and the library without the exported target in between.
#
# Static is excluded on purpose rather than skipped for convenience: a static
# libfinufft leaves its FFT and OpenMP dependencies to the consumer, and the
# exported CMake target is the only thing that knows what they are. Windows is
# excluded because cl takes none of these flags.
if [[ "$cuda" == "0" && "$linking" == "Shared" && "${RUNNER_OS:-}" != "Windows" ]]; then
	libdir=$stage/lib
	[[ -d "$libdir" ]] || libdir=$stage/lib64
	"${CXX:-c++}" -std=c++17 -O2 test/cmake_consume/main.cpp \
		-I"$stage/include" -L"$libdir" -lfinufft -Wl,-rpath,"$libdir" -o _plain_app
	./_plain_app
fi

[[ "${CONTROLS:-0}" == "1" ]] || exit 0

if [[ "$cuda" == "1" ]]; then
	# The guard on the guard. Empty the exported link interface in a copy of the
	# install and require the consumer to stop linking: main.cpp calls cudaMalloc
	# and cudaMemcpy itself, so the symbols can only come from CUDA::cudart. A
	# check that has never failed cannot be told from one that cannot fail.
	rm -rf _broken _broken_consume
	cp -a _stage _broken
	sed -i 's/CUDA::cudart;CUDA::cufft//' _broken/lib*/cmake/finufft/finufftTargets.cmake
	if cmake -S "$consumer" -B _broken_consume -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_PREFIX_PATH="$PWD/_broken" >broken.log 2>&1 &&
		cmake --build _broken_consume >>broken.log 2>&1; then
		echo "ERROR: consumer built against an empty link interface, so the guard proves nothing"
		exit 1
	fi
	# mold says "undefined symbol: cudaMemcpy", GNU ld "undefined reference to \`cudaMalloc'".
	grep -qE "undefined (reference to|symbol)[ :]*.?cuda" broken.log ||
		{
			echo "ERROR: the consumer failed for some other reason than the empty link interface"
			tail -30 broken.log
			exit 1
		}
	exit 0
fi

# A package manager that forbids downloads (vcpkg passes
# -DFETCHCONTENT_FULLY_DISCONNECTED=ON) used to get an unrelated "install
# TARGETS given target fftw3 which does not exist" at the end of the configure.
# setupFFTW.cmake now says so where it happens; this is the control proving the
# message still appears.
if cmake -S . -B _nofetch -DFINUFFT_USE_DUCC0=OFF \
	-DFINUFFT_FFTW_LIBRARIES=DOWNLOAD \
	-DFETCHCONTENT_FULLY_DISCONNECTED=ON >nofetch.log 2>&1; then
	echo "ERROR: configure succeeded with fetching disabled"
	exit 1
fi
grep -q "FINUFFT could not fetch FFTW" nofetch.log ||
	{
		echo "ERROR: guard message missing"
		tail -30 nofetch.log
		exit 1
	}

# A hand-supplied FFTW cannot be exported, so a static install leaves the
# consumer to link it. The DEFAULT configure is the negative control: it exports
# its FFTW and must stay silent.
cmake -S . -B _userfftw -DFINUFFT_USE_DUCC0=OFF -DFINUFFT_STATIC_LINKING=ON \
	-DFINUFFT_FFTW_LIBRARIES="fftw3;fftw3f" >userfftw.log 2>&1
grep -q "is supplied by hand" userfftw.log ||
	{
		echo "ERROR: warning missing"
		tail -30 userfftw.log
		exit 1
	}
cmake -S . -B _deffftw -DFINUFFT_USE_DUCC0=OFF -DFINUFFT_STATIC_LINKING=ON \
	>deffftw.log 2>&1
if grep -q "is supplied by hand" deffftw.log; then
	echo "ERROR: DEFAULT warned as well, so the check proves nothing"
	exit 1
fi
