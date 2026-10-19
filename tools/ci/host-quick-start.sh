#!/bin/sh
# Put cmake on a Jenkins mac host agent that ships Xcode's clang and make but
# no cmake, then run the cross-platform quick-start test. The archive lands in
# CI_TOOLS inside the workspace, so the second test arm skips the download. An
# agent that carries its own cmake is untouched.
#
# Env: CI_TOOLS (install root; its bin/ is on PATH through PATH+CI_TOOLS)
set -eu
: "${CI_TOOLS:?}"
CMAKE_VERSION=${CMAKE_VERSION:-3.31.6}

if ! command -v cmake >/dev/null 2>&1; then
	archive=cmake-${CMAKE_VERSION}-macos-universal
	mkdir -p "$CI_TOOLS/bin"
	if [ ! -x "$CI_TOOLS/$archive/CMake.app/Contents/bin/cmake" ]; then
		curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${archive}.tar.gz" |
			tar -xz -C "$CI_TOOLS"
	fi
	ln -sf "$CI_TOOLS/$archive/CMake.app/Contents/bin/cmake" "$CI_TOOLS/bin/cmake"
fi
hash -r 2>/dev/null || true

CMAKE_BUILD_PARALLEL_LEVEL=$(sysctl -n hw.ncpu) cmake -P tools/ci/install-test.cmake
