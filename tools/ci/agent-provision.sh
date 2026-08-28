#!/bin/bash

# Put cmake, ninja and MATLAB on a Jenkins host agent that ships with none of
# them. Probed on 2026-08-28, macpro and macm1 carry only Xcode's clang and
# make: no cmake, no ninja, no brew, no module system and no MATLAB.
#
# Everything lands in CI_TOOLS, outside the workspace, so the ~10 GB MATLAB
# install survives the next build. Each step is skipped when its target is
# already there, so only the first build on an agent pays for it.
#
# Nothing here needs admin rights: the cmake and ninja archives are relocatable,
# and mpm installs MATLAB wherever --destination points.
#
# Environment:
#   CI_TOOLS        install root, required
#   MATLAB_RELEASE  e.g. R2025b, required
set -euo pipefail

: "${CI_TOOLS:?}" "${MATLAB_RELEASE:?}"
CMAKE_VERSION=${CMAKE_VERSION:-3.31.6}
NINJA_VERSION=${NINJA_VERSION:-1.12.1}
MATLAB_PRODUCTS=${MATLAB_PRODUCTS:-"MATLAB Parallel_Computing_Toolbox"}

mkdir -p "$CI_TOOLS/bin" "$CI_TOOLS/tmp"

case "$(uname -s)" in
Darwin)
	cmake_archive=cmake-${CMAKE_VERSION}-macos-universal
	cmake_bin=$CI_TOOLS/$cmake_archive/CMake.app/Contents/bin
	ninja_zip=ninja-mac.zip
	# maci64 and maca64 are separate mpm builds, and an Intel mpm cannot install
	# an Apple silicon MATLAB.
	[[ "$(uname -m)" == arm64 ]] && mpm_arch=maca64 || mpm_arch=maci64
	# matlabroot on macOS is the .app bundle itself, and mpm appends .app to a
	# destination that lacks it, so build 9 installed one level over from where
	# --destination pointed and the rename found nothing.
	matlab_suffix=.app
	;;
Linux)
	cmake_archive=cmake-${CMAKE_VERSION}-linux-x86_64
	cmake_bin=$CI_TOOLS/$cmake_archive/bin
	ninja_zip=ninja-linux.zip
	mpm_arch=glnxa64
	matlab_suffix=
	;;
*)
	echo "agent-provision.sh does not cover $(uname -s)" >&2
	exit 1
	;;
esac

if [[ ! -x "$cmake_bin/cmake" ]]; then
	curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${cmake_archive}.tar.gz" |
		tar -xz -C "$CI_TOOLS"
fi
ln -sf "$cmake_bin/cmake" "$cmake_bin/ctest" "$cmake_bin/cpack" "$CI_TOOLS/bin/"

if [[ ! -x "$CI_TOOLS/bin/ninja" ]]; then
	curl -fsSL -o "$CI_TOOLS/tmp/$ninja_zip" \
		"https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/${ninja_zip}"
	unzip -oq "$CI_TOOLS/tmp/$ninja_zip" -d "$CI_TOOLS/bin"
	chmod +x "$CI_TOOLS/bin/ninja"
fi

# mpm resumes nothing: a half-finished install would be taken for a good one, so
# it builds under .partial and is renamed only once mpm has returned 0.
matlab_root=$CI_TOOLS/matlab/$MATLAB_RELEASE$matlab_suffix
partial=$CI_TOOLS/matlab/$MATLAB_RELEASE.partial$matlab_suffix
if [[ ! -x "$matlab_root/bin/matlab" ]]; then
	rm -rf "$partial" "$partial.app"
	curl -fsSL -o "$CI_TOOLS/tmp/mpm" "https://www.mathworks.com/mpm/$mpm_arch/mpm"
	chmod +x "$CI_TOOLS/tmp/mpm"
	"$CI_TOOLS/tmp/mpm" install --release="$MATLAB_RELEASE" \
		--destination="$partial" --products $MATLAB_PRODUCTS
	# Build 9: both macs reported "Products will be installed to:
	# <destination>.app" whatever the destination was. Take whichever exists, so
	# a destination that already ends in .app cannot grow a second one.
	[[ -d "$partial" ]] || partial=$partial.app
	mv "$partial" "$matlab_root"
fi

# Assert rather than trust: a truncated install has to fail here, not later in a
# stage where it reads as a code failure. No `matlab -batch`, which would need a
# license the image and this agent are not given.
for f in "$matlab_root/bin/matlab" "$matlab_root/bin/mex" \
	"$matlab_root/extern/include/mex.h" \
	"$matlab_root/toolbox/parallel/gpu/extern/include/gpu/mxGPUArray.h"; do
	[[ -e "$f" ]] || {
		echo "ERROR: $f missing after provisioning"
		exit 1
	}
done

"$CI_TOOLS/bin/cmake" --version | head -1
"$CI_TOOLS/bin/ninja" --version
echo "MATLAB $MATLAB_RELEASE at $matlab_root"
