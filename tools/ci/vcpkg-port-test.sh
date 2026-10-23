#!/bin/bash
# Build the overlay port tools/vcpkg/finufft with a pinned vcpkg and consume it
# the two ways docs/install.rst gives: classic mode (vcpkg install, then the
# vcpkg toolchain) and manifest mode (the consumer's vcpkg.json).
#
# Env: VCPKG_TAG (vcpkg tool tag), VCPKG_WORK (clone and build root)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
work=${VCPKG_WORK:-$HOME/_vcpkg}
vcpkg_root=$work/vcpkg
vcpkg=$vcpkg_root/vcpkg
toolchain=$vcpkg_root/scripts/buildsystems/vcpkg.cmake
overlay=$repo/tools/vcpkg
consumer=$repo/examples/quick-start/find_package
# The port builds the checkout, which the ABI hash does not see: a cached binary
# would test an older checkout.
export VCPKG_BINARY_SOURCES=clear VCPKG_DISABLE_METRICS=1

port_version=$(sed -n 's/^  "version": "\(.*\)",$/\1/p' "$overlay/finufft/vcpkg.json")
cmake_version=$(sed -n 's/^project(FINUFFT VERSION \([0-9.]*\) .*/\1/p' "$repo/CMakeLists.txt")
if [ -z "$cmake_version" ] || [ "$port_version" != "$cmake_version" ]; then
	echo "FAIL: vcpkg.json version '$port_version' != CMakeLists.txt version '$cmake_version'"
	exit 1
fi
echo "PASS: vcpkg.json version $port_version matches CMakeLists.txt"

mkdir -p "$work"
if [ ! -x "$vcpkg" ]; then
	git clone --depth 1 --branch "${VCPKG_TAG:-2026.07.29}" https://github.com/microsoft/vcpkg "$vcpkg_root"
	"$vcpkg_root/bootstrap-vcpkg.sh" -disableMetrics
fi
# An install left by an earlier run would satisfy every install below.
rm -rf "$vcpkg_root/installed" "$vcpkg_root/packages" "$vcpkg_root/buildtrees"

# Control: with no FFT feature the port must refuse to build.
if "$vcpkg" install 'finufft[core]' --overlay-ports="$overlay" >"$work/core.log" 2>&1; then
	echo "FAIL: finufft[core] installed with no FFT"
	exit 1
fi
grep -F "finufft needs an FFT" "$work/core.log" || {
	cat "$work/core.log"
	echo "FAIL: finufft[core] failed without the no-FFT message"
	exit 1
}
echo "PASS: finufft[core] fails with the no-FFT message"

# consume <build dir> <source dir> <build type> <lib dir the link line must name>
consume() {
	rm -rf "$1"
	cmake -S "$2" -B "$1" -DCMAKE_BUILD_TYPE="$3" -DCMAKE_TOOLCHAIN_FILE="$toolchain"
	cmake --build "$1" --verbose >"$1.log" 2>&1 || {
		cat "$1.log"
		exit 1
	}
	grep -F "$4/libfinufft" "$1.log" >/dev/null || {
		echo "FAIL: $3 consumer does not link $4/libfinufft"
		exit 1
	}
	"$1/app"
	echo "PASS: $3 consumer links $4/libfinufft and runs"
}

# Classic mode, default features (fftw, openmp).
"$vcpkg" install finufft --overlay-ports="$overlay"
installed=$vcpkg_root/installed/x64-linux
consume "$work/classic-release" "$consumer" Release "$installed/lib"
consume "$work/classic-debug" "$consumer" Debug "$installed/debug/lib"

# Manifest mode, DUCC0 instead of FFTW: vcpkg installs the manifest at configure.
app=$work/manifest-app
rm -rf "$app"
mkdir -p "$app"
cp "$consumer/CMakeLists.txt" "$consumer/main.cpp" "$app/"
cat >"$app/vcpkg.json" <<'EOF'
{
  "dependencies": [
    { "name": "finufft", "default-features": false, "features": ["ducc0", "openmp"] }
  ]
}
EOF
printf '{ "overlay-ports": ["%s"] }\n' "$overlay" >"$app/vcpkg-configuration.json"
# CMake names the manifest install relative to the build dir.
consume "$work/manifest-release" "$app" Release "vcpkg_installed/x64-linux/lib"
if [ -e "$work/manifest-release/vcpkg_installed/x64-linux/lib/libfftw3.a" ]; then
	echo "FAIL: the ducc0 feature installed fftw3"
	exit 1
fi
echo "PASS: the ducc0 feature installs no fftw3"
