#!/bin/bash
# Build a numbl-compatible native shared library that exposes the finufft
# mexFunction (from upstream matlab/finufft.cpp) via a small mex shim.
# Produces finufft.so (Linux) or finufft.dylib (macOS) in this directory.
# All FINUFFT and ducc0 dependencies are statically linked.
#
# Usage:
#   cd finufft/matlab/numbl && bash build_native.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FINUFFT_SRC="${FINUFFT_SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="$SCRIPT_DIR/build_native"

# Detect platform
case "$(uname -s)" in
  Linux*)  EXT="so";    SHARED_FLAGS="-shared -fPIC" ;;
  Darwin*) EXT="dylib"; SHARED_FLAGS="-dynamiclib" ;;
  *)       echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

echo "FINUFFT source: $FINUFFT_SRC"
echo "Build directory: $BUILD_DIR"
echo "Output: finufft.$EXT"

# Step 1: Build FINUFFT static libraries
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -f "Makefile" ]; then
  echo "=== Configuring FINUFFT with CMake ==="
  cmake "$FINUFFT_SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFINUFFT_USE_OPENMP=OFF \
    -DFINUFFT_USE_DUCC0=ON \
    -DFINUFFT_STATIC_LINKING=ON \
    -DFINUFFT_BUILD_TESTS=OFF \
    -DFINUFFT_BUILD_EXAMPLES=OFF \
    -DFINUFFT_ENABLE_INSTALL=OFF \
    -DCMAKE_C_FLAGS="-fPIC" \
    -DCMAKE_CXX_FLAGS="-fPIC"
fi

echo "=== Building FINUFFT static library ==="
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j"$NPROC" finufft

# Step 2: Find built libraries
LIBFINUFFT="$BUILD_DIR/src/libfinufft.a"
LIBCOMMON="$BUILD_DIR/src/common/libfinufft_common.a"

LIBDUCC0=$(find "$BUILD_DIR" -name "libducc0.a" -print -quit 2>/dev/null || true)
if [ -z "$LIBDUCC0" ]; then
  echo "Warning: libducc0.a not found, trying without it"
  LIBDUCC0=""
fi

echo "Libraries:"
echo "  finufft: $LIBFINUFFT"
echo "  common:  $LIBCOMMON"
echo "  ducc0:   ${LIBDUCC0:-not found}"

# Step 3: Compile finufft.cpp (the upstream mwrap-generated MEX source) and
# our mex shim, then link them with the static finufft libs.
echo "=== Linking finufft.$EXT ==="
cd "$SCRIPT_DIR"

LINK_LIBS="$LIBFINUFFT $LIBCOMMON"
if [ -n "$LIBDUCC0" ]; then
  LINK_LIBS="$LINK_LIBS $LIBDUCC0"
fi

# Place our shim's mex.h ahead of any system mex.h.
SHIM_INC="-I$SCRIPT_DIR/mex_shim"

# Only export the mex_* and my_* symbols that JS calls into.
EXTRA_LINK=""
VERSION_SCRIPT=""
if [ "$EXT" = "so" ]; then
  VERSION_SCRIPT='{ global: mex_*; my_*; local: *; };'
fi

COMPILE_FLAGS=(
  $SHIM_INC
  -I"$FINUFFT_SRC/include"
  -DMX_HAS_INTERLEAVED_COMPLEX=1
  -O2
  $SHARED_FLAGS
  -fvisibility=hidden
)

SOURCES=(
  "$FINUFFT_SRC/matlab/finufft.cpp"
  "$SCRIPT_DIR/mex_shim.cpp"
)

if [ "$EXT" = "so" ]; then
  echo "$VERSION_SCRIPT" | g++ "${SOURCES[@]}" \
    "${COMPILE_FLAGS[@]}" \
    $LINK_LIBS \
    -Wl,--version-script=/dev/stdin \
    -lstdc++ -lm -lpthread \
    -o "finufft.$EXT"
else
  g++ "${SOURCES[@]}" \
    "${COMPILE_FLAGS[@]}" \
    $LINK_LIBS \
    -lstdc++ -lm -lpthread \
    -o "finufft.$EXT"
fi

echo "=== Built finufft.$EXT ($(wc -c < "finufft.$EXT") bytes) ==="
