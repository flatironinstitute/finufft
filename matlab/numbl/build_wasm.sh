#!/bin/bash
# Build FINUFFT as a WebAssembly module for use with numbl.
# Produces finufft.wasm in this directory.
#
# Prerequisites: emcc, emcmake, emmake on PATH (Emscripten SDK)
#
# Usage:
#   cd finufft/matlab/numbl && bash build_wasm.sh
#
# After building, run numbl with:
#   npx tsx src/cli.ts run \
#     --extra-path path/to/finufft/matlab/numbl \
#     --extra-path path/to/finufft/matlab \
#     your_script.m

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FINUFFT_SRC="${FINUFFT_SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="$SCRIPT_DIR/build"

if ! command -v emcc &> /dev/null; then
  echo "Error: emcc (Emscripten) not found on PATH." >&2
  echo "Install: https://emscripten.org/docs/getting_started/downloads.html" >&2
  exit 1
fi

echo "FINUFFT source: $FINUFFT_SRC"
echo "Build directory: $BUILD_DIR"

# Step 1: Build FINUFFT static library with Emscripten
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -f "Makefile" ]; then
  echo "=== Configuring FINUFFT with Emscripten CMake ==="
  emcmake cmake "$FINUFFT_SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFINUFFT_USE_OPENMP=OFF \
    -DFINUFFT_USE_DUCC0=ON \
    -DFINUFFT_STATIC_LINKING=ON \
    -DFINUFFT_BUILD_TESTS=OFF \
    -DFINUFFT_BUILD_EXAMPLES=OFF \
    -DFINUFFT_ARCH_FLAGS="" \
    -DFINUFFT_ENABLE_INSTALL=OFF \
    -DCMAKE_C_FLAGS="-msimd128" \
    -DCMAKE_CXX_FLAGS="-msimd128"
fi

echo "=== Building FINUFFT static library ==="
emmake make -j"$(nproc)" finufft

# Step 2: Find the built libraries
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

# Step 3: Link wrapper + FINUFFT into standalone WASM
echo "=== Linking finufft.wasm ==="
cd "$SCRIPT_DIR"

LINK_LIBS="$LIBFINUFFT $LIBCOMMON"
if [ -n "$LIBDUCC0" ]; then
  LINK_LIBS="$LINK_LIBS $LIBDUCC0"
fi

em++ finufft_wrapper.cpp \
  -I"$FINUFFT_SRC/include" \
  $LINK_LIBS \
  -O2 \
  -msimd128 \
  -s STANDALONE_WASM \
  --no-entry \
  -s TOTAL_MEMORY=67108864 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -o finufft.wasm

echo "=== Built finufft.wasm ($(wc -c < finufft.wasm) bytes) ==="
