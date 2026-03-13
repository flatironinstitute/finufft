#!/bin/bash
# Build FINUFFT as a standalone native shared library for numbl.
# Produces finufft.so (Linux) or finufft.dylib (macOS) in this directory.
# All dependencies (FINUFFT, ducc0) are statically linked.
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

# Step 3: Link wrapper + static libs into shared library
echo "=== Linking finufft.$EXT ==="
cd "$SCRIPT_DIR"

LINK_LIBS="$LIBFINUFFT $LIBCOMMON"
if [ -n "$LIBDUCC0" ]; then
  LINK_LIBS="$LINK_LIBS $LIBDUCC0"
fi

# Platform-specific linker flags
EXTRA_LINK=""
if [ "$EXT" = "so" ]; then
  EXTRA_LINK="-Wl,--version-script=/dev/stdin"
  # Only export guru_* symbols
  VERSION_SCRIPT=$(cat <<'VEOF'
{ global: guru_*; local: *; };
VEOF
)
fi

if [ "$EXT" = "so" ]; then
  echo "$VERSION_SCRIPT" | g++ finufft_native_wrapper.cpp \
    -I"$FINUFFT_SRC/include" \
    $LINK_LIBS \
    -O2 $SHARED_FLAGS -fvisibility=hidden \
    -Wl,--version-script=/dev/stdin \
    -lstdc++ -lm -lpthread \
    -o "finufft.$EXT"
else
  g++ finufft_native_wrapper.cpp \
    -I"$FINUFFT_SRC/include" \
    $LINK_LIBS \
    -O2 $SHARED_FLAGS -fvisibility=hidden \
    -lstdc++ -lm -lpthread \
    -o "finufft.$EXT"
fi

echo "=== Built finufft.$EXT ($(wc -c < "finufft.$EXT") bytes) ==="
