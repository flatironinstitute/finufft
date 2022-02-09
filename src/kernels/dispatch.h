#pragma once

// Utilities for cpuid-based dispatching of functions.

namespace finufft {

// Summary of the capability of the current cpu.
// Note that we use a simplified approach categorizing the capabilities as:
// - Scalar: no SIMD instruction capability
// - SSE4: SSE4.1 instructions
// - AVX2: AVX2 + FMA instructions
// - AVX512: AVX512F + AVX512VL + AVX512DQ
// Note that it is assumed that these capabilities are inclusive, that is,
// AVX512 implies AVX2 and SSE4 etc.
struct Dispatch {
    enum Type : int {
        Scalar = 0,
        SSE4 = 1,
        AVX2 = 2,
        AVX512 = 3
    };
};

// Query the current cpu for its capability.
Dispatch::Type get_current_capability() noexcept;

}
