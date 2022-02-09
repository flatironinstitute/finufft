#include "dispatch.h"

#include <cpuid.h>

namespace finufft {

Dispatch::Type get_current_capability() noexcept {
    // baseline dispatch type
    Dispatch::Type result = Dispatch::Scalar;

    unsigned eax, ebx, ecx, edx, flag = 0;

    // query basic cpuid info
    int cpuidret = __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    if (!cpuidret) {
        // failed to query cpuid
        return result;
    }

    if (ecx & bit_SSE4_1) {
        result = Dispatch::SSE4;
    }

    // query advanced cpuid info
    cpuidret = __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);

    if (!cpuidret) {
        // failed to query advanced cpuid
        return result;
    }

    if ((ebx & bit_AVX512F) && (ebx & bit_AVX512VL) && (ebx & bit_AVX512DQ)) {
        return Dispatch::AVX512;
    }

    if (ebx & bit_AVX2) {
        return Dispatch::AVX2;
    }

    return result;
}

} // namespace finufft
