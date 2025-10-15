#ifndef FINUFFT_XSIMD_HPP
#define FINUFFT_XSIMD_HPP

#include <xsimd/config/xsimd_config.hpp>

#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
#undef XSIMD_NO_SUPPORTED_ARCHITECTURE
#define XSIMD_WITH_EMULATED 1
#define XSIMD_DEFAULT_ARCH  emulated<128>
#endif

#include <xsimd/xsimd.hpp>

#endif // FINUFFT_XSIMD_HPP
