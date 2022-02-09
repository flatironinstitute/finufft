#pragma once

/** Inner loops for type 3 1-d kernel functionality.
 *
 */

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace finufft {
// Plain C++ scalar version of the inner loop.
template <typename FT>
void onedim_nuft_kernel_scalar(
    size_t nk, int q, FT const *f, FT const *z, FT const *k, FT *phihat) {
    for (size_t j = 0; j < nk; ++j) { // loop along output array
        FT x = 0.0;                   // register
        for (int n = 0; n < q; ++n)
            x += f[n] * 2 * std::cos(k[j] * z[n]); // pos & neg freq pair.
        phihat[j] = x;
    }
}

// Define instantiations for different instruction sets
#define NUFT_DECLARE_KERNEL_INSTRUCTION(EXT_NAME)                                                  \
    void onedim_nuft_kernel_ ##EXT_NAME(                                                            \
        size_t nk, int q, float const *f, float const *z, float const *k, float *phihat);          \
    void onedim_nuft_kernel_ ##EXT_NAME(                                                            \
        size_t nk, int q, double const *f, double const *z, double const *k, double *phihat);

NUFT_DECLARE_KERNEL_INSTRUCTION(sse4)
NUFT_DECLARE_KERNEL_INSTRUCTION(avx2)
NUFT_DECLARE_KERNEL_INSTRUCTION(avx512)

#undef NUFT_DECLARE_KERNEL_INSTRUCTION

// RAII class to disable handling of denormals within the scope.
// Constructing this class will cache the current value of the FTZ and DAZ flags,
// and will restore them when the object is destroyed.
class DisableDenormals {
  private:
    std::uint32_t ftz_mode;
    std::uint32_t daz_mode;

  public:
    DisableDenormals();
    ~DisableDenormals();
    DisableDenormals(DisableDenormals const &) = delete;
};

} // namespace finufft