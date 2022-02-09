#include "onedim_nuft.h"
#include <sctl/vec.hpp>

namespace {

void onedim_nuft_avx2_main(
    size_t nk, int q, float const *f, float const *z, float const *k, float *phihat) {
    typedef sctl::Vec<float, 8> Vec8f;

    for (size_t j = 0; j < nk - 7; j += 8) {
        auto x = Vec8f::Zero();
        auto kj = Vec8f::Load(k + j);

        for (int n = 0; n < q; ++n) {
            // broadcast z and f values
            auto zn = Vec8f::Load1(z + n);
            auto fn = Vec8f::Load1(f + n);
            fn *= 2.0f; // TODO: investigate whether this can be pre-processed

            Vec8f znkj_cos, znkj_sin;
            sincos(znkj_sin, znkj_cos, kj * zn);

            x += fn * znkj_cos;
        }

        x.Store(phihat + j);
    }
}

} // namespace

namespace finufft {

void onedim_nuft_kernel_avx2(
    size_t nk, int q, float const *f, float const *z, float const *k, float *phihat) {

    // Disable proper denormal handling
    // Very important for performance (up to 3x difference) as they can be produced in the polynomial approximation.
    DisableDenormals disable_denormals;

    onedim_nuft_avx2_main(nk, q, f, z, k, phihat);

    // tail loop
    size_t remainder = nk % 8;
    size_t offset = nk - remainder;

    onedim_nuft_kernel_scalar(remainder, q, f, z, k + offset, phihat + offset);
}

} // namespace finufft