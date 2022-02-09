#pragma once

// Utilities for benchmarking and testing 1-dimensional type-3 NUFT kernels.

#include <cmath>
#include <cstddef>
#include <vector>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>


extern "C" {
#include "../contrib/legendre_rule_fast.h"
}


namespace finufft {

namespace testing {

template <typename FT> FT eval_kernel(FT x, int width) {
    FT ES_c = 4.0 / (width * width);
    FT ES_beta = 2.3 * width;

    if (std::abs(x) > width / 2) {
        return 0.0;
    }

    return std::exp(ES_beta * std::sqrt(1.0 - ES_c * x * x));
}

template <typename T> std::vector<T> generate_random_data(int n, int seed) {
    typedef r123::Philox2x32 RNG;
    RNG rng;

    RNG::ctr_type ctr = {{}};
    RNG::ukey_type key = {{}};
    key[0] = seed;

    std::vector<T> result(n);

    for (int i = 0; i < n; i++) {
        ctr[0] = i;
        auto r = rng(ctr, key);
        result[i] = r123::u01<T>(r[0]);
    }

    return result;
}

/** Invoke one-dimensional nuft with given kernel.
 *
 * This function sets-up a problem of the given width and invokes it on the given data.
 *
 * k is the input array, of length nk
 * phihat is an output array, of length nk
 *
 */
template <int nspread, typename FT, typename Fn>
void onedim_nuft_with_method(size_t nk, FT const *k, FT *phihat, Fn &&fn) {
    // Setup constants

    FT J2 = nspread / 2.0; // J/2, half-width of ker z-support
    // # quadr nodes in z (from 0 to J/2; reflections will be added)...
    const int q = 2 + nspread;

    FT f[q];
    double z[2 * q], w[2 * q];         // glr needs double
    legendre_compute_glr(2 * q, z, w); // only half the nodes used, eg on (0,1)

    FT zf[q];

    for (int n = 0; n < q; ++n) {
        zf[n] = z[n] * J2; // quadr nodes for [0,J/2]
        f[n] = J2 * w[n] * eval_kernel(zf[n], nspread);
    }

    // Invoke evaluation loop.
    fn(nk, q, f, zf, k, phihat);
}


} // namespace testing

} // namespace finufft