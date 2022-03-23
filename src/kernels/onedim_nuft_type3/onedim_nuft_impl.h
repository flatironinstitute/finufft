#pragma once

#include "onedim_nuft.h"
#include "sctl/vec.hpp"

// Template implementation for the 1-d type 3 NUFT kernel.
// This file provides a generic implementation of the vectorized kernel backed by SCTL
// It is compiled to various target architectures in the onedim_nuft_*.cpp files
// Users should use the functions exposed in onedim_nuft.h

namespace finufft {

namespace detail {

// This is the implementation for the main loop of the 1-d type 3 NUFT kernel.
// It is vectorized using a given type T and to a given width N.
// It is crucial that this template be instantiated with a combination of T and N that is supported,
// in order to leverage the corresponding intrinsics.
template <typename T, int N>
void onedim_nuft_kernel_main(
    size_t nk, int q, T const *f, T const *z, T const *k, T *phihat) noexcept {
    typedef sctl::Vec<T, N> Vec;

    for (size_t j = 0; j < nk - N + 1; j += N) {
        auto x = Vec::Zero();
        auto kj = Vec::Load(k + j);

        for (int n = 0; n < q; ++n) {
            // broadcast z and f values
            auto zn = Vec::Load1(z + n);
            auto fn = Vec::Load1(f + n);
            fn *= Vec(static_cast<T>(2.0)); // TODO: investigate whether this can be pre-processed

            Vec znkj_sin, znkj_cos;
            sincos(znkj_sin, znkj_cos, kj * zn);

            x += fn * znkj_cos;
        }

        x.Store(phihat + j);
    }
}

template <typename T, int N>
void onedim_nuft_kernel_impl(
    size_t nk, int q, T const *f, T const *z, T const *k, T *phihat) noexcept {
    DisableDenormals disable_denormals;

    onedim_nuft_kernel_main<T, N>(nk, q, f, z, k, phihat);

    size_t remainder = nk % N;
    size_t offset = nk - remainder;

    onedim_nuft_kernel_scalar(remainder, q, f, z, k + offset, phihat + offset);
}

} // namespace detail

// Define a function calling the given implementation of the kernel
#define INSTANTIATE_NUFT_IMPLEMENTATION_FOR_TYPE(NAME, N, TYPE)                                    \
    void NAME(                                                                                     \
        size_t nk, int q, TYPE const *f, TYPE const *z, TYPE const *k, TYPE *phihat) noexcept {    \
        finufft::detail::onedim_nuft_kernel_impl<TYPE, N>(nk, q, f, z, k, phihat);                 \
    }

// Instantiate the template for the given width in single precision floating point,
// and half the given width in double precision floating point
#define INSTANTIATE_NUFT_IMPLEMENTATIONS_WITH_WIDTH(NAME, N)                                       \
    INSTANTIATE_NUFT_IMPLEMENTATION_FOR_TYPE(NAME, N, float)                                       \
    INSTANTIATE_NUFT_IMPLEMENTATION_FOR_TYPE(NAME, N / 2, double)

} // namespace finufft
