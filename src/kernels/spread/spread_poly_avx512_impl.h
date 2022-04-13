#pragma once

#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include "spread_kernel_weights.h"

namespace finufft {
namespace detail {


template<typename T>
void copy_from_source_7_to_8(T&& source, float* target) {
    target[0] = source[0];
    target[1] = source[1];
    target[2] = source[4];
    target[3] = source[5];
    target[4] = source[2];
    target[5] = source[3];
    target[6] = source[6];
    target[7] = 0;
}


template<typename T>
void copy_from_source_7_to_16x2(T&& source, float* target) {
    copy_from_source_7_to_8(source, target);
    copy_from_source_7_to_8(source, target + 8);
}


__m512 inline __attribute__((always_inline)) evaluate_polynomial_horner_avx512(__m512 z, __m512 c0) { return c0; }

template <typename... Ts> __m512 inline __attribute__((always_inline)) evaluate_polynomial_horner_avx512(__m512 z, __m512 c0, Ts... c) {
    return _mm512_fmadd_ps(z, evaluate_polynomial_horner_avx512(z, c...), c0);
}

template <typename... Ts> __m512 evaluate_polynomial_horner_avx512_memory(__m512 z, Ts... mem_c) {
    return evaluate_polynomial_horner_avx512(z, _mm512_load_ps(mem_c)...);
}

struct ker_horner_avx512_w7_op {
    constexpr static const int width = 7;
    constexpr static const int degree = 10;
    constexpr static const double beta = 16.099999999999998;
    constexpr static const int stride = 2;
    constexpr static const int required_elements = 2;

    alignas(64) float c0d_[16];
    alignas(64) float c1d_[16];
    alignas(64) float c2d_[16];
    alignas(64) float c3d_[16];
    alignas(64) float c4d_[16];
    alignas(64) float c5d_[16];
    alignas(64) float c6d_[16];
    alignas(64) float c7d_[16];
    alignas(64) float c8d_[16];
    alignas(64) float c9d_[16];

    template<typename T>
    ker_horner_avx512_w7_op(T const& data) {
        // During construction, we partially swizzle the polynomial
        // coefficients to save on 
        copy_from_source_7_to_16x2(std::get<0>(data), c0d_);
        copy_from_source_7_to_16x2(std::get<1>(data), c1d_);
        copy_from_source_7_to_16x2(std::get<2>(data), c2d_);
        copy_from_source_7_to_16x2(std::get<3>(data), c3d_);
        copy_from_source_7_to_16x2(std::get<4>(data), c4d_);
        copy_from_source_7_to_16x2(std::get<5>(data), c5d_);
        copy_from_source_7_to_16x2(std::get<6>(data), c6d_);
        copy_from_source_7_to_16x2(std::get<7>(data), c7d_);
        copy_from_source_7_to_16x2(std::get<8>(data), c8d_);
        copy_from_source_7_to_16x2(std::get<9>(data), c9d_);
    }

    void compute(std::array<float, 2> xi, std::array<float, 4> dd, __m512& v1, __m512& v2) const {
        __m512 x = _mm512_insertf32x8(_mm512_set1_ps(xi[0]), _mm256_set1_ps(xi[1]), 1);
        __m512 z = _mm512_add_ps(_mm512_add_ps(x, x), _mm512_set1_ps(width - 1.0f));

        __m512 k = evaluate_polynomial_horner_avx512_memory(
            z, c0d_, c1d_, c2d_, c3d_, c4d_, c5d_, c6d_, c7d_, c8d_, c9d_);

        __m512 w_re = _mm512_insertf32x8(_mm512_set1_ps(dd[0]), _mm256_set1_ps(dd[2]), 1);
        __m512 w_im = _mm512_insertf32x8(_mm512_set1_ps(dd[1]), _mm256_set1_ps(dd[3]), 1);

        __m512 k_re = _mm512_mul_ps(k, w_re);
        __m512 k_im = _mm512_mul_ps(k, w_im);

        __m512 hi = _mm512_unpackhi_ps(k_re, k_im);
        __m512 lo = _mm512_unpacklo_ps(k_re, k_im);

        v1 = _mm512_insertf32x8(lo, _mm512_extractf32x8_ps(hi, 0), 1);
        v2 = _mm512_insertf32x8(hi, _mm512_extractf32x8_ps(lo, 1), 0);
    }

    void operator()(float* __restrict du, float const* __restrict kx, float const* __restrict dd, std::size_t i) const {
        float x1 = kx[i];
        float x2 = kx[i + 1];

        float i1f = std::ceil(x1 - 0.5f * width);
        float i2f = std::ceil(x2 - 0.5f * width);

        std::size_t i1 = static_cast<std::size_t>(i1f);
        std::size_t i2 = static_cast<std::size_t>(i2f);

        float xi1 = i1f - x1;
        float xi2 = i2f - x2;

        std::array<float, 2> xi = {xi1, xi2};
        std::array<float, 4> dd_ = {dd[2 * i], dd[2 * i + 1], dd[2 * i + 2], dd[2 * i + 3]};

        __m512 v1;
        __m512 v2;

        compute(xi, dd_, v1, v2);

        float* out_1 = du + 2 * i1;
        float* out_2 = du + 2 * i2;

        __m512 out_v1 = _mm512_loadu_ps(out_1);
        out_v1 = _mm512_add_ps(out_v1, v1);
        _mm512_storeu_ps(out_1, out_v1);

        __m512 out_v2 = _mm512_loadu_ps(out_2);
        out_v2 = _mm512_add_ps(out_v2, v2);
        _mm512_storeu_ps(out_2, out_v2);
    }
};

static const ker_horner_avx512_w7_op ker_horner_avx512_w7(weights_w7);

} // namespace detail
} // namespace finufft
