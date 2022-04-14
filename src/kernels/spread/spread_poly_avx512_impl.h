#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <immintrin.h>

#include "spread_kernel_weights.h"

namespace finufft {
namespace detail {

template <typename T> void copy_from_source_7_to_8(T &&source, float *target) {
    // target[0] = source[0];
    // target[1] = source[1];
    // target[2] = source[4];
    // target[3] = source[5];
    // target[4] = source[2];
    // target[5] = source[3];
    // target[6] = source[6];
    std::copy_n(source.begin(), 7, target);
    target[7] = 0;
}

template <typename T> void copy_from_source_7_to_16x2(T &&source, float *target) {
    copy_from_source_7_to_8(source, target);
    copy_from_source_7_to_8(source, target + 8);
}

__m512 inline __attribute__((always_inline))
evaluate_polynomial_horner_avx512(__m512 z, __m512 c0) {
    return c0;
}

template <typename... Ts>
__m512 inline __attribute__((always_inline))
evaluate_polynomial_horner_avx512(__m512 z, __m512 c0, Ts... c) {
    return _mm512_fmadd_ps(z, evaluate_polynomial_horner_avx512(z, c...), c0);
}

template <typename... Ts> __m512 evaluate_polynomial_horner_avx512_memory(__m512 z, Ts... mem_c) {
    return evaluate_polynomial_horner_avx512(z, _mm512_load_ps(mem_c)...);
}

inline void accumulate_add_complex_interleaved(float *du, int i, __m512 v) {
    float *out = du + 2 * i;
    __m512 out_v = _mm512_loadu_ps(out);
    out_v = _mm512_add_ps(out_v, v);
    _mm512_storeu_ps(out, out_v);
}

// The next two functions compute the lookup table for shuffles
// corresponding to element shifts of the given number of elements.
// Note that this is similar to valignd, except that we are able to
// specify the shift amount at runtime, whereas valignd uses an immediate value.
//
// The constructed table conceptually shifts quadword elements (i.e. two floats at a time)
// as the values held in the registers conceptually represent interleaved complex numbers.
//
// I have reproduced a 4-wide version of the tables below, with a single offset between lines.
// The true tables are 16-wide, and offset by 2 each line. Note that here, s denotes an index
// which fetches from the second vector of the shuffle, whereas all other indices fetch from
// the first vector of the shuffle. We use a zero vector as the second vector, so s denotes zero
// values.
//
// shuffle_low   shuffle_high
//  0 1 2 3       s s s s
//  s 0 1 2       3 s s s
//  s s 0 1       2 3 s s
//  s s s 0       1 2 3 s

inline constexpr std::array<int, 16 * 8> make_shuffle_low() {
    std::array<int, 16 * 8> result = {};

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < i; ++j) {
            result[i * 16 + 2 * j] = 0b10000;
            result[i * 16 + 2 * j + 1] = 0b10000;
        }

        for (int j = i; j < 8; ++j) {
            result[i * 16 + 2 * j] = 2 * (j - i);
            result[i * 16 + 2 * j + 1] = 2 * (j - i) + 1;
        }
    }

    return result;
}

inline constexpr std::array<int, 16 * 8> make_shuffle_high() {
    std::array<int, 16 * 8> result = {};

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < i; ++j) {
            result[i * 16 + 2 * j] = 2 * (8 - i) + 2 * j;
            result[i * 16 + 2 * j + 1] = 2 * (8 - i) + 2 * j + 1;
        }

        for (int j = i; j < 8; ++j) {
            result[i * 16 + 2 * j] = 0b10000;
            result[i * 16 + 2 * j + 1] = 0b10000;
        }
    }

    return result;
}

alignas(64) static constexpr std::array<int, 16 * 8> align_shuffles_low = make_shuffle_low();
alignas(64) static constexpr std::array<int, 16 * 8> align_shuffles_high = make_shuffle_high();

// Accumulates and stores the given vector into du at the given index.
// This function splits the stores to ensure that they are aligned,
// and ensuring that store-to-load forwarding may be successful (as all stores
// operate on aligned addresses, they either coincide exactly or do not alias).
//
// Operationally, given the vector of elements v, we conceptually consider its
// representation offset by 2 * i elements. This representation is then split
// into the lower 16 elements and the upper 16 elements using a shuffle.
// To avoid branches, the shuffle is produced by a lookup table.
//
inline void accumulate_add_complex_interleaved_aligned(float *du, int i, __m512 v) {
    int i_aligned = i & ~7;
    int i_remainder = i - i_aligned;

    float *out = du + 2 * i_aligned;

    __m512 v_lo = _mm512_permutex2var_ps(
        v, _mm512_load_epi32(align_shuffles_low.data() + i_remainder * 16), _mm512_setzero_ps());

    __m512 v_hi = _mm512_permutex2var_ps(
        v, _mm512_load_epi32(align_shuffles_high.data() + i_remainder * 16), _mm512_setzero_ps());

    __m512 out_lo = _mm512_load_ps(out);
    __m512 out_hi = _mm512_load_ps(out + 16);

    out_lo = _mm512_add_ps(out_lo, v_lo);
    out_hi = _mm512_add_ps(out_hi, v_hi);

    _mm512_store_ps(out, out_lo);
    _mm512_store_ps(out + 16, out_hi);
}

// Standard implmentation of kernel computation
// This is slower than the unrolled implementation below
// with vectorized index computation.
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

    template <typename T> ker_horner_avx512_w7_op(T const &data) {
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

    void compute(std::array<float, 2> xi, std::array<float, 4> dd, __m512 &v1, __m512 &v2) const {
        // This contains the main computation of the kernel and its interleaved outputs
        // It is separated for ease of testing.

        // This kernel computes two values at once, each returned in interleaved complex format
        // in v1 and v2.

        // Load the input position and scale it to the range [-1, 1]
        // The two inputs are split into 256-bit lanes.
        __m512 x = _mm512_insertf32x8(_mm512_set1_ps(xi[0]), _mm256_set1_ps(xi[1]), 1);
        __m512 z = _mm512_add_ps(_mm512_add_ps(x, x), _mm512_set1_ps(width - 1.0f));

        // Evaluate polynomial for all values.
        __m512 k = evaluate_polynomial_horner_avx512_memory(
            z, c0d_, c1d_, c2d_, c3d_, c4d_, c5d_, c6d_, c7d_, c8d_, c9d_);

        // Load real and imaginary coefficients, split by 256-bit lane.
        __m512 w_re = _mm512_insertf32x8(_mm512_set1_ps(dd[0]), _mm256_set1_ps(dd[2]), 1);
        __m512 w_im = _mm512_insertf32x8(_mm512_set1_ps(dd[1]), _mm256_set1_ps(dd[3]), 1);

        __m512 k_re = _mm512_mul_ps(k, w_re);
        __m512 k_im = _mm512_mul_ps(k, w_im);

        // Write-out the results in interleaved format.
        const int from_b = (1 << 4);
        const int from_a = (0 << 4);

        // clang-format off
        v1 = _mm512_permutex2var_ps(
            k_re,
            _mm512_setr_epi32(
                from_a | 0, from_b | 0, from_a | 1, from_b | 1, from_a | 2, from_b | 2, from_a | 3, from_b | 3,
                from_a | 4, from_b | 4, from_a | 5, from_b | 5, from_a | 6, from_b | 6, from_a | 7, from_b | 7),
            k_im);
        v2 = _mm512_permutex2var_ps(
            k_re,
            _mm512_setr_epi32(
                from_a | 8, from_b | 8, from_a | 9, from_b | 9, from_a | 10, from_b | 10, from_a | 11, from_b | 11,
                from_a | 12, from_b | 12, from_a | 13, from_b | 13, from_a | 14, from_b | 14, from_a | 15, from_b | 15),
            k_im);
        // clang-format on
    }

    void operator()(
        float *__restrict du, float const *__restrict kx, float const *__restrict dd,
        std::size_t i) const {
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

        // accumulate_add_complex_interleaved(du, i1, v1);
        // accumulate_add_complex_interleaved(du, i2, v2);
        accumulate_add_complex_interleaved_aligned(du, i1, v1);
        accumulate_add_complex_interleaved_aligned(du, i2, v2);
    }
};

static const ker_horner_avx512_w7_op ker_horner_avx512_w7(weights_w7);

// 4-wise unrolled width 7 kernel
// The unrolling allows us to vectorize the computation of the index,
// which leads to a big speedup compared to the non-unrolled version (2x).
// Note that it is essential to use `accumulate_add_complex_interleaved_aligned`
// instead of the naive implementation, or the latency introduced by failed store-to-load
// forwarding due to loading at aliased addresses will overwhelm any benefit.
struct ker_horner_avx512_w7_r4_op {
    constexpr static const int width = 7;
    constexpr static const int degree = 10;
    constexpr static const double beta = 16.099999999999998;
    constexpr static const int stride = 8;
    constexpr static const int required_elements = 8;

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

    template <typename T> ker_horner_avx512_w7_r4_op(T const &data) {
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

    void compute(__m512 x, float const *dd, __m512 &v1, __m512 &v2) const {
        // This contains the main computation of the kernel and its interleaved outputs
        // It is separated for ease of testing.

        // This kernel computes two values at once, each returned in interleaved complex format
        // in v1 and v2.

        // Load the input position and scale it to the range [-1, 1]
        // The two inputs are split into 256-bit lanes.
        __m512 z = _mm512_add_ps(_mm512_add_ps(x, x), _mm512_set1_ps(width - 1.0f));

        // Evaluate polynomial for all values.
        __m512 k = evaluate_polynomial_horner_avx512_memory(
            z, c0d_, c1d_, c2d_, c3d_, c4d_, c5d_, c6d_, c7d_, c8d_, c9d_);

        // Load real and imaginary coefficients, split by 256-bit lane.
        __m512 w_re = _mm512_insertf32x8(_mm512_set1_ps(dd[0]), _mm256_set1_ps(dd[2]), 1);
        __m512 w_im = _mm512_insertf32x8(_mm512_set1_ps(dd[1]), _mm256_set1_ps(dd[3]), 1);

        __m512 k_re = _mm512_mul_ps(k, w_re);
        __m512 k_im = _mm512_mul_ps(k, w_im);

        // Write-out the results in interleaved format.
        const int from_b = (1 << 4);
        const int from_a = (0 << 4);

        // clang-format off
        v1 = _mm512_permutex2var_ps(
            k_re,
            _mm512_setr_epi32(
                from_a | 0, from_b | 0, from_a | 1, from_b | 1, from_a | 2, from_b | 2, from_a | 3, from_b | 3,
                from_a | 4, from_b | 4, from_a | 5, from_b | 5, from_a | 6, from_b | 6, from_a | 7, from_b | 7),
            k_im);
        v2 = _mm512_permutex2var_ps(
            k_re,
            _mm512_setr_epi32(
                from_a | 8, from_b | 8, from_a | 9, from_b | 9, from_a | 10, from_b | 10, from_a | 11, from_b | 11,
                from_a | 12, from_b | 12, from_a | 13, from_b | 13, from_a | 14, from_b | 14, from_a | 15, from_b | 15),
            k_im);
        // clang-format on
    }

    void operator()(
        float *__restrict du, float const *__restrict kx, float const *__restrict dd,
        std::size_t i) const {
        __m256 x = _mm256_loadu_ps(kx + i);
        __m256 x_ceil = _mm256_ceil_ps(_mm256_sub_ps(x, _mm256_set1_ps(0.5f * width)));
        __m256i x_ceili = _mm256_cvtps_epi32(x_ceil);
        __m256 xi = _mm256_sub_ps(x_ceil, x);

        // Prepare x register so that we can obtain pairs using vpermilps
        // Q: Is there a two-register instruction to do this? We need lane-crossing shuffle
        //    but don't need to blend. Other options exist but are slower on recent intel
        //    processors as vpermt2ps is one-cycle throughput (w/ 3 cycle latency).
        __m512 xid = _mm512_permutex2var_ps(
            _mm512_castps256_ps512(xi),
            _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7),
            _mm512_castps256_ps512(xi));

        __m512 v1;
        __m512 v2;
        float *out_1, *out_2;

        alignas(16) int indices[8];
        _mm256_store_epi32(indices, x_ceili);

        // Unrolled loop to compute 8 values, two at once.
        compute(_mm512_permute_ps(xid, 0), dd + 2 * i, v1, v2);
        accumulate_add_complex_interleaved_aligned(du, indices[0], v1);
        accumulate_add_complex_interleaved_aligned(du, indices[1], v2);

        compute(_mm512_permute_ps(xid, 0b01010101), dd + 2 * i + 4, v1, v2);
        accumulate_add_complex_interleaved_aligned(du, indices[2], v1);
        accumulate_add_complex_interleaved_aligned(du, indices[3], v2);

        compute(_mm512_permute_ps(xid, 0b10101010), dd + 2 * i + 8, v1, v2);
        accumulate_add_complex_interleaved_aligned(du, indices[4], v1);
        accumulate_add_complex_interleaved_aligned(du, indices[5], v2);

        compute(_mm512_permute_ps(xid, 0b11111111), dd + 2 * i + 12, v1, v2);
        accumulate_add_complex_interleaved_aligned(du, indices[6], v1);
        accumulate_add_complex_interleaved_aligned(du, indices[7], v2);
    }
};

static const ker_horner_avx512_w7_r4_op ker_horner_avx512_w7_r4(weights_w7);

} // namespace detail
} // namespace finufft
