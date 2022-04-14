#pragma once

/** This file is based on optimizations done by @mirounga
 * It employs a two-step evaluation + accumulation strategy.
 * We look at the w7 kernel here.
 * 
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <immintrin.h>

#include "spread_kernel_weights.h"

namespace mirounga {

typedef int BIGINT;

template <typename SortIdxT>
inline void spread_subproblem_1d_accumulate(
    SortIdxT sort_indices, BIGINT off1, BIGINT size1, float *du, const float *dd, BIGINT *i1,
    const float *kernel_vals1, BIGINT begin, BIGINT end, int width) {
    int ns = width;                        // a.k.a. w
    float ns2 = (float)ns / 2;             // half spread width
    int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

    for (BIGINT i = 0; i < 2 * size1; ++i) // zero output
        du[i] = 0.0;

    const float *pKer1 = kernel_vals1 + begin * nsPadded;

    __m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
    __m256i _broadcast2 = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
    __m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
    __m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

    switch (nsPadded) {
    case 4:
        for (BIGINT i = begin; i < end; i++) { // loop over NU pts
            BIGINT si = sort_indices[i];
            __m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
            __m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

            // offset rel to subgrid, starts the output indices
            float *pDu = du + 2 * (i1[i] - off1);

            __m256 _k0 = _mm256_castps128_ps256(_mm_load_ps(pKer1 + 0));

            __m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);

            __m256 _du0 = _mm256_loadu_ps(pDu + 0);

            _du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);

            _mm256_storeu_ps(pDu + 0, _du0);

            pKer1 += nsPadded;
        }
        break;
    case 8:
        for (BIGINT i = begin; i < end; i++) { // loop over NU pts
            BIGINT si = sort_indices[i];
            __m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
            __m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

            // offset rel to subgrid, starts the output indices
            float *pDu = du + 2 * (i1[i] - off1);

            __m256 _k0 = _mm256_load_ps(pKer1 + 0);

            __m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
            __m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);

            __m256 _du0 = _mm256_loadu_ps(pDu + 0);
            __m256 _du1 = _mm256_loadu_ps(pDu + 8);

            _du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);
            _du1 = _mm256_fmadd_ps(_dd0, _kk1, _du1);

            _mm256_storeu_ps(pDu + 0, _du0);
            _mm256_storeu_ps(pDu + 8, _du1);

            pKer1 += nsPadded;
        }
        break;
    case 12:
        for (BIGINT i = begin; i < end; i++) { // loop over NU pts
            BIGINT si = sort_indices[i];
            __m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
            __m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

            // offset rel to subgrid, starts the output indices
            float *pDu = du + 2 * (i1[i] - off1);

            __m256 _k0 = _mm256_loadu_ps(pKer1 + 0);
            __m256 _k2 = _mm256_castps128_ps256(_mm_load_ps(pKer1 + 8));

            __m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
            __m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
            __m256 _kk2 = _mm256_permutevar8x32_ps(_k2, _spreadlo);

            __m256 _du0 = _mm256_loadu_ps(pDu + 0);
            __m256 _du1 = _mm256_loadu_ps(pDu + 8);
            __m256 _du2 = _mm256_loadu_ps(pDu + 16);

            _du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);
            _du1 = _mm256_fmadd_ps(_dd0, _kk1, _du1);
            _du2 = _mm256_fmadd_ps(_dd0, _kk2, _du2);

            _mm256_storeu_ps(pDu + 0, _du0);
            _mm256_storeu_ps(pDu + 8, _du1);
            _mm256_storeu_ps(pDu + 16, _du2);

            pKer1 += nsPadded;
        }
        break;
    case 16:
        for (BIGINT i = begin; i < end; i++) { // loop over NU pts
            BIGINT si = sort_indices[i];
            __m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
            __m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

            // offset rel to subgrid, starts the output indices
            float *pDu = du + 2 * (i1[i] - off1);

            __m256 _k0 = _mm256_load_ps(pKer1 + 0);
            __m256 _k2 = _mm256_load_ps(pKer1 + 8);

            __m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
            __m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
            __m256 _kk2 = _mm256_permutevar8x32_ps(_k2, _spreadlo);
            __m256 _kk3 = _mm256_permutevar8x32_ps(_k2, _spreadhi);

            __m256 _du0 = _mm256_loadu_ps(pDu + 0);
            __m256 _du1 = _mm256_loadu_ps(pDu + 8);
            __m256 _du2 = _mm256_loadu_ps(pDu + 16);
            __m256 _du3 = _mm256_loadu_ps(pDu + 24);

            _du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);
            _du1 = _mm256_fmadd_ps(_dd0, _kk1, _du1);
            _du2 = _mm256_fmadd_ps(_dd0, _kk2, _du2);
            _du3 = _mm256_fmadd_ps(_dd0, _kk3, _du3);

            _mm256_storeu_ps(pDu + 0, _du0);
            _mm256_storeu_ps(pDu + 8, _du1);
            _mm256_storeu_ps(pDu + 16, _du2);
            _mm256_storeu_ps(pDu + 24, _du3);

            pKer1 += nsPadded;
        }
        break;
    default:
        // Should never get here
        break;
    }
}

inline void
eval_kernel_bulk_w7_d10(const float *c, float *kernel_vals, const float *x1, const BIGINT size) {
    __m512 _two = _mm512_set1_ps(2.0f);
    __m512 _ns_m_1 = _mm512_set1_ps(6.0f);

    BIGINT size16 = size - size % 16;

    float *ker = kernel_vals;

    __m512 _c0 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 0));
    __m512 _c1 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 8));
    __m512 _c2 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 16));
    __m512 _c3 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 24));
    __m512 _c4 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 32));
    __m512 _c5 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 40));
    __m512 _c6 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 48));
    __m512 _c7 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 56));
    __m512 _c8 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 64));
    __m512 _c9 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 72));

    // main loop
    for (BIGINT i = 0; i < size16; i += 16) {
        __m512 _x_ab = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 0]), _mm256_set1_ps(x1[i + 1]), 1);
        __m512 _x_cd = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 2]), _mm256_set1_ps(x1[i + 3]), 1);
        __m512 _x_ef = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 4]), _mm256_set1_ps(x1[i + 5]), 1);
        __m512 _x_gh = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 6]), _mm256_set1_ps(x1[i + 7]), 1);
        __m512 _x_ij = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 8]), _mm256_set1_ps(x1[i + 9]), 1);
        __m512 _x_kl =
            _mm512_insertf32x8(_mm512_set1_ps(x1[i + 10]), _mm256_set1_ps(x1[i + 11]), 1);
        __m512 _x_mn =
            _mm512_insertf32x8(_mm512_set1_ps(x1[i + 12]), _mm256_set1_ps(x1[i + 13]), 1);
        __m512 _x_op =
            _mm512_insertf32x8(_mm512_set1_ps(x1[i + 14]), _mm256_set1_ps(x1[i + 15]), 1);

        // scale so local grid offset z in [-1,1]
        __m512 _z_ab = _mm512_fmadd_ps(_x_ab, _two, _ns_m_1);
        __m512 _z_cd = _mm512_fmadd_ps(_x_cd, _two, _ns_m_1);
        __m512 _z_ef = _mm512_fmadd_ps(_x_ef, _two, _ns_m_1);
        __m512 _z_gh = _mm512_fmadd_ps(_x_gh, _two, _ns_m_1);
        __m512 _z_ij = _mm512_fmadd_ps(_x_ij, _two, _ns_m_1);
        __m512 _z_kl = _mm512_fmadd_ps(_x_kl, _two, _ns_m_1);
        __m512 _z_mn = _mm512_fmadd_ps(_x_mn, _two, _ns_m_1);
        __m512 _z_op = _mm512_fmadd_ps(_x_op, _two, _ns_m_1);

        __m512 _k_ab = _mm512_fmadd_ps(_c9, _z_ab, _c8);
        __m512 _k_cd = _mm512_fmadd_ps(_c9, _z_cd, _c8);
        __m512 _k_ef = _mm512_fmadd_ps(_c9, _z_ef, _c8);
        __m512 _k_gh = _mm512_fmadd_ps(_c9, _z_gh, _c8);
        __m512 _k_ij = _mm512_fmadd_ps(_c9, _z_ij, _c8);
        __m512 _k_kl = _mm512_fmadd_ps(_c9, _z_kl, _c8);
        __m512 _k_mn = _mm512_fmadd_ps(_c9, _z_mn, _c8);
        __m512 _k_op = _mm512_fmadd_ps(_c9, _z_op, _c8);

        _k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c7);
        _k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c7);
        _k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c7);
        _k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c7);
        _k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c7);
        _k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c7);
        _k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c7);
        _k_op = _mm512_fmadd_ps(_k_op, _z_op, _c7);

        _k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c6);
        _k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c6);
        _k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c6);
        _k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c6);
        _k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c6);
        _k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c6);
        _k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c6);
        _k_op = _mm512_fmadd_ps(_k_op, _z_op, _c6);

        _k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c5);
        _k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c5);
        _k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c5);
        _k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c5);
        _k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c5);
        _k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c5);
        _k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c5);
        _k_op = _mm512_fmadd_ps(_k_op, _z_op, _c5);

        _k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c4);
        _k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c4);
        _k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c4);
        _k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c4);
        _k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c4);
        _k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c4);
        _k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c4);
        _k_op = _mm512_fmadd_ps(_k_op, _z_op, _c4);

        _k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c3);
        _k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c3);
        _k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c3);
        _k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c3);
        _k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c3);
        _k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c3);
        _k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c3);
        _k_op = _mm512_fmadd_ps(_k_op, _z_op, _c3);

        _k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c2);
        _k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c2);
        _k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c2);
        _k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c2);
        _k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c2);
        _k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c2);
        _k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c2);
        _k_op = _mm512_fmadd_ps(_k_op, _z_op, _c2);

        _k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c1);
        _k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c1);
        _k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c1);
        _k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c1);
        _k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c1);
        _k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c1);
        _k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c1);
        _k_op = _mm512_fmadd_ps(_k_op, _z_op, _c1);

        _k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c0);
        _k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c0);
        _k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c0);
        _k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c0);
        _k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c0);
        _k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c0);
        _k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c0);
        _k_op = _mm512_fmadd_ps(_k_op, _z_op, _c0);

        _mm512_store_ps(ker + 0, _k_ab);
        _mm512_store_ps(ker + 16, _k_cd);
        _mm512_store_ps(ker + 32, _k_ef);
        _mm512_store_ps(ker + 48, _k_gh);
        _mm512_store_ps(ker + 64, _k_ij);
        _mm512_store_ps(ker + 80, _k_kl);
        _mm512_store_ps(ker + 96, _k_mn);
        _mm512_store_ps(ker + 112, _k_op);

        ker += 128;
    }

    // short tail
    for (BIGINT i = size16; i < size; i++) {
        __m512 _x_a = _mm512_set1_ps(x1[i]);
        // scale so local grid offset z in [-1,1]
        __m512 _z_a = _mm512_fmadd_ps(_x_a, _two, _ns_m_1);

        __m512 _k_a = _mm512_fmadd_ps(_c9, _z_a, _c8);

        _k_a = _mm512_fmadd_ps(_k_a, _z_a, _c7);
        _k_a = _mm512_fmadd_ps(_k_a, _z_a, _c6);
        _k_a = _mm512_fmadd_ps(_k_a, _z_a, _c5);
        _k_a = _mm512_fmadd_ps(_k_a, _z_a, _c4);
        _k_a = _mm512_fmadd_ps(_k_a, _z_a, _c3);
        _k_a = _mm512_fmadd_ps(_k_a, _z_a, _c2);
        _k_a = _mm512_fmadd_ps(_k_a, _z_a, _c1);
        _k_a = _mm512_fmadd_ps(_k_a, _z_a, _c0);

        _mm256_store_ps(ker, _mm512_castps512_ps256(_k_a));

        ker += 8;
    }
}

std::array<float, 80> make_w7_d10_coeffs() {
    std::array<float, 80> coeffs;

    for (int d = 0; d < finufft::detail::weights_w7.size(); ++d) {
        std::copy_n(finufft::detail::weights_w7[0].begin(), 7, coeffs.begin() + d * 8);
        coeffs[d * 8 + 7] = 0;
    }

    return coeffs;
}

alignas(64) static const std::array<float, 80> w7_d10_coeffs = make_w7_d10_coeffs();

struct identity_index {
    template <typename I> I operator[](I i) const { return i; }
};

inline void spread_subproblem_1d(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) {
    // This is currently only implemented for width = 7 and standard es_beta, es_c parameters.
    // It is a lightweight wrapper around eval_kernel_bulk + spread_subproblem to implement the spread_subproblem_1d interface.
    // Based on the existing code, it sets up a block size of 8192

    int width_padded = 4 * (1 + (width - 1) / 4);

    // Process by block to reduce cache usage.
    // Note: unfortunately, not faster in tests.
    // Benchmark uses 2 << 16 elements, so set block size to same value here.
    const int block_size = 2 << 16;

    float *kernel_values = static_cast<float*>(aligned_alloc(64, sizeof(float) * width_padded * block_size));
    BIGINT *i1 = static_cast<BIGINT*>(aligned_alloc(64, sizeof(BIGINT) * block_size));

    for(int i = 0; i < M; i += block_size)  {
        int thisblock_size = std::min({block_size, (int)M - i});

        std::transform(kx + i, kx + i + thisblock_size, i1, [width](float x) { return (BIGINT)std::ceil(x - 0.5f * width); });

        eval_kernel_bulk_w7_d10(w7_d10_coeffs.data(), kernel_values, kx, thisblock_size);
        spread_subproblem_1d_accumulate(
            identity_index{}, off1, size1, du, dd, i1, kernel_values, 0, thisblock_size, width);
    }

    free(kernel_values);
    free(i1);
}

} // namespace mirounga
