#pragma once

#include <immintrin.h>

namespace finufft {
namespace detail {

inline void accumulate_kernel_result_avx2(__m256 result, float* out, float w_re, float w_im) {
    // Evaluate p * w_re and p * w_im
    __m256 w_re_v = _mm256_set1_ps(w_re);
    __m256 w_im_v = _mm256_set1_ps(w_im);

    __m256 k_re = _mm256_mul_ps(result, w_re_v);
    __m256 k_im = _mm256_mul_ps(result, w_im_v);

    // Interleave real and imaginary parts to write to output

    // First interleave in-lane
    __m256 lo_in = _mm256_unpacklo_ps(k_re, k_im);
    __m256 hi_in = _mm256_unpackhi_ps(k_re, k_im);

    // Then interleave between-lanes
    __m256 lo_full = _mm256_permute2f128_ps(lo_in, hi_in, 0x20);
    __m256 hi_full = _mm256_permute2f128_ps(lo_in, hi_in, 0x31);

    // Load output and accumulate
    __m256 out_lo = _mm256_loadu_ps(out);
    __m256 out_hi = _mm256_loadu_ps(out + 8);

    out_lo = _mm256_add_ps(out_lo, lo_full);
    out_hi = _mm256_add_ps(out_hi, hi_full);

    _mm256_storeu_ps(out, out_lo);
    _mm256_storeu_ps(out + 8, out_hi);
}

struct ker_horner_avx2_w7 {
    constexpr static const int width = 7;
    constexpr static const int degree = 10;
    constexpr static const double beta = 16.099999999999998;

    // clang-format off
    alignas(32) static constexpr float c0d[] = { +3.9948351830486999e+03, +5.4715865608590737e+05, +5.0196413492771862e+06, +9.8206709220713433e+06, +5.0196413492771862e+06, +5.4715865608590667e+05, +3.9948351830486918e+03, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c1d[] = { +1.5290160332974701e+04, +8.7628248584320524e+05, +3.4421061790934489e+06, -3.9835867937654257e-10, -3.4421061790934494e+06, -8.7628248584320489e+05, -1.5290160332974716e+04, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c2d[] = { +2.4458227486779255e+04, +5.3904618484139442e+05, +2.4315566181017528e+05, -1.6133959371974319e+06, +2.4315566181017458e+05, +5.3904618484139361e+05, +2.4458227486779258e+04, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c3d[] = { +2.1166189345881678e+04, +1.3382732160223197e+05, -3.3113450969689508e+05, +4.4272606020794267e-10, +3.3113450969689374e+05, -1.3382732160223188e+05, -2.1166189345881670e+04, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c4d[] = { +1.0542795672344848e+04, -7.0739172265099560e+03, -6.5563293056052033e+04, +1.2429734005959056e+05, -6.5563293056056486e+04, -7.0739172265108718e+03, +1.0542795672344835e+04, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c5d[] = { +2.7903491906228473e+03, -1.0975382873973198e+04, +1.3656979541144021e+04, +1.4371041916315029e-10, -1.3656979541143579e+04, +1.0975382873973147e+04, -2.7903491906228460e+03, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c6d[] = { +1.6069721418054596e+02, -1.5518707872250675e+03, +4.3634273936636073e+03, -5.9891976420594583e+03, +4.3634273936652353e+03, -1.5518707872248597e+03, +1.6069721418054911e+02, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c7d[] = { -1.2289277373866585e+02, +2.8583630927770741e+02, -2.8318194617410188e+02, +1.5993327933278236e-10, +2.8318194617325446e+02, -2.8583630927773356e+02, +1.2289277373866099e+02, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c8d[] = { -3.2270164914227294e+01, +9.1892112257324740e+01, -1.6710678096325356e+02, +2.0317049305843574e+02, -1.6710678096044728e+02, +9.1892112258105087e+01, -3.2270164914214298e+01, +0.0000000000000000e+00 };
    alignas(32) static constexpr float c9d[] = { -1.4761409685256255e-01, -9.1862771302337198e-01, +1.2845147729883215e+00, +9.3195459181156685e-11, -1.2845147734751150e+00, +9.1862771305896052e-01, +1.4761409685935642e-01, +0.0000000000000000e+00 };
    // clang-format on

    void operator()(float *out, float x, float w_re, float w_im) {

        // load polynomial coefficients
        __m256 c0 = _mm256_load_ps(c0d);
        __m256 c1 = _mm256_load_ps(c1d);
        __m256 c2 = _mm256_load_ps(c2d);
        __m256 c3 = _mm256_load_ps(c3d);
        __m256 c4 = _mm256_load_ps(c4d);
        __m256 c5 = _mm256_load_ps(c5d);
        __m256 c6 = _mm256_load_ps(c6d);
        __m256 c7 = _mm256_load_ps(c7d);
        __m256 c8 = _mm256_load_ps(c8d);
        __m256 c9 = _mm256_load_ps(c9d);

        __m256 z = _mm256_set1_ps(2 * x + width - 1.0);

        // Horner evaluation of polynomial
        __m256 t0 = _mm256_fmadd_ps(z, c9, c8);
        __m256 t1 = _mm256_fmadd_ps(z, t0, c7);
        __m256 t2 = _mm256_fmadd_ps(z, t1, c6);
        __m256 t3 = _mm256_fmadd_ps(z, t2, c5);
        __m256 t4 = _mm256_fmadd_ps(z, t3, c4);
        __m256 t5 = _mm256_fmadd_ps(z, t4, c3);
        __m256 t6 = _mm256_fmadd_ps(z, t5, c2);
        __m256 t7 = _mm256_fmadd_ps(z, t6, c1);
        __m256 k = _mm256_fmadd_ps(z, t7, c0);

        accumulate_kernel_result_avx2(k, out, w_re, w_im);
    }
};
} // namespace detail
} // namespace finufft