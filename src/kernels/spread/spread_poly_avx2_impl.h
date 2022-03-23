#pragma once

#include <cassert>
#include <immintrin.h>

namespace finufft {
namespace detail {

inline void accumulate_kernel_result_avx2(__m256 result, float *out, float w_re, float w_im) {
    // TODO: we can save a shuffle by pre-shuffling the lanes of the operation.

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

__m256 inline __attribute__((always_inline)) evaluate_polynomial_horner_avx2(__m256 z, __m256 c0) { return c0; }

template <typename... Ts> __m256 inline __attribute__((always_inline)) evaluate_polynomial_horner_avx2(__m256 z, __m256 c0, Ts... c) {
    return _mm256_fmadd_ps(z, evaluate_polynomial_horner_avx2(z, c...), c0);
}

template <typename... Ts> __m256 evaluate_polynomial_horner_avx2_memory(__m256 z, Ts... mem_c) {
    return evaluate_polynomial_horner_avx2(z, _mm256_load_ps(mem_c)...);
}

struct ker_horner_avx2_w4_x2 {
    // This kernel packs the computation of two elements (requiring 8 locations) into a single AVX2 lane.

    constexpr static const int width = 4;
    constexpr static const int stride = 2;
    constexpr static const int required_elements = 2;

    constexpr static const int degree = 7;
    constexpr static const double beta = 9.52;

    static constexpr float c0[] = { +5.4284366850212916e+02, +1.0073871433088410e+04, +1.0073871433088414e+04, +5.4284366850212893e+02, +5.4284366850212916e+02, +1.0073871433088410e+04, +1.0073871433088414e+04, +5.4284366850212893e+02 };
    static constexpr float c1[] = { +1.4650917259256960e+03, +6.1905285583602945e+03, -6.1905285583602945e+03, -1.4650917259256962e+03, +1.4650917259256960e+03, +6.1905285583602945e+03, -6.1905285583602945e+03, -1.4650917259256962e+03 };
    static constexpr float c2[] = { +1.4186910680718358e+03, -1.3995339862725548e+03, -1.3995339862725618e+03, +1.4186910680718338e+03, +1.4186910680718358e+03, -1.3995339862725548e+03, -1.3995339862725618e+03, +1.4186910680718338e+03 };
    static constexpr float c3[] = { +5.1133995502497521e+02, -1.4191608683682964e+03, +1.4191608683682950e+03, -5.1133995502497589e+02, +5.1133995502497521e+02, -1.4191608683682964e+03, +1.4191608683682950e+03, -5.1133995502497589e+02 };
    static constexpr float c4[] = { -4.8293622641173734e+01, +3.9393732546129058e+01, +3.9393732546122692e+01, -4.8293622641175475e+01, -4.8293622641173734e+01, +3.9393732546129058e+01, +3.9393732546122692e+01, -4.8293622641175475e+01 };
    static constexpr float c5[] = { -7.8386867802392629e+01, +1.4918904800408814e+02, -1.4918904800408717e+02, +7.8386867802392331e+01, -7.8386867802392629e+01, +1.4918904800408814e+02, -1.4918904800408717e+02, +7.8386867802392331e+01 };
    static constexpr float c6[] = { -1.0039212571700405e+01, +5.0626747735619313e+00, +5.0626747735626703e+00, -1.0039212571700107e+01, -1.0039212571700405e+01, +5.0626747735619313e+00, +5.0626747735626703e+00, -1.0039212571700107e+01 };

    void operator()(float* __restrict du, float const* __restrict kx, float const* __restrict dd, std::size_t i) {
        float x1 = kx[i];
        float x2 = kx[i + 1];

        float i1f = std::ceil(x1 - 0.5f * width);
        float i2f = std::ceil(x2 - 0.5f * width);

        std::size_t i1 = static_cast<std::size_t>(i1f);
        std::size_t i2 = static_cast<std::size_t>(i2f);

        float xi1 = i1f - x1;
        float xi2 = i2f - x2;

        __m256 x = _mm256_setr_ps(xi1, xi1, xi1, xi1, xi2, xi2, xi2, xi2);
        __m256 z = _mm256_add_ps(_mm256_add_ps(x, x), _mm256_set1_ps(width - 1.0f));
        __m256 k = evaluate_polynomial_horner_avx2_memory(z, c0, c1, c2, c3, c4, c5, c6);

        __m256 w_re = _mm256_set_m128(_mm_set1_ps(dd[2 * i]), _mm_set1_ps(dd[2 * i + 2]));
        __m256 w_im = _mm256_set_m128(_mm_set1_ps(dd[2 * i + 1]), _mm_set1_ps(dd[2 * i + 3]));

        __m256 k_re = _mm256_mul_ps(k, w_re);
        __m256 k_im = _mm256_mul_ps(k, w_im);

        // First interleave in lane
        __m256 lo_in = _mm256_unpacklo_ps(k_re, k_im);
        __m256 hi_in = _mm256_unpackhi_ps(k_re, k_im);

        // Then interleave between-lanes
        __m256 lo_full = _mm256_permute2f128_ps(lo_in, hi_in, 0x20);
        __m256 hi_full = _mm256_permute2f128_ps(lo_in, hi_in, 0x31);

        float* out_1 = du + 2 * i1;
        float* out_2 = du + 2 * i2;

        // Load output and accumulate
        // Note: must do them in series, as they may alias.
        __m256 out_lo = _mm256_loadu_ps(out_1);
        out_lo = _mm256_add_ps(out_lo, lo_full);
        _mm256_storeu_ps(out_1, out_lo);

        __m256 out_hi = _mm256_loadu_ps(out_2);
        out_hi = _mm256_add_ps(out_hi, hi_full);
        _mm256_storeu_ps(out_2, out_hi);
    }
};

struct ker_horner_avx2_w5_x3 {
    // This kernel packs the computation of three elements (requiring 15 locations)
    // into two AVX2 lanes (16 locations) to avoid extraneous padding.
    // Note that the shuffle here is not optimized, and more thought is required
    // to determine the optimal shuffle.

    constexpr static const int width = 5;
    constexpr static const int stride = 3;
    constexpr static const int required_elements = 4;

    constexpr static const int degree = 8;
    constexpr static const double beta = 11.5;

    static constexpr float c0l0[] = { +9.9223677575397733e+02, +3.7794697666613312e+04, +9.8715771010760684e+04, +3.7794697666613363e+04, +9.9223677575397721e+02, +9.9223677575397733e+02, +3.7794697666613312e+04, +9.8715771010760684e+04 };
    static constexpr float c1l0[] = { +3.0430174925083870e+03, +3.7938404259811447e+04, -9.5212726591853425e-12, -3.7938404259811447e+04, -3.0430174925083857e+03, +3.0430174925083870e+03, +3.7938404259811447e+04, -9.5212726591853425e-12 };
    static constexpr float c2l0[] = { +3.6092689177271232e+03, +7.7501368899498957e+03, -2.2704627332475007e+04, +7.7501368899498430e+03, +3.6092689177271213e+03, +3.6092689177271232e+03, +7.7501368899498957e+03, -2.2704627332475007e+04 };
    static constexpr float c3l0[] = { +1.9990077310495431e+03, -3.8875294641277169e+03, +6.2020485180220895e-12, +3.8875294641277155e+03, -1.9990077310495435e+03, +1.9990077310495431e+03, -3.8875294641277169e+03, +6.2020485180220895e-12 };
    static constexpr float c4l0[] = { +4.0071733590403858e+02, -1.5861137916762891e+03, +2.3839858699097495e+03, -1.5861137916763146e+03, +4.0071733590403568e+02, +4.0071733590403858e+02, -1.5861137916762891e+03, +2.3839858699097495e+03 };
    static constexpr float c5l0[] = { -9.1301168206168228e+01, +1.2316471075214720e+02, +2.1150843077612209e-13, -1.2316471075214298e+02, +9.1301168206166679e+01, -9.1301168206168228e+01, +1.2316471075214720e+02, +2.1150843077612209e-13 };
    static constexpr float c6l0[] = { -5.5339722671222880e+01, +1.1960590540261835e+02, -1.5249941358312162e+02, +1.1960590540262601e+02, -5.5339722671222347e+01, -5.5339722671222880e+01, +1.1960590540261835e+02, -1.5249941358312162e+02 };
    static constexpr float c7l0[] = { -3.3762488150349141e+00, +2.2839981872915227e+00, +3.5935329765699997e-12, -2.2839981873024771e+00, +3.3762488150345660e+00, -3.3762488150349141e+00, +2.2839981872915227e+00, +3.5935329765699997e-12 };

    static constexpr float c0l1[] = { +3.7794697666613363e+04, +9.9223677575397721e+02, +9.9223677575397733e+02, +3.7794697666613312e+04, +9.8715771010760684e+04, +3.7794697666613363e+04, +9.9223677575397721e+02, +0.0000000000000000e+00 };
    static constexpr float c1l1[] = { -3.7938404259811447e+04, -3.0430174925083857e+03, +3.0430174925083870e+03, +3.7938404259811447e+04, -9.5212726591853425e-12, -3.7938404259811447e+04, -3.0430174925083857e+03, +0.0000000000000000e+00 };
    static constexpr float c2l1[] = { +7.7501368899498430e+03, +3.6092689177271213e+03, +3.6092689177271232e+03, +7.7501368899498957e+03, -2.2704627332475007e+04, +7.7501368899498430e+03, +3.6092689177271213e+03, +0.0000000000000000e+00 };
    static constexpr float c3l1[] = { +3.8875294641277155e+03, -1.9990077310495435e+03, +1.9990077310495431e+03, -3.8875294641277169e+03, +6.2020485180220895e-12, +3.8875294641277155e+03, -1.9990077310495435e+03, +0.0000000000000000e+00 };
    static constexpr float c4l1[] = { -1.5861137916763146e+03, +4.0071733590403568e+02, +4.0071733590403858e+02, -1.5861137916762891e+03, +2.3839858699097495e+03, -1.5861137916763146e+03, +4.0071733590403568e+02, +0.0000000000000000e+00 };
    static constexpr float c5l1[] = { -1.2316471075214298e+02, +9.1301168206166679e+01, -9.1301168206168228e+01, +1.2316471075214720e+02, +2.1150843077612209e-13, -1.2316471075214298e+02, +9.1301168206166679e+01, +0.0000000000000000e+00 };
    static constexpr float c6l1[] = { +1.1960590540262601e+02, -5.5339722671222347e+01, -5.5339722671222880e+01, +1.1960590540261835e+02, -1.5249941358312162e+02, +1.1960590540262601e+02, -5.5339722671222347e+01, +0.0000000000000000e+00 };
    static constexpr float c7l1[] = { -2.2839981873024771e+00, +3.3762488150345660e+00, -3.3762488150349141e+00, +2.2839981872915227e+00, +3.5935329765699997e-12, -2.2839981873024771e+00, +3.3762488150345660e+00, +0.0000000000000000e+00 };

    void operator()(float* __restrict du, float const* __restrict kx, float const* __restrict dd, std::size_t i) {
        __m128 neg_ns2 = _mm_set1_ps(-0.5f * width);

        __m128 x = _mm_loadu_ps(kx + i);
        __m128 i1f = _mm_ceil_ps(_mm_add_ps(x, neg_ns2));
        __m128 xi = _mm_sub_ps(i1f, x);

        xi = _mm_min_ps(xi, neg_ns2);
        xi = _mm_max_ps(xi, _mm_add_ps(neg_ns2, _mm_set1_ps(1.0f)));

        __m128i i1 = _mm_cvtps_epi32(i1f);

        __m128 z = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(2.0f), xi), _mm_set1_ps(width - 1.0f));

        __m256 z1 = _mm256_setr_ps(z[0], z[0], z[0], z[0], z[0], z[1], z[1], z[1]);
        __m256 z2 = _mm256_setr_ps(z[2], z[2], z[2], z[3], z[3], z[3], z[3], z[3]);

        __m256 k_l0 = evaluate_polynomial_horner_avx2_memory(z1, c0l0, c1l0, c2l0, c3l0, c4l0, c5l0, c6l0, c7l0);
        __m256 k_l1 = evaluate_polynomial_horner_avx2_memory(z2, c0l1, c1l1, c2l1, c3l1, c4l1, c5l1, c6l1, c7l1);

        __m256 k1 = _mm256_setr_ps(k_l0[0], k_l0[1], k_l0[2], k_l0[3], k_l0[4], 0.0f, 0.0f, 0.0f);
        __m256 k2 = _mm256_setr_ps(k_l0[5], k_l0[6], k_l0[7], k_l1[0], k_l1[1], 0.0f, 0.0f, 0.0f);
        __m256 k3 = _mm256_setr_ps(k_l1[2], k_l1[3], k_l1[4], k_l1[5], k_l1[6], 0.0f, 0.0f, 0.0f);

        accumulate_kernel_result_avx2(k1, du + 2 * _mm_extract_epi32(i1, 0), dd[2 * i], dd[2 * i + 1]);
        accumulate_kernel_result_avx2(k2, du + 2 * _mm_extract_epi32(i1, 1), dd[2 * i + 2], dd[2 * i + 3]);
        accumulate_kernel_result_avx2(k3, du + 2 * _mm_extract_epi32(i1, 2), dd[2 * i + 4], dd[2 * i + 5]);
    }
};

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
        __m256 z = _mm256_set1_ps(2 * x + width - 1.0);
        __m256 k = evaluate_polynomial_horner_avx2_memory(
            z, c0d, c1d, c2d, c3d, c4d, c5d, c6d, c7d, c8d, c9d);
        accumulate_kernel_result_avx2(k, out, w_re, w_im);
    }
};

struct ker_horner_avx2_w8 {
    constexpr static const int width = 8;
    constexpr static const int degree = 11;
    constexpr static const double beta = 18.4;

    // clang-format off
    alignas(32) static constexpr float c0[] = { +7.3898000697446732e+03, +1.7297637497600003e+06, +2.5578341605285820e+07, +8.4789650417103425e+07, +8.4789650417103484e+07, +2.5578341605285816e+07, +1.7297637497600005e+06, +7.3898000697446632e+03 };
    alignas(32) static constexpr float c1[] = { +3.0719636811267595e+04, +3.1853145713323969e+06, +2.3797981861403719e+07, +2.4569731244678490e+07, -2.4569731244678497e+07, -2.3797981861403715e+07, -3.1853145713323960e+06, -3.0719636811267643e+04 };
    alignas(32) static constexpr float c2[] = { +5.4488498478251735e+04, +2.4101183255475149e+06, +6.4554051283428427e+06, -8.9200440393090490e+06, -8.9200440393090770e+06, +6.4554051283428138e+06, +2.4101183255475112e+06, +5.4488498478251742e+04 };
    alignas(32) static constexpr float c3[] = { +5.3926359802542196e+04, +9.0469037926849490e+05, -6.0897036277695163e+05, -3.0743852105799834e+06, +3.0743852105799885e+06, +6.0897036277695210e+05, -9.0469037926849560e+05, -5.3926359802542211e+04 };
    alignas(32) static constexpr float c4[] = { +3.2444118016247536e+04, +1.3079802224392018e+05, -5.8652889370130643e+05, +4.2333306008143269e+05, +4.2333306008142326e+05, -5.8652889370133332e+05, +1.3079802224391738e+05, +3.2444118016247525e+04 };
    alignas(32) static constexpr float c5[] = { +1.1864306345505318e+04, -2.2700360645708701e+04, -5.0713607251417554e+04, +1.8308704458210853e+05, -1.8308704458211191e+05, +5.0713607251416361e+04, +2.2700360645707970e+04, -1.1864306345505303e+04 };
    alignas(32) static constexpr float c6[] = { +2.2812256770903441e+03, -1.1569135767377295e+04, +2.0942387020803755e+04, -1.1661592834938250e+04, -1.1661592834934985e+04, +2.0942387020804625e+04, -1.1569135767376711e+04, +2.2812256770903568e+03 };
    alignas(32) static constexpr float c7[] = { +8.5503535636797103e+00, -9.7513976461231766e+02, +3.8242995179182967e+03, -6.9201295567235575e+03, +6.9201295567246671e+03, -3.8242995179203367e+03, +9.7513976461124082e+02, -8.5503535637092227e+00 };
    alignas(32) static constexpr float c8[] = { -1.0230637348338531e+02, +2.8246898554217074e+02, -3.8638201738577567e+02, +1.9106407995489519e+02, +1.9106407997370167e+02, -3.8638201736398116e+02, +2.8246898554517196e+02, -1.0230637348336067e+02 };
    alignas(32) static constexpr float c9[] = { -1.9200143062936828e+01, +6.1692257626735497e+01, -1.2981109187870936e+02, +1.8681284209970522e+02, -1.8681284209923302e+02, +1.2981109187916846e+02, -6.1692257625523915e+01, +1.9200143062952392e+01 };
    alignas(32) static constexpr float c10[] = { +3.7894993757854711e-01, -1.7334408851563938e+00, +2.5271183935574721e+00, -1.2600964146266189e+00, -1.2600963959530540e+00, +2.5271184111954108e+00, -1.7334408828250987e+00, +3.7894993761309681e-01 };
    // clang-format on

    void operator()(float *out, float x, float w_re, float w_im) {
        __m256 z = _mm256_set1_ps(2 * x + width - 1.0);
        __m256 k =
            evaluate_polynomial_horner_avx2_memory(z, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10);
        accumulate_kernel_result_avx2(k, out, w_re, w_im);
    }
};

typedef std::tuple<ker_horner_avx2_w7, ker_horner_avx2_w8> all_avx2_float_accumulators_tuple;

} // namespace detail
} // namespace finufft