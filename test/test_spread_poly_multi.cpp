// Tests for polynomial spreaders with multi-evaluation support.

#include <numeric>
#include <gtest/gtest.h>

#include "../src/kernels/spread/spread_impl.h"
#include "../src/kernels/spread/spread_poly_avx2_impl.h"
#include "../src/kernels/spread/spread_poly_avx512_impl.h"
#include "../src/kernels/spread/spread_poly_scalar_impl.h"
#include "testing_utilities.h"

TEST(SpreadPolyMulti, w4x2) {
    std::vector<float> output(24, 0.0f);
    std::vector<float> output_expected(24, 0.0f);
    std::vector<float> kx = {5, 6.5};
    std::vector<float> dd = {1.0, 2.0, 3.0, 4.0};

    finufft::detail::ker_horner_avx2_w4_x2 ker;

    ker(output.data(), kx.data(), dd.data(), 0);

    finufft::detail::spread_subproblem_1d_impl(
        0,
        output.size() / 2,
        output_expected.data(),
        kx.size(),
        kx.data(),
        dd.data(),
        ker.width,
        finufft::detail::VectorKernelAccumulator<finufft::detail::ker_horner_scalar_2, 4>{});

    for (int i = 0; i < output.size(); i++) {
        EXPECT_FLOAT_EQ(output[i], output_expected[i]) << "i = " << i;
    }
}

TEST(SpreadPolyMulti, w5x3) {
    std::vector<float> output(24, 0.0f);
    std::vector<float> output_expected(24, 0.0f);
    std::vector<float> kx = {5.13, 5.35, 6.14, 0};
    std::vector<float> dd = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    finufft::detail::ker_horner_avx2_w5_x3 ker;
    ker(output.data(), kx.data(), dd.data(), 0);

    finufft::detail::spread_subproblem_1d_impl(
        0,
        output.size() / 2,
        output_expected.data(),
        ker.stride,
        kx.data(),
        dd.data(),
        ker.width,
        finufft::detail::VectorKernelAccumulator<finufft::detail::ker_horner_scalar_3, 8>{});

    for (int i = 0; i < output.size(); i++) {
        EXPECT_FLOAT_EQ(output[i], output_expected[i]) << "i = " << i;
    }
}

TEST(AccumulateAdd, Aligned) {
    float* output = static_cast<float*>(std::aligned_alloc(64, sizeof(float) * 32));

    std::vector<float> values(16);
    std::vector<float> result(32);

    std::iota(values.begin(), values.end(), 1);

    for(int i = 0; i < 8; ++i) {
        std::fill_n(output, 32, 0.0f);
        std::fill(result.begin(), result.end(), 0.0f);

        // Copy default
        std::copy(values.begin(), values.end(), result.begin() + 2 * i);

        // Copy vectorized
        finufft::detail::accumulate_add_complex_interleaved_aligned(output, i, _mm512_loadu_ps(values.data()));

        for(int j = 0; j < 32; ++j) {
            EXPECT_FLOAT_EQ(result[j], output[j]) << "i = " << i << ", j = " << j;
        }
    }

    free(output);
}

TEST(SpreadPolyMulti, w7x2) {
    float* output = static_cast<float*>(std::aligned_alloc(64, 32 * sizeof(float)));
    std::fill_n(output, 32, 0.0f);

    std::vector<float> output_expected(24, 0.0f);
    std::vector<float> kx = {5.13, 5.35};
    std::vector<float> dd = {1.0, 2.0, 3.0, 4.0};

    auto const &ker = finufft::detail::ker_horner_avx512_w7;

    finufft::detail::spread_subproblem_1d_impl(
        0,
        output_expected.size() / 2,
        output_expected.data(),
        ker.stride,
        kx.data(),
        dd.data(),
        ker.width,
        finufft::detail::VectorKernelAccumulator<finufft::detail::ker_horner_scalar_5, 8>{});

    ker(output, kx.data(), dd.data(), 0);

    for (int i = 0; i < output_expected.size(); i++) {
        EXPECT_FLOAT_EQ(output[i], output_expected[i]) << "i = " << i;
    }

    free(output);
}

TEST(SpreadPolyMulti, w7x2r4) {
    auto const& ker = finufft::detail::ker_horner_avx512_w7_r4;

    std::vector<float> output(26, 0.0f);
    std::vector<float> output_expected(26, 0.0f);
    std::vector<float> kx(ker.stride);
    std::vector<float> dd(ker.stride * 2);

    finufft::fill_random(kx.data(), kx.size(), 123, 4.0f, 8.0f);
    finufft::fill_random(dd.data(), dd.size(), 456, -1.0f, 1.0f);

    finufft::detail::spread_subproblem_1d_impl(
        0,
        output.size() / 2,
        output_expected.data(),
        ker.stride,
        kx.data(),
        dd.data(),
        ker.width,
        finufft::detail::VectorKernelAccumulator<finufft::detail::ker_horner_scalar_5, 8>{});

    ker(output.data(), kx.data(), dd.data(), 0);

    for (int i = 0; i < output.size(); i++) {
        EXPECT_FLOAT_EQ(output[i], output_expected[i]) << "i = " << i;
    }
}

namespace {

template <std::size_t width, typename T, typename W>
std::array<T, width> evaluate_polynomial(T x, W const &weights) {
    const int degree = std::tuple_size_v<W> - 1;
    std::array<T, width> result;

    for (int i = 0; i < width; i++) {
        if (i >= weights[0].size()) {
            result[i] = 0;
            continue;
        }

        T v = 0;
        for (int d = degree; d >= 0; --d) {
            // Note: must use FMA explicitly to match vector code
            v = std::fma(v, x, weights[d][i]);
        }
        result[i] = v;
    }

    return result;
}

template<std::size_t width, typename T>
std::array<T, 2 * width> interleave(std::array<T, width> const &a, std::array<T, width> const &b) {
    std::array<T, 2 * width> result;
    for (int i = 0; i < width; i++) {
        result[2 * i] = a[i];
        result[2 * i + 1] = b[i];
    }
    return result;
}

template<std::size_t out_width, typename T, typename W>
std::array<T, 2 * out_width> evaluate_kernel(T x, int width, T w_re, T w_im, W const& weights) {
    T z = (x + x) + (width - 1.0f);
    auto k = evaluate_polynomial<out_width>(z, weights);

    std::array<T, out_width> k_re;
    std::array<T, out_width> k_im;

    for(int i = 0; i < out_width; i++) {
        k_re[i] = k[i] * w_re;
        k_im[i] = k[i] * w_im;
    }

    return interleave(k_re, k_im);
}

std::array<float, 16> to_array(__m512 x) {
    std::array<float, 16> result;
    _mm512_storeu_ps(result.data(), x);
    return result;
}

} // namespace

TEST(SpreadPolyMultiCompute, w7x2) {
    auto const &ker = finufft::detail::ker_horner_avx512_w7;
    auto left = -0.5f * ker.width;
    auto right = left + 1;

    auto kx = std::array{left + 0.1f, left + 0.2f};
    auto dd = std::array{1.0f, 2.0f, 3.0f, 4.0f};

    __m512 v1, v2;
    ker.compute(kx, dd, v1, v2);

    auto r0 = to_array(v1);
    auto r0_expected = evaluate_kernel<std::tuple_size_v<decltype(r0)> / 2>(kx[0], ker.width, dd[0], dd[1], finufft::detail::weights_w7);

    for(int i = 0; i < r0.size(); i++) {
        EXPECT_FLOAT_EQ(r0[i], r0_expected[i]) << "i = " << i;
    }

    auto r1 = to_array(v2);
    auto r1_expected = evaluate_kernel<std::tuple_size_v<decltype(r1)> / 2>(kx[1], ker.width, dd[2], dd[3], finufft::detail::weights_w7);

    for(int i = 0; i < r1.size(); i++) {
        EXPECT_FLOAT_EQ(r1[i], r1_expected[i]) << "i = " << i;
    }
}
