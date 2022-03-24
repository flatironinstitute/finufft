// Tests for polynomial spreaders with multi-evaluation support.

#include <gtest/gtest.h>

#include "../src/kernels/spread/spread_impl.h"
#include "../src/kernels/spread/spread_poly_avx2_impl.h"
#include "../src/kernels/spread/spread_poly_scalar_impl.h"

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
