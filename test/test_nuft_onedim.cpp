// Tests for various 1-d type 3 NUFT implementations

#include <iostream>
#include <numeric>
#include <tuple>

#include "../perftest/testing_utils.h"
#include <gtest/gtest.h>

#include "../src/kernels/dispatch.h"
#include "../src/kernels/onedim_nuft.h"

// Utility macro to correctly invoke the kernel (disambiguates overloads and function objects)
#define FINUFFT_INVOKE_KERNEL(name, T)                                                             \
    [](size_t nk, size_t q, T const *f, T const *zf, T const *k, T *phihat) {                      \
        name(nk, q, f, zf, k, phihat);                                                             \
    }

namespace {
template <typename T, typename Fn>
std::tuple<std::vector<T>, std::vector<T>>
run_nuft_random(size_t num_points, int seed, Fn &&kernel) {
    auto input = finufft::testing::generate_random_data<T>(num_points, 0);
    auto output_expected = std::vector<T>(num_points, 0.0f);
    auto output_actual = std::vector<T>(num_points, 0.0f);

    finufft::testing::onedim_nuft_with_method<8>(
        num_points, input.data(), output_actual.data(), std::forward<Fn>(kernel));
    finufft::testing::onedim_nuft_with_method<8>(
        num_points,
        input.data(),
        output_expected.data(),
        FINUFFT_INVOKE_KERNEL(finufft::onedim_nuft_kernel_scalar, T));

    return std::make_tuple(output_expected, output_actual);
}

} // namespace

#define MAKE_TEST(SUFFIX, TYPE, DISPATCH)                                                          \
    TEST(OneDimKernel, SUFFIX##_##TYPE) {                                                          \
        if (finufft::get_current_capability() < DISPATCH) {                                        \
            GTEST_SKIP() << "Instruction set " << #SUFFIX << " not supported";                     \
            return;                                                                                \
        }                                                                                          \
        auto num_points = 123;                                                                     \
        auto input = finufft::testing::generate_random_data<TYPE>(num_points, 0);                  \
        std::vector<TYPE> output_expected;                                                         \
        std::vector<TYPE> output_actual;                                                           \
        std::tie(output_expected, output_actual) = run_nuft_random<TYPE>(                          \
            num_points, 0, FINUFFT_INVOKE_KERNEL(finufft::onedim_nuft_kernel_##SUFFIX, TYPE));     \
        for (size_t i = 0; i < num_points; i++) {                                                  \
            EXPECT_FLOAT_EQ(output_expected[i], output_actual[i]) << "i = " << i;                  \
        }                                                                                          \
    }

MAKE_TEST(sse4, float, finufft::Dispatch::SSE4)
MAKE_TEST(avx2, float, finufft::Dispatch::AVX2)
MAKE_TEST(avx512, float, finufft::Dispatch::AVX512)

MAKE_TEST(sse4, double, finufft::Dispatch::SSE4)
MAKE_TEST(avx2, double, finufft::Dispatch::AVX2)
MAKE_TEST(avx512, double, finufft::Dispatch::AVX512)

#undef MAKE_TEST
