// Tests for polynomial approximation of spreading kernel

#include <cmath>
#include <gtest/gtest.h>

#include "../src/kernels/spread/spread_poly_scalar_impl.h"
#include "../src/kernels/spread/spread_poly_avx2_impl.h"

namespace {
    void evaluate_kernel_allw(double x, double* ker, int w) {
        #define FLT double
        FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
#include "../src/ker_horner_allw_loop.c"
        #undef FLT
    }

    template<int w>
    struct MatlabKernel {
        static constexpr int width = w;
        static constexpr int out_width = (w + 3) / 4 * 4;
        static constexpr double beta = 2.3 * w;

        void operator()(double x, double* ker) const noexcept {
            evaluate_kernel_allw(x, ker, w);
        }
    };

    template<typename T>
    void evaluate_kernel_reference(T x, T* ker, int width, double beta) {
        double c = 4.0 / (width * width);

        for(int i = 0; i < width; ++i) {
            auto xi = x + i;
            ker[i] = static_cast<T>(std::exp(beta * std::sqrt(1 - c * xi * xi)));
        }
    }

    template<typename T, typename K>
    double max_difference_to_kernel() {
        T ker_v[K::out_width];
        T ker_r[K::out_width];

        double beta = K::beta;
        double width = K::width;
        double c = 4.0 / (width * width);

        K ker;

        double max_diff = 0.0;
        double scale = std::exp(K::beta);

        for (int i = 0; i < 50; ++i) {
            T x1 = static_cast<T>(i) / 50.0 - width / 2;
            ker(x1, ker_v);
            evaluate_kernel_reference(x1, ker_r, K::width, beta);

            for (int j = 0; j < K::width; ++j) {
                double diff = static_cast<double>(std::abs(ker_v[j] - ker_r[j])) / scale;
                max_diff = std::max(max_diff, diff);
            }
        }

        return max_diff;
    }

    template<typename K>
    class PolyKernelTest : public ::testing::Test {
    };

}

TYPED_TEST_SUITE_P(PolyKernelTest);

TYPED_TEST_P(PolyKernelTest, Accuracy) {
    double tol = 1e-5;

    // Custom tolerance for different widths
    // TODO: compute tolerances corresponding to widths
    switch(TypeParam::width) {
        case 2:
            tol = 5e-2;
            break;
        case 3:
            tol = 1e-2;
            break;
        case 4:
            tol = 1e-3;
            break;
        case 5:
            tol = 1e-4;
            break;
        default:
            break;
    }

    double max_difference_float = max_difference_to_kernel<float, TypeParam>();
    EXPECT_NEAR(max_difference_float, 0.0, tol) << "width: " << TypeParam::width;

    double max_difference_double = max_difference_to_kernel<double, TypeParam>();
    EXPECT_NEAR(max_difference_double, 0.0, tol) << "width: " << TypeParam::width;
}

REGISTER_TYPED_TEST_SUITE_P(PolyKernelTest, Accuracy);

namespace {
    template<typename T>
    struct TupleToTestTypes;

    template<typename... Ts>
    struct TupleToTestTypes<std::tuple<Ts...>> {
        using type = ::testing::Types<Ts...>;
    };

    using KernelTypes = typename TupleToTestTypes<finufft::detail::all_scalar_kernels_tuple>::type;
}

INSTANTIATE_TYPED_TEST_SUITE_P(Poly, PolyKernelTest, KernelTypes);

TEST(PolyKernelTests, AVX2_W7) {
    finufft::detail::ker_horner_avx2_w7 ker_avx2;
    finufft::detail::ker_horner_scalar_5 ker_scalar;

    ASSERT_EQ(ker_avx2.width, ker_scalar.width);
    ASSERT_EQ(ker_avx2.beta, ker_scalar.beta);

    float x = -ker_scalar.width / 2.0f + 0.5f;

    float ker_scalar_v[ker_scalar.out_width];
    float acc_scalar[2 * 8] = {0};
    ker_scalar(x, ker_scalar_v);
    for(int i = 0; i < ker_scalar.width; ++i) {
        acc_scalar[2 * i] += ker_scalar_v[i];
        acc_scalar[2 * i + 1] += ker_scalar_v[i];
    }

    float acc_avx2_v[2 * 8] = {0};
    ker_avx2(acc_avx2_v, x, 1.0, 1.0);

    for(int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(acc_scalar[2 * i], acc_avx2_v[2 * i]);
        EXPECT_FLOAT_EQ(acc_scalar[2 * i + 1], acc_avx2_v[2 * i + 1]);
    }
}
