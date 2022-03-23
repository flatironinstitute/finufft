// Tests for polynomial approximation of spreading kernel

#include <cmath>
#include <gtest/gtest.h>

#include "../src/kernels/spread/spread_poly_scalar_impl.h"

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
