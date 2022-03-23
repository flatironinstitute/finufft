// Tests for various 1-d spreading implementations

#include <tuple>
#include <vector>

#include "../src/kernels/spread/spread.h"
#include "../src/kernels/spread/spread_impl.h"
#include "testing_utilities.h"
#include <gtest/gtest.h>

// Warning: this header does not really correctly handle dual precision
// TODO: fix type handling
#include <spreadinterp.h>

// Forward declare current implementation
void spread_subproblem_1d(
    BIGINT off1, BIGINT size1, double *du, BIGINT M, double *kx, double *dd,
    const SPREAD_OPTS &opts);

namespace {

struct SpreaderConfig {
    int width;
    double beta;
    double c;
};

template <typename T>
std::tuple<std::vector<T>, std::vector<T>>
make_spread_data(std::size_t n, int width, std::size_t num_output, int seed) {
    std::vector<T> kx(n);
    std::vector<T> dd(2 * n);

    finufft::fill_random(kx.data(), n, seed, width / 2.0, num_output - width / 2.0 - 1);
    std::sort(kx.begin(), kx.end());

    finufft::fill_random(dd.data(), 2 * n, seed, -1.0, 1.0);

    return std::make_pair(std::move(kx), std::move(dd));
}

// Logic for configuring the spreader, taken from `setup_spreader`
SpreaderConfig configure_spreader(double eps, double upsample_fraction) {
    int ns;

    if (upsample_fraction == 2.0)                // standard sigma (see SISC paper)
        ns = std::ceil(-std::log10(eps / 10.0)); // 1 digit per power of 10
    else                                         // custom sigma
        ns = std::ceil(
            -std::log(eps) / (M_PI * sqrt(1.0 - 1.0 / upsample_fraction))); // formula, gam=1

    ns = std::max(2, ns); // (we don't have ns=1 version yet)

    // setup for reference kernel eval (via formula): select beta width param...
    // (even when kerevalmeth=1, this ker eval needed for FTs in onedim_*_kernel)
    auto ES_halfwidth = ns / 2; // constants to help (see below routines)
    auto ES_c = 4.0 / (ns * ns);

    double betaoverns = 2.30; // gives decent betas for default sigma=2.0
    if (ns == 2)
        betaoverns = 2.20; // some small-width tweaks...
    if (ns == 3)
        betaoverns = 2.26;
    if (ns == 4)
        betaoverns = 2.38;

    if (upsample_fraction != 2.0) { // again, override beta for custom sigma
        double gamma = 0.97;        // must match devel/gen_all_horner_C_code.m !
        betaoverns =
            gamma * M_PI * (1.0 - 1.0 / (2 * upsample_fraction)); // formula based on cutoff
    }
    auto ES_beta = betaoverns * ns; // set the kernel beta parameter

    return SpreaderConfig{ns, ES_beta, ES_c};
}

} // namespace

TEST(OneDimSpread, Baseline) {
    auto num_points = 10;
    auto num_result = 32;
    auto config = configure_spreader(1e-5, 2.0);

    std::vector<double> kx;
    std::vector<double> dd;

    std::tie(kx, dd) = make_spread_data<double>(num_points, config.width, num_result, 0);

    std::vector<double> result(2 * num_result);
    std::vector<double> result_expected(2 * num_result);

    auto accumulator = finufft::detail::ScalarKernelAccumulator<double>{
        config.width, static_cast<double>(config.beta), static_cast<double>(config.c)};

    finufft::detail::spread_subproblem_1d_impl(
        0, num_result, result.data(), num_points, kx.data(), dd.data(), config.width, accumulator);

    SPREAD_OPTS opts;
    setup_spreader(opts, 1e-5, 2.0, 0, 0, 1, 1);

    spread_subproblem_1d(
        0, num_result, result_expected.data(), num_points, kx.data(), dd.data(), opts);

    for (int i = 0; i < 2 * num_result; i++) {
        EXPECT_DOUBLE_EQ(result[i], result_expected[i]);
    }
}

TEST(OneDimSpread, Scalar) {
    auto num_points = 10;
    auto num_result = 32;
    auto config = configure_spreader(1e-5, 2.0);
    auto width_padded = (config.width + 3) / 4 * 4;

    std::vector<double> kx;
    std::vector<double> dd;

    std::tie(kx, dd) = make_spread_data<double>(num_points, width_padded, num_result, 0);

    std::vector<double> result(2 * num_result);
    std::vector<double> result_expected(2 * num_result);

    auto accumulator = finufft::detail::ScalarKernelAccumulator<double>{
        config.width, static_cast<double>(config.beta), static_cast<double>(config.c)};

    finufft::detail::spread_subproblem_1d_impl(
        0,
        num_result,
        result_expected.data(),
        num_points,
        kx.data(),
        dd.data(),
        config.width,
        accumulator);

    finufft::detail::spread_subproblem_1d_scalar(
        0,
        num_result,
        result.data(),
        num_points,
        kx.data(),
        dd.data(),
        config.width,
        config.beta,
        config.c);

    double tol = 1e-5 * std::exp(config.beta);

    for (int i = 0; i < 2 * num_result; i++) {
        EXPECT_NEAR(result[i], result_expected[i], tol);
    }
}

namespace {
    class SpreadTest : public ::testing::TestWithParam<int> {};

    double tolerance_from_width(int width) {
        return std::pow(10.0, -width + 1.5);
    }
}

TEST_P(SpreadTest, AVX2) {
    auto num_points = 10;
    auto num_result = 128;
    auto config = configure_spreader(tolerance_from_width(GetParam()), 2.0);
    auto width_padded = (config.width + 7) / 8 * 8;

    std::vector<float> kx;
    std::vector<float> dd;

    std::tie(kx, dd) = make_spread_data<float>(num_points, width_padded, num_result, 0);

    std::vector<float> result(2 * num_result);
    std::vector<float> result_expected(2 * num_result);

    finufft::detail::spread_subproblem_1d_avx2(
        0,
        num_result,
        result_expected.data(),
        num_points,
        kx.data(),
        dd.data(),
        config.width,
        config.beta,
        config.c);

    finufft::detail::spread_subproblem_1d_scalar(
        0,
        num_result,
        result.data(),
        num_points,
        kx.data(),
        dd.data(),
        config.width,
        config.beta,
        config.c);

    double tol = 1e-5;
    switch(config.width) {
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
    tol *= std::exp(config.beta);

    for (int i = 0; i < 2 * num_result; i++) {
        EXPECT_NEAR(result[i], result_expected[i], tol);
    }
}

INSTANTIATE_TEST_SUITE_P(OneDimSpread, SpreadTest, ::testing::Range(2, 16));
