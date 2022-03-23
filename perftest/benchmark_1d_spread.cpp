#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include "../test/testing_utilities.h"

#include "../src/kernels/spread/spread.h"

#include <spread_opts.h>

// Forward declare manually due to issue with multi-type code.
int setup_spreader(
    spreads_optsd &opts, double eps, double upsampfac, int kerevalmeth, int debug, int showwarn,
    int dim);
int setup_spreader(
    spreads_optsf &opts, float eps, double upsampfac, int kerevalmeth, int debug, int showwarn,
    int dim);

void spread_subproblem_1d(
    BIGINT off1, BIGINT size1, double *du, BIGINT M, double *kx, double *dd,
    const spreads_optsd &opts);
void spread_subproblem_1d(
    BIGINT off1, BIGINT size1, float *du, BIGINT M, float *kx, float *dd,
    const spreads_optsf &opts);

namespace {

// Hack to write a templated code for current implementation with both types.
template <typename T> struct type_to_spread_opts;
template <> struct type_to_spread_opts<double> { using type = spreads_optsd; };
template <> struct type_to_spread_opts<float> { using type = spreads_optsf; };

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

double beta_from_width(int ns) {
    // This sets the default for upsample_fraction=2.0
    double betaoverns = 2.30; // gives decent betas for default sigma=2.0
    if (ns == 2)
        betaoverns = 2.20; // some small-width tweaks...
    if (ns == 3)
        betaoverns = 2.26;
    if (ns == 4)
        betaoverns = 2.38;
    return betaoverns * ns;
}

template <typename T>
void bench_spread_scalar_with_width(
    benchmark::State &state, std::size_t num_points, std::size_t num_output, int width) {
    double beta = beta_from_width(width);
    double c = 4.0 / (width * width);

    auto [kx, dd] = make_spread_data<T>(num_points, width, num_output, 0);
    std::vector<T> du(2 * num_output);

    for (auto _ : state) {
        benchmark::ClobberMemory();
        finufft::detail::spread_subproblem_1d_scalar(
            0, num_output, du.data(), num_output, kx.data(), dd.data(), width, beta, c);
        benchmark::DoNotOptimize(du[du.size() - 1]);
    }

    state.SetBytesProcessed(state.iterations() * num_points * width * sizeof(T));
}

template <typename T>
void bench_spread_avx2_with_width(
    benchmark::State &state, std::size_t num_points, std::size_t num_output, int width) {
    double beta = beta_from_width(width);
    double c = 4.0 / (width * width);

    auto [kx, dd] = make_spread_data<T>(num_points, width, num_output, 0);
    std::vector<T> du(2 * num_output);

    for (auto _ : state) {
        benchmark::ClobberMemory();
        finufft::detail::spread_subproblem_1d_avx2(
            0, num_output, du.data(), num_output, kx.data(), dd.data(), width, beta, c);
        benchmark::DoNotOptimize(du[du.size() - 1]);
    }

    state.SetBytesProcessed(state.iterations() * num_points * width * sizeof(T));
}

template <typename T> void bench_spread_current_with_width(benchmark::State &state, int width) {
    auto num_points = state.range(0);
    auto num_output = num_points;

    typename type_to_spread_opts<T>::type opts;
    double tol = std::pow(10.0, -width + 1.5);
    setup_spreader(opts, static_cast<T>(tol), 2.0, 1, 0, 1, 1);

    auto [kx, dd] = make_spread_data<T>(num_points, width, num_output, 0);
    std::vector<T> du(2 * num_output);

    for (auto _ : state) {
        benchmark::ClobberMemory();
        spread_subproblem_1d(0, num_output, du.data(), num_output, kx.data(), dd.data(), opts);
        benchmark::DoNotOptimize(du[du.size() - 1]);
    }

    state.SetBytesProcessed(state.iterations() * num_points * width * sizeof(T));
}

} // namespace

#define MAKE_BENCHMARK_CURRENT(width, type)                                                        \
    static void bench_spread_current_w##width##_##type(benchmark::State &state) {                  \
        bench_spread_current_with_width<type>(state, width);                                       \
    }                                                                                              \
    BENCHMARK(bench_spread_current_w##width##_##type)->Arg(2 << 14)->Unit(benchmark::kMicrosecond);

MAKE_BENCHMARK_CURRENT(5, float);
MAKE_BENCHMARK_CURRENT(5, double);
MAKE_BENCHMARK_CURRENT(7, float);
MAKE_BENCHMARK_CURRENT(7, double);

#undef MAKE_BENCHMARK_CURRENT

#define MAKE_BENCHMARK(width, instr, type)                                                         \
    static void bench_spread_##instr##_w##width##_##type(benchmark::State &state) {                \
        auto num_points = state.range(0);                                                          \
        bench_spread_##instr##_with_width<type>(state, num_points, num_points, width);             \
    }                                                                                              \
    BENCHMARK(bench_spread_##instr##_w##width##_##type)                                            \
        ->Arg(2 << 14)                                                                             \
        ->Unit(benchmark::kMicrosecond);

MAKE_BENCHMARK(5, scalar, float);
MAKE_BENCHMARK(5, scalar, double);
MAKE_BENCHMARK(7, scalar, float);
MAKE_BENCHMARK(7, scalar, double);

MAKE_BENCHMARK(5, avx2, float);
MAKE_BENCHMARK(5, avx2, double);
MAKE_BENCHMARK(7, avx2, float);
MAKE_BENCHMARK(7, avx2, double);

#undef MAKE_BENCHMARK
