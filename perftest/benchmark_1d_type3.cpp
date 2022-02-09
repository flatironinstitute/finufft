// Micro-benchmark for 1-d type-3 operations
// This implements micro-benchmarks using the google benchmark framework to evaluate the performance
// of 1-d type-3 kernels.

#include "../src/kernels/onedim_nuft.h"

#include <benchmark/benchmark.h>

#include "testing_utils.h"

namespace {

template <typename T, typename Fn> void benchmark_1d_nuft_kernel(benchmark::State &state, Fn &&fn) {
    auto num_points = state.range(0);
    auto input = finufft::testing::generate_random_data<T>(num_points, 0);
    auto output = std::vector<T>(num_points, 0.0f);

    for (auto _ : state) {
        benchmark::ClobberMemory();

        finufft::testing::onedim_nuft_with_method<8>(num_points, input.data(), output.data(), std::forward<Fn>(fn));
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * num_points);
}

// Utility macro to correctly invoke the kernel (disambiguates overloads and function objects)
#define FINUFFT_INVOKE_KERNEL(name, T)                                                             \
    [](size_t nk, size_t q, T const *f, T const *zf, T const *k, T *phihat) {                      \
        name(nk, q, f, zf, k, phihat);                                                             \
    }

#define FINUFFT_DEFINE_BENCHMARK(suffix, kernel)                                                   \
    void benchmark_1d_nuft_f32_##suffix(benchmark::State &state) {                                 \
        benchmark_1d_nuft_kernel<float>(state, FINUFFT_INVOKE_KERNEL(kernel, float));              \
    }                                                                                              \
    void benchmark_1d_nuft_f64_##suffix(benchmark::State &state) {                                 \
        benchmark_1d_nuft_kernel<double>(state, FINUFFT_INVOKE_KERNEL(kernel, double));            \
    }

FINUFFT_DEFINE_BENCHMARK(scalar, finufft::onedim_nuft_kernel_scalar)
FINUFFT_DEFINE_BENCHMARK(sse4, finufft::onedim_nuft_kernel_sse4)
FINUFFT_DEFINE_BENCHMARK(avx2, finufft::onedim_nuft_kernel_avx2)
FINUFFT_DEFINE_BENCHMARK(avx512, finufft::onedim_nuft_kernel_avx512)

} // namespace

#define FINUFFT_INSTANTIATE_BENCHMARK(prec, suffix)                                                \
    BENCHMARK(benchmark_1d_nuft_##prec##_##suffix)->RangeMultiplier(4)->Range(1024, 2 << 13);

#define FINUFFT_INSTANTIATE_ALL_BENCHMARKS(prec)                                                   \
    FINUFFT_INSTANTIATE_BENCHMARK(prec, scalar)                                                    \
    FINUFFT_INSTANTIATE_BENCHMARK(prec, sse4)                                                      \
    FINUFFT_INSTANTIATE_BENCHMARK(prec, avx2)                                                      \
    FINUFFT_INSTANTIATE_BENCHMARK(prec, avx512)

FINUFFT_INSTANTIATE_ALL_BENCHMARKS(f32)
FINUFFT_INSTANTIATE_ALL_BENCHMARKS(f64)
