// Micro-benchmark for 1-d type-3 operations
// This implements micro-benchmarks using the google benchmark framework to evaluate the performance
// of 1-d type-3 kernels.

#include "../src/kernels/onedim_nuft.h"

#include <benchmark/benchmark.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

extern "C" {

#include "../contrib/legendre_rule_fast.h"
}

namespace {

template <typename FT> FT eval_kernel(FT x, int width) {
    FT ES_c = 4.0 / (width * width);
    FT ES_beta = 2.3 * width;

    if (std::abs(x) > width / 2) {
        return 0.0;
    }

    return std::exp(ES_beta * std::sqrt(1.0 - ES_c * x * x));
}

template <typename T> std::vector<T> generate_random_data(int n, int seed) {
    typedef r123::Philox2x32 RNG;
    RNG rng;

    RNG::ctr_type ctr = {{}};
    RNG::ukey_type key = {{}};
    key[0] = seed;

    std::vector<T> result(n);
    float scale = 0.8 * n;

    for (int i = 0; i < n; i++) {
        ctr[0] = i;
        auto r = rng(ctr, key);
        result[i] = r123::u01<T>(r[0]) * scale + 0.1 * n;
    }

    return result;
}

/** Invoke one-dimensional nuft with given kernel.
 *
 * This function sets-up a problem of the given width and invokes it on the given data.
 *
 * k is the input array, of length nk
 * phihat is an output array, of length nk
 *
 */
template <int nspread, typename FT, typename Fn>
void onedim_nuft_with_kernel(size_t nk, FT const *k, FT *phihat, Fn &&fn) {
    // Setup constants

    FT J2 = nspread / 2.0; // J/2, half-width of ker z-support
    // # quadr nodes in z (from 0 to J/2; reflections will be added)...
    const int q = 2 + nspread;

    FT f[q];
    double z[2 * q], w[2 * q];         // glr needs double
    legendre_compute_glr(2 * q, z, w); // only half the nodes used, eg on (0,1)

    FT zf[q];

    for (int n = 0; n < q; ++n) {
        zf[n] *= z[n] * J2; // quadr nodes for [0,J/2]
        f[n] = J2 * w[n] * eval_kernel(zf[n], nspread);
    }

    // Invoke kernel
    fn(nk, q, f, zf, k, phihat);
}

template <typename T, typename Fn> void benchmark_1d_nuft_kernel(benchmark::State &state, Fn &&fn) {
    auto num_points = state.range(0);
    auto input = generate_random_data<T>(num_points, 0);
    auto output = std::vector<T>(num_points);

    for (auto _ : state) {
        benchmark::ClobberMemory();

        onedim_nuft_with_kernel<8>(num_points, input.data(), output.data(), std::forward<Fn>(fn));
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
