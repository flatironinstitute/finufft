// Micro-benchmark for 1-d spread operations.
// This implements micro-benchmarks using the google benchmark framework to evaluate the performance of 1-d spreading kernels.

#include <vector>

#include <benchmark/benchmark.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include <spreadinterp.h>

#include "../test/testing_utilities.h"


void spread_subproblem_1d(BIGINT off1, BIGINT size1, float *du, BIGINT M, float *kx, float  *dd, const spread_opts& opts);

namespace {

#ifdef FLT
#undef FLT
#endif

#define FLT float
__attribute__((always_inline)) inline void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
    FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
    // insert the auto-generated code which expects z, w args, writes to ker...
    #include "../src/ker_horner_allw_loop.c"
}

template<int w>
__attribute__((always_inline)) inline void accumulate_kernel_vec_horner(float* out, const float x, float w_re, float w_im) {
    const int w_rounded = (w + 3) / 4 * 4;

    float ker[w_rounded];
    float z = 2 * x + w - 1.0;

#include "../src/ker_horner_allw_loop.c"

    for (int j = 0; j < w_rounded; ++j) {
        out[2 * j] += w_re * ker[j];
        out[2 * j + 1] += w_im * ker[j];
    }
}

#undef FLT

std::vector<float> generate_random_data(int n, int seed) {
    std::vector<float> result(n);
    finufft::fill_random(result.data(), n, seed, 0.1f * n, 0.8f * n);

    return result;
}

void benchmark_spread_subproblem_1d(benchmark::State& state) {
    int num_points = state.range(0);

    auto positions = generate_random_data(num_points, 0);
    auto strengths = generate_random_data(num_points * 2, 1);

    spread_opts opts;
    setup_spreader(opts, 1e-6f, 2.0, 1, 0, 1, 1);

    for(auto _ : state) {
        benchmark::ClobberMemory();

        std::vector<float> result(num_points * 2);
        spread_subproblem_1d(0, result.size() / 2, result.data(), positions.size(), positions.data(), strengths.data(), opts);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * num_points);
}

template<typename Fn>
void eval_kernel_impl(benchmark::State& state, int w, Fn const& accumulate) {
    const int unroll = 64;

    int out_size = (w + 3) / 4 * 4;

    std::vector<float> result(2 * out_size);

    const auto data = generate_random_data(10000, 0);
    const auto weights = generate_random_data(2 * 10000, 1);

    size_t ctr = 0;

    for(auto _ : state) {
        std::fill(result.begin(), result.end(), 0.0f);

        benchmark::ClobberMemory();

        for(int j = 0; j < unroll; ++j) {
            accumulate(result.data(), data[ctr], weights[2 * ctr], weights[2 * ctr + 1]);
            ctr += 1;
        }

        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(result.data());

        ctr += unroll;
        if (ctr >= data.size() - unroll) {
            ctr = 0;
        }
    }

    state.SetItemsProcessed(state.iterations() * unroll);
}

void benchmark_eval_kernel(benchmark::State& state) {
    const int unroll = 64;

    int w = state.range(0);
    int out_size = (w + 3) / 4 * 4;

    std::vector<float> result(2 * out_size);
    std::vector<float> buffer(out_size);

    eval_kernel_impl(state, state.range(0), [&](float* out, const float x, const float w_re, const float w_im) {
        eval_kernel_vec_Horner(buffer.data(), x, w);

        for(int j = 0; j < out_size; ++j) {
            out[2 * j] += w_re * buffer[j];
            out[2 * j + 1] += w_im * buffer[j];
        }
    });
}

void benchmark_eval_kernel_7(benchmark::State& state) {
    const int unroll = 64;
    const int w = 7;

    int out_size = (w + 3) / 4 * 4;

    std::vector<float> buffer(2 * out_size);

    eval_kernel_impl(state, w, [&](float* out, const float x, const float w_re, const float w_im) {
        accumulate_kernel_vec_horner<w>(out, x, w_re, w_im);
    });
}

}

BENCHMARK(benchmark_spread_subproblem_1d)->RangeMultiplier(4)->Range(128, 1<<14)->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_eval_kernel)->Arg(4)->Arg(6)->Arg(7)->Arg(8);
BENCHMARK(benchmark_eval_kernel_7);

