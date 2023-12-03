#include <cmath>
#include <memory>
#include <vector>


#include <benchmark/benchmark.h>

#include "../test/testing_utilities.h"

#define SINGLE
#include <finufft/defs.h>
#undef SINGLE
#define FINUFFT_BIGINT int64_t

namespace {
    template<typename T>
    struct nuft_plan;

    template<>
    struct nuft_plan<float> {
        typedef finufftf_plan type;
    };

    template<>
    struct nuft_plan<double> {
        typedef finufft_plan type;
    };

    int finufft_makeplan(int type, int dim, FINUFFT_BIGINT* n_modes, int iflag, int n_transf, FLT tol, finufftf_plan* plan, finufft_opts* o) {
        return finufftf_makeplan(type, dim, n_modes, iflag, n_transf, tol, plan, o);
    }
    int finufft_setpts(finufftf_plan plan , FINUFFT_BIGINT M, FLT *xj, FLT *yj, FLT *zj, FINUFFT_BIGINT N, FLT *s, FLT *t, FLT *u) {
        return finufftf_setpts(plan, M, xj, yj, zj, N, s, t, u);
    }
    int finufft_execute(finufftf_plan plan, CPX* weights, CPX* result) {
        return finufftf_execute(plan, weights, result);
    }
    int finufft_destroy(finufftf_plan plan) {
        return finufftf_destroy(plan);
    }


    // Utility class which wraps a finufft plan.
    template<typename T>
    class Nuft1DFixture {
        typename nuft_plan<T>::type plan_;

        public:
            Nuft1DFixture(int type, int iflag, FINUFFT_BIGINT num_output_modes, double eps, finufft_opts* opts) {
                FINUFFT_BIGINT n_modes[]={num_output_modes,1,1};

                finufft_makeplan(
                    type,
                    /*dim*/1,
                    n_modes,
                    iflag,
                    /*ntransf*/1,
                    eps,
                    &plan_,
                    opts);
            }

            ~Nuft1DFixture() {
                finufft_destroy(plan_);
            }

            void operator()(FINUFFT_BIGINT size, T* x, std::complex<T>* weights, std::complex<T>* result) {
                finufft_setpts(plan_, size, x, nullptr, nullptr, 0, nullptr, nullptr, nullptr);
                finufft_execute(plan_, weights, result);
            }
    };

    template<typename T>
    void benchmark_nuft_1d_type1(benchmark::State& state) {
        Nuft1DFixture<T> nuft(1, 1, state.range(0), 1e-6, nullptr);

        std::vector<T> x(state.range(0));
        std::vector<std::complex<T>> weights(state.range(0));
        std::vector<std::complex<T>> result(state.range(0));

        finufft::fill_random(x.data(), x.size(), 42, -M_PI, M_PI);
        finufft::fill_random(weights.data(), weights.size(), 43, -1, 1);

        for(auto _ : state) {
            benchmark::ClobberMemory();
            nuft(x.size(), x.data(), weights.data(), result.data());
            benchmark::DoNotOptimize(result[0]);
        }

        state.SetItemsProcessed(state.iterations() * x.size());
    }

    void benchmark_nuft_1d_type1_f32(benchmark::State& state) {
        benchmark_nuft_1d_type1<float>(state);
    }

    void benchmark_nuft_1d_type1_f64(benchmark::State& state) {
        benchmark_nuft_1d_type1<double>(state);
    }
}

BENCHMARK(benchmark_nuft_1d_type1_f32)->RangeMultiplier(4)->Range(128, 1<<14)->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_nuft_1d_type1_f64)->RangeMultiplier(4)->Range(128, 1<<14)->Unit(benchmark::kMicrosecond);

