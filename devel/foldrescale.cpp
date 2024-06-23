#include "finufft/defs.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <random>
// no vectorize
// #pragma GCC optimize("no-tree-vectorize")
/* local NU coord fold+rescale macro: does the following affine transform to x:
     when p=true:   map [-3pi,-pi) and [-pi,pi) and [pi,3pi)    each to [0,N)
     otherwise,     map [-N,0) and [0,N) and [N,2N)             each to [0,N)
   Thus, only one period either side of the principal domain is folded.
   (It is *so* much faster than slow std::fmod that we stick to it.)
   This explains FINUFFT's allowed input domain of [-3pi,3pi).
   Speed comparisons of this macro vs a function are in devel/foldrescale*.
   The macro wins hands-down on i7, even for modern GCC9.
   This should be done in C++ not as a macro, someday.
*/
#define FOLDRESCALE(x, N, p)                                                \
  (p ? (x + (x >= -PI ? (x < PI ? PI : -PI) : 3 * PI)) * ((FLT)M_1_2PI * N) \
     : (x >= 0.0 ? (x < (FLT)N ? x : x - (FLT)N) : x + (FLT)N))

#define FOLDRESCALE04(x, N, p)                                                       \
  (p ? ((x * FLT(M_1_2PI) + FLT(0.5)) - floor(x * FLT(M_1_2PI) + FLT(0.5))) * FLT(N) \
     : ((x / FLT(N)) - floor(x / FLT(N))) * FLT(N))

#define FOLDRESCALE05(x, N, p)                                                       \
  FLT(N) * (p ? ((x * FLT(M_1_2PI) + FLT(0.5)) - floor(x * FLT(M_1_2PI) + FLT(0.5))) \
              : ((x / FLT(N)) - floor(x / FLT(N))))

inline __attribute__((always_inline)) FLT foldRescale00(FLT x, BIGINT N, bool p) {
  FLT result;
  FLT fN = FLT(N);
  if (p) {
    static constexpr FLT x2pi = FLT(M_1_2PI);
    result                    = x * x2pi + FLT(0.5);
    result -= floor(result);
  } else {
    const FLT invN = FLT(1.0) / fN;
    result         = x * invN;
    result -= floor(result);
  }
  return result * fN;
}

inline __attribute__((always_inline)) FLT foldRescale01(FLT x, BIGINT N, bool p) {
  return p ? (x + (x >= -PI ? (x < PI ? PI : -PI) : 3 * PI)) * ((FLT)M_1_2PI * N)
           : (x >= 0.0 ? (x < (FLT)N ? x : x - (FLT)N) : x + (FLT)N);
}

template<bool p>
inline __attribute__((always_inline)) FLT foldRescale02(FLT x, BIGINT N) {
  if constexpr (p) {
    return (x + (x >= -PI ? (x < PI ? PI : -PI) : 3 * PI)) * ((FLT)M_1_2PI * N);
  } else {
    return (x >= 0.0 ? (x < (FLT)N ? x : x - (FLT)N) : x + (FLT)N);
  }
}

template<bool p>
inline __attribute__((always_inline)) FLT foldRescale03(FLT x, BIGINT N) {
  FLT result;
  FLT fN = FLT(N);
  if constexpr (p) {
    static constexpr FLT x2pi = FLT(M_1_2PI);
    result                    = std::fma(x, x2pi, FLT(0.5));
    result -= floor(result);
  } else {
    const FLT invN = FLT(1.0) / fN;
    result         = x * invN;
    result -= floor(result);
  }
  return result * fN;
}


static std::mt19937_64 gen;
static std::uniform_real_distribution<> dis(-10, 10);
static const auto N = std::uniform_int_distribution<>{0, 1000}(gen);
static std::uniform_real_distribution<> disN(-N, 2 * N);
static volatile auto pirange    = true;
static volatile auto notPirange = !pirange;

static void BM_BASELINE(benchmark::State &state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(dis(gen));
  }
}

static void BM_FoldRescaleMacro(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = dis(gen);
    benchmark::DoNotOptimize(FOLDRESCALE(x, N, pirange));
  }
}

static void BM_FoldRescaleMacroN(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = disN(gen);
    benchmark::DoNotOptimize(FOLDRESCALE(x, N, notPirange));
  }
}

static void BM_FoldRescale00(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = dis(gen);
    benchmark::DoNotOptimize(foldRescale00(x, N, pirange));
  }
}

static void BM_FoldRescale00N(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = disN(gen);
    benchmark::DoNotOptimize(foldRescale00(x, N, notPirange));
  }
}

static void BM_FoldRescale01(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = dis(gen);
    benchmark::DoNotOptimize(foldRescale01(x, N, pirange));
  }
}

static void BM_FoldRescale01N(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = disN(gen);
    benchmark::DoNotOptimize(foldRescale01(x, N, notPirange));
  }
}

static void BM_FoldRescale02(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = dis(gen);
    benchmark::DoNotOptimize(foldRescale02<true>(x, N));
  }
}

static void BM_FoldRescale02N(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = disN(gen);
    benchmark::DoNotOptimize(foldRescale02<false>(x, N));
  }
}

static void BM_FoldRescale03(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = dis(gen);
    benchmark::DoNotOptimize(foldRescale03<true>(x, N));
  }
}

static void BM_FoldRescale03N(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = disN(gen);
    benchmark::DoNotOptimize(foldRescale03<false>(x, N));
  }
}

static void BM_FoldRescale04(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = dis(gen);
    benchmark::DoNotOptimize(FOLDRESCALE04(x, N, pirange));
  }
}

static void BM_FoldRescale04N(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = disN(gen);
    benchmark::DoNotOptimize(FOLDRESCALE04(x, N, notPirange));
  }
}

static void BM_FoldRescale05(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = dis(gen);
    benchmark::DoNotOptimize(FOLDRESCALE05(x, N, pirange));
  }
}

static void BM_FoldRescale05N(benchmark::State &state) {
  for (auto _ : state) {
    FLT x = disN(gen);
    benchmark::DoNotOptimize(FOLDRESCALE05(x, N, notPirange));
  }
}


BENCHMARK(BM_BASELINE)->Iterations(10000000);
BENCHMARK(BM_FoldRescaleMacro)->Iterations(1000000);
BENCHMARK(BM_FoldRescale00)->Iterations(1000000);
BENCHMARK(BM_FoldRescale01)->Iterations(1000000);
BENCHMARK(BM_FoldRescale02)->Iterations(1000000);
BENCHMARK(BM_FoldRescale03)->Iterations(10000000);
BENCHMARK(BM_FoldRescale04)->Iterations(1000000);
BENCHMARK(BM_FoldRescale05)->Iterations(1000000);
BENCHMARK(BM_FoldRescaleMacroN)->Iterations(1000000);
BENCHMARK(BM_FoldRescale00N)->Iterations(1000000);
BENCHMARK(BM_FoldRescale01N)->Iterations(1000000);
BENCHMARK(BM_FoldRescale02N)->Iterations(1000000);
BENCHMARK(BM_FoldRescale03N)->Iterations(1000000);
BENCHMARK(BM_FoldRescale04N)->Iterations(1000000);
BENCHMARK(BM_FoldRescale05N)->Iterations(1000000);

void testFoldRescaleFunctions() {
  for (bool p : {true}) {
    for (int i = 0; i < 1024; ++i) { // Run the test 1000 times
      FLT x           = dis(gen);
      FLT resultMacro = FOLDRESCALE(x, N, p);
      FLT result00    = foldRescale00(x, N, p);
      FLT result01    = foldRescale01(x, N, p);
      FLT result02    = p ? foldRescale02<true>(x, N) : foldRescale02<false>(x, N);
      FLT result03    = p ? foldRescale03<true>(x, N) : foldRescale03<false>(x, N);
      FLT result04    = FOLDRESCALE04(x, N, p);
      FLT result05    = FOLDRESCALE05(x, N, p);

      // function that compares two floating point number with a tolerance, using
      // relative error
      auto compare = [](FLT a, FLT b) {
        return std::abs(a - b) > std::max(std::abs(a), std::abs(b)) * 10e-13;
      };

      if (compare(resultMacro, result00)) {
        std::cout << "resultMacro: " << resultMacro << " result00: " << result00
                  << std::endl;
        throw std::runtime_error("function00 is wrong");
      }
      if (compare(resultMacro, result01)) {
        std::cout << "resultMacro: " << resultMacro << " result01: " << result01
                  << std::endl;
        throw std::runtime_error("function01 is wrong");
      }
      if (compare(resultMacro, result02)) {
        std::cout << "resultMacro: " << resultMacro << " result02: " << result02
                  << std::endl;
        throw std::runtime_error("function02 is wrong");
      }
      if (compare(resultMacro, result03)) {
        std::cout << "resultMacro: " << resultMacro << " result03: " << result03
                  << std::endl;
        throw std::runtime_error("function03 is wrong");
      }
      if (compare(resultMacro, result04)) {
        std::cout << "resultMacro: " << resultMacro << " result04: " << result04
                  << std::endl;
        throw std::runtime_error("function04 is wrong");
      }
      if (compare(resultMacro, result05)) {
        std::cout << "resultMacro: " << resultMacro << " result05: " << result05
                  << std::endl;
        throw std::runtime_error("function05 is wrong");
      }
    }
  }
}

class BaselineSubtractingReporter : public benchmark::ConsoleReporter {
public:
  bool ReportContext(const Context &context) override {
    return benchmark::ConsoleReporter::ReportContext(context);
  }

  void ReportRuns(const std::vector<Run> &reports) override {
    for (const auto &run : reports) {
      if (run.benchmark_name() == "BM_BASELINE") {
        baseline_time = run.cpu_accumulated_time;
      } else {
        Run modified_run = run;
        modified_run.cpu_accumulated_time -= baseline_time;
        benchmark::ConsoleReporter::ReportRuns({modified_run});
      }
    }
  }

private:
  double baseline_time = 0.0;
};

int main(int argc, char **argv) {
  pirange    = argc & 1;
  notPirange = !pirange;
  static std::random_device rd;
  const auto seed = rd();
  std::cout << "Seed: " << seed << "\n";
  gen.seed(seed);
  testFoldRescaleFunctions();
  ::benchmark::Initialize(&argc, argv);
  BaselineSubtractingReporter reporter;
  ::benchmark::RunSpecifiedBenchmarks(&reporter);
  return 0;
}
