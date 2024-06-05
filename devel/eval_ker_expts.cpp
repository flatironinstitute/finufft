/* exponential sqrt kernel eval speed tester, single-thread,
   extracted from FINUFFT.

   compile with:

g++ eval_ker_expts.cpp -o eval_ker_expts -Ofast -funroll-loops -march=native; time
./eval_ker_expts

   Barnett 3/28/18 for JD Patel (Intel).
   Single-prec version also of interest, if faster.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Choose prec...
typedef double FLT;
// typedef float FLT;

static inline void evaluate_kernel_vector(FLT *__restrict__ ker,
                                          const FLT *__restrict__ args, const FLT beta,
                                          const FLT c, const int N)
/* Evaluate kernel for a vector of N arguments.
   Can comment out either or both loops.
   The #pragra's need to be removed for icpc if -fopenmp not used.
 */
{
#pragma omp simd
  for (int i = 0; i < N; i++) // Loop 1: Compute exponential arguments
    ker[i] = beta * sqrt(1.0 - c * args[i] * args[i]);
    // ker[i] = beta * (1.0 - c*args[i]*args[i]);   // no-sqrt version

#pragma omp simd
  for (int i = 0; i < N; i++) // Loop 2: Compute exponentials
    ker[i] = exp(ker[i]);
}

int main(int argc, char *argv[]) {
  int M    = (int)1e7;                   // # of reps
  int w    = 10;                         // spread width (small), needn't be mult of 4
  FLT beta = 2.3 * w, c = 4.0 / (w * w); // ker params
  FLT iw  = 1.0 / (FLT)w;
  FLT ans = 0.0;                         // dummy answer
  std::vector<FLT> x(w);
  std::vector<FLT> f(w);
  for (int i = 1; i < M; ++i) {
    FLT xi = i / (FLT)M;         // dummy offset to make each rep different
    for (int j = 0; j < w; ++j)  // fill a simple argument vector (cheap)
      x[j] = -1.0 + xi + iw * j; // note each x in [-1,1]
    evaluate_kernel_vector(&f[0], &x[0], beta, c, w); // eval kernel into f
    for (int j = 0; j < w; ++j) ans += f[j]; // do something cheap to use f output
    // we don't do anything with f, but compiler hasn't figured this out :)
  }
  printf("ans=%.15g\n", ans);
  return 0;
}

// RESULTS for one core of xeon E5-2643 v3 @ 3.40GHz, which has up to avx2.
// (1 s would be 0.1 G ker evals/sec)

// GCC 4.8.5:
// 2.8 s    no flags
// 1.8 s    -Ofast -funroll-loops -march=native

// GCC 7.3.0
// 2.7 s    no flags
// 2.0 s    -O2
// 1.6 s    -Ofast
// 1.6 s    -Ofast -funroll-loops -march=native -ftree-vectorize makes no diff
// 1.8 s    -Ofast  etc -fopenmp

// ICC 17.0.4:
// 0.75 s   no flags
// 0.9 s    -Ofast -funroll-loops -xHost  ... is slower
// 0.75 s   -fopenmp
// 0.63 s   -fopenmp -xHost
// __restrict__ makes no diff
// 1.2 s    -fopenmp -xHost -lsvml         .. twice as slow!
// With the exp removed, we get:
// 0.52 s   -fopenmp -xHost        ... this suggests exp is not the bottleneck,
//                                     which we know it should be!
// With the exp and sqrt removed:
// 0.4 s    -fopenmp -xHost    -- with is weird since fmas should be fast.
// But, only 0.2 s for GCC 7.3.

// Single-prec ICC 17.0.4:
// 0.73 s   -fopenmp -xHost ... is slower than double!
