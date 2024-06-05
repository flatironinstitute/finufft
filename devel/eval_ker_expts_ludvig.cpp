/* exponential sqrt kernel eval speed tester, single-thread,
   extracted from FINUFFT.

   compile with:

g++-7 eval_ker_expts_ludvig.cpp -o eval_ker_expts_ludvig -Ofast -funroll-loops
-march=native -fopt-info; time ./eval_ker_expts_ludvig

Update: (8/8/19)
g++-8 is less brittle - it is able to get 0.2 s runtime for i=1 or 0 start.

Ludvig's tweak of eval_ker_expts, 3/29/18.  Can get <0.2s for M=1e7, w=12.
Note that the range of arguments is wrong [-1,1] not [-w/2,w/2].
This might explain the v. fast 0.2 s timing possible.

A result: 2.0s even if opt-info shows 13-length loops unrolled.
It's
eval_ker_expts_ludvig.cpp:69:17: note: loop vectorized
that correlates w/ 0.2s magic.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef VCL
// Use Agner Fog's vector class library
// http://www.agner.org/optimize/#vectorclass (extract zip in this directory)
#include "vectorclass.h"
#include "vectormath_exp.h"
#endif

// Choose prec...
typedef double FLT;
// typedef float FLT;

static inline void evaluate_kernel_vector(FLT *__restrict__ ker,
                                          const FLT *__restrict__ args, const FLT beta,
                                          const FLT c, const int N)
/* Evaluate kernel for a vector of N arguments.
 */
{
#ifdef VCL
  for (int i = 0; i < N; i += 4) // Assume w divisible by 4
  {
    Vec4d vec;
    vec.load(args + i);
    vec = exp(beta * sqrt(1.0 - c * vec * vec));
    vec.store(ker + i);
  }
#else
  for (int i = 0; i < N; i++) // Straight computation, note no pragma omp simd
    ker[i] = exp(beta * sqrt(1.0 - c * args[i] * args[i]));
#endif
}

int main(int argc, char *argv[]) {
  int M = (int)1e7; // # of reps
  int w = 12; // 12, spread width (small), needn't be mult of 4, 15 takes 3.2s but 12
              // only 0.2s, in g++-7. But not in gcc 5.4.0

  if (1) {    // 0 makes 10x slower (2s) than 1, which is 0.2 s, for g++-7 - ahb
    if (argc == 3) {
      sscanf(argv[1], "%d", &M);
      // sscanf(argv[2],"%d",&w);  // slows down from 0.2s to 0.44s if use - why??
    }
  }

  FLT beta = 2.3 * w, c = 4.0 / (w * w); // ker params
  FLT iw  = 1.0 / (FLT)w;
  FLT ans = 0.0;                         // dummy answer
  std::vector<FLT> x(w);
  std::vector<FLT> f(w);

  for (int i = 1; i <= M; ++i) { // changing from i=1 to i=0 slows from 0.2s to 2.4s!!!!
                                 // I don't understand - has to be a better way to
                                 // control (assembly code?)
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
