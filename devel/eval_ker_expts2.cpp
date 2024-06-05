/* exponential sqrt kernel eval speed tester, single-thread, trying openmp simd.
   compile with:

g++-7 eval_ker_expts2.cpp -o eval_ker_expts2 -Ofast -march=native -fopt-info; time
./eval_ker_expts2 10000000

Barnett 4/23/18. See below for concls.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Choose prec...
typedef double FLT;
// typedef float FLT;

static inline void evaluate_kernel_vector(FLT *ker, const FLT *args, const FLT beta,
                                          const FLT c, const int N)
/* Evaluate kernel for a vector of N arguments.
   The #pragmas need to be removed for icpc if -fopenmp not used.
For g++-7, this pragma (with -fopenmp) slows it down from 0.2s to 0.4s!
THe __restrict__ on the I/O args don't matter.
 */
{
  // #pragma omp simd
  for (int i = 0; i < N; i++) ker[i] = exp(beta * sqrt(FLT(1.0) - c * args[i] * args[i]));
  // FLT(1.0) suggested by mreineck

  // slows down from 0.2s to 2.0s for w=12, unless it's at 0.4s when no effect...
  //  for (int i = 0; i < N; i++)
  //   if (fabs(args[i]) >= (FLT)N/2)    // note fabs not abs!
  // ker[i] = 0.0;
}

int main(int argc, char *argv[]) {
  int M = (int)1e7;                        // # of reps
  if (argc > 1) sscanf(argv[1], "%d", &M); // find not needed to get the 0.2 s time.
  int w = 11; // spread width: 10 0.17s, 11 1.8s, 12 0.2s, 13 2.0s, 15 2.5s
  // if (argc>2)                 // even including this code slows to 0.4s !!
  // sscanf(argv[2],"%d",&w);       //  .. but speeds up w=13 from 2s to 0.4s !
  FLT beta = 2.3 * w, c = 4.0 / (w * w); // typ ker params
  FLT iw  = 1.0 / (FLT)w;
  FLT ans = 0.0;                         // dummy answer
  std::vector<FLT> x(w);
  std::vector<FLT> f(w);
  for (int i = 1; i <= M; ++i) {         // i=0 to M-1 : 2.1s;  i=1 to M : 0.2s !!!!!
    FLT xi = -w / (FLT)2.0 + i / (FLT)M; // dummy offset to make each rep different
    for (int j = 0; j < w; ++j)          // fill a simple argument vector (cheap)
      x[j] = xi + (FLT)j;                // note each x in [-w/2,w/2]
    evaluate_kernel_vector(&f[0], &x[0], beta, c, w); // eval kernel into f
    for (int j = 0; j < w; ++j) ans += f[j]; // do something cheap to use f output
  }
  printf("ans=%.15g\n", ans);
  return 0;
}

// even i7, gcc 5.4.0, find if sscanf for M, goes from 2.0s to 0.16s  !!!!
// No: now only can get for g++-7 (7.2.0), and it's 0.2s

// Since the answer varies with the speed, believe it's doing some special
// loop reordering only.

// once get to fast 0.2s, v. dep on w (eg w=11,15 10x slower)

// -fopenmp w/ pragma omd simd slows down to 0.4s

// __restrict__ has no effect

// study the optim reports more...
