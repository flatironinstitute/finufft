/* exponential sqrt kernel eval speed tester, single-thread, trying openmp simd.
   compile with:

g++ eval_ker_expts2.cpp -o eval_ker_expts2 -Ofast -march=native -fopenmp; time ./eval_ker_expts2 10000000

Barnett 4/23/18
*/

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Choose prec...
typedef double FLT;
//typedef float FLT;

static inline void evaluate_kernel_vector(FLT *ker, const FLT *args, const FLT beta, const FLT c, const int N)
/* Evaluate kernel for a vector of N arguments.
   The #pragra's need to be removed for icpc if -fopenmp not used.
 */
{
#pragma omp simd
  for (int i = 0; i < N; i++)
    ker[i] = exp(beta * sqrt(1.0 - c*args[i]*args[i]));
}

int main(int argc, char* argv[])
{
  int M = (int) 1e7;          // # of reps
  if (argc>1)
    sscanf(argv[1],"%d",&M);    // weirdly makes 10x faster, even on gcc 5.4.0, with omp simd
  int w=16;                   // spread width (small), needn't be mult of 4
  FLT beta=2.3*w, c = 4.0/(w*w); // ker params
  FLT iw = 1.0/(FLT)w;
  FLT ans = 0.0;                 // dummy answer
  std::vector<FLT> x(w);
  std::vector<FLT> f(w);
  for (int i=1;i<M;++i) {
    FLT xi = i/(FLT)M;        // dummy offset to make each rep different
    for (int j=0;j<w;++j)           // fill a simple argument vector (cheap)
      x[j] = -1.0 + xi + iw*j;      // note each x in [-1,1]
    evaluate_kernel_vector(&f[0],&x[0],beta,c,w);   // eval kernel into f
    for (int j=0;j<w;++j)
      ans += f[j];                  // do something cheap to use f output
  }
  printf("ans=%.15g\n",ans);
  return 0;
}

// even i7, gcc 5.4.0, find if sscanf for M, goes from 2.0s to 0.16s  !!!!
