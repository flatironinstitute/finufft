/* exponential sqrt kernel eval speed tester, single-thread,
   extracted from FINUFFT.

   compile with:

g++-7 eval_ker_expts_ludvig.cpp -o eval_ker_expts_ludvig -Ofast -funroll-loops -march=native; time ./eval_ker_expts_ludvig

Ludvig's tweak of eval_ker_expts, 3/29/18
*/

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#ifdef VCL
// Use Agner Fog's vector class library
// http://www.agner.org/optimize/#vectorclass (extract zip in this directory)
#include "vectorclass.h"
#include "vectormath_exp.h"
#endif

// Choose prec...
typedef double FLT;
//typedef float FLT;

static inline void evaluate_kernel_vector(FLT * __restrict__ ker, const FLT * __restrict__ args, const FLT beta, const FLT c, const int N)
/* Evaluate kernel for a vector of N arguments.
   Can comment out either or both loops.
   The #pragra's need to be removed for icpc if -fopenmp not used.
*/
{
#ifdef VCL 
  for (int i = 0; i < N; i+=4) // Assume w divisible by 4
  {
    Vec4d vec;
    vec.load(args + i);
    vec = exp(beta*sqrt(1.0 - c*vec*vec));
    vec.store(ker + i);
  }  
#else
  for (int i = 0; i < N; i++) // Straight computation
    ker[i] = exp(beta * sqrt(1.0 - c*args[i]*args[i]));
#endif
  
}

int main(int argc, char* argv[])
{
  int M = (int) 1e7;                // # of reps
  int w=12;                         // 12, spread width (small), needn't be mult of 4, 15 takes 3.2s but 12 only 0.2s, in g++-7

  if (1) {   // 0 makes 10x slower (2s) than 1, which is 0.2 s, for g++-7 - ahb
  if (argc == 3)
  {
    sscanf(argv[1],"%d",&M);
    //sscanf(argv[2],"%d",&w);
  }
  }
  
  
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
    // we don't do anything with f, but compiler hasn't figured this out :)
  }
  printf("ans=%.15g\n",ans);
  return 0;
}
