#include "../src/finufft.h"
#include <stdio.h>
#include <stdlib.h>
#include <complex>

// Basic pass-fail test of library. exit code 0 success, failure otherwise.
// This is useful for brew recipe.
// Doesn't use any macros other than finufft.h
// Works for single/double or multi-/single-thread.
// Simplified from Amit Moscovitz and example1d1. Barnett 11/1/18.

int main()
{
  BIGINT M = 1e3, N = 1e3;   // defaults: M = # srcs, N = # modes out
  double tol = 1e-5;         // req tol, covers both single & double prec cases
  nufft_opts opts; finufft_default_opts(&opts);     // set default opts
  int isign = +1;            // exponential sign for NUFFT
  static const CPX I = CPX(0.0,1.0);      // imaginary unit. Note: avoid (CPX)
  CPX* F = (CPX*)malloc(sizeof(CPX)*N);   // alloc output mode coeffs

  // Make the input data....................................
  FLT* x = (FLT*)malloc(sizeof(FLT)*M);   // NU pts locs
  CPX* c = (CPX*)malloc(sizeof(CPX)*M);   // strengths 
  for (BIGINT j=0; j<M; ++j) {
    x[j] = M_PI*(2*((FLT)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi)
    c[j] = 2*((FLT)rand()/RAND_MAX)-1 + I*(2*((FLT)rand()/RAND_MAX)-1);
  }
  // Run it.................................................
  int ier = finufft1d1(M,x,c,isign,tol,N,F,opts);
  if (ier!=0) {
    printf("basicpassfail: finufft1d1 error (ier=%d)!",ier);
    exit(ier);
  }
  // Check correct math for a single mode...................
  BIGINT n = (BIGINT)(0.37*N);   // choose some mode near the top (N/2)
  CPX Ftest = CPX(0.0,0.0);      // crude exact answer & error check...
  for (BIGINT j=0; j<M; ++j)
    Ftest += c[j] * exp((FLT)isign*I*(FLT)n*x[j]);
  BIGINT nout = n+N/2;           // index in output array for freq mode n
  FLT Finfnrm = 0.0;             // compute inf norm of F...
  for (int m=0; m<N; ++m) {
    FLT aF = abs(F[m]);          // note C++ abs complex type, not C fabs(f)
    if (aF>Finfnrm) Finfnrm=aF;
  }
  FLT relerr = abs(F[nout] - Ftest)/Finfnrm;
  //printf("requested tol %.3g: rel err for one mode %.3g\n",tol,relerr);
  free(x); free(c); free(F);
  return (std::isnan(relerr) || relerr > 10.0*tol);  // ne.0 -> make test error
}
