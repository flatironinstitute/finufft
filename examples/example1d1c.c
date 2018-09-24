// this is all you must include to access finufft from C...
#include "../src/finufft.h"

// also needed for this example...
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>

int main(int argc, char* argv[])
/* Simple example of calling the FINUFFT library from C, using C complex type,
   with a math test. Double-precision. Barnett 3/10/17. Opts control 6/19/18.

   Compile with:
   gcc -fopenmp example1d1c.c ../lib-static/libfinufft.a -o example1d1c -lfftw3 -lfftw3_omp -lm -lstdc++
   or if you have built a single-core version:
   gcc example1d1c.c ../lib-static/libfinufft.a -o example1d1c -lfftw3 -lm -lstdc++

   Usage: ./example1d1c
*/
{
  int M = 1e6;            // number of nonuniform points
  int N = 1e6;            // number of modes
  double acc = 1e-9;      // desired accuracy
  int j,ier,n,m,nout;
  double *x,err,aF,Fmax;
  double complex *c,*F,Ftest;

  // generate some random nonuniform points (x) and complex strengths (c):
  x = (double *)malloc(sizeof(double)*M);
  c = (double complex*)malloc(sizeof(double complex)*M);
  for (j=0; j<M; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi)
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }
  // allocate complex output array for the Fourier modes
  F = (double complex*)malloc(sizeof(double complex)*N);

  nufft_opts opts;
  finufft_default_opts(&opts);            // set default opts (must do this)
  opts.debug = 2;                         // show how to override a default
  //opts.upsampfac =1.25;                 // other opts...
  
  // call the NUFFT (with iflag=+1); this is the same code as from C++:
  ier = finufft1d1(M,x,c,+1,acc,N,F,opts);

  n = 142519;         // check the answer just for this mode...
  Ftest = 0.0;
  for (j=0; j<M; ++j)
    Ftest += c[j] * cexp(I*(double)n*x[j]);
  nout = n+N/2;       // index in output array for freq mode n
  Fmax = 0.0;         // compute inf norm of F
  for (m=0; m<N; ++m) {
    aF = cabs(F[m]);
    if (aF>Fmax) Fmax=aF;
  }
  err = cabs(F[nout] - Ftest)/Fmax;
  printf("1D type-1 NUFFT done. ier=%d, err in F[%d] rel to max(F) is %.3g\n",ier,n,err);

  free(x); free(c); free(F);
  return ier;
}
