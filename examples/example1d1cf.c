// this is all you must include...
#include "../src/finufft.h"
// also needed for this example...
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>

int main(int argc, char* argv[])
/* Simple example of calling the FINUFFT library from C, using C complex type,
   with a math test.
   Single-precision version (must be linked with single-precision libfinufft.a)
   Barnett 4/5/17. opts ctrl, t1 prefac convention, smaller prob size 9/14/18

   Compile with:
   gcc -fopenmp example1d1cf.c ../lib-static/libfinufft.a -o example1d1cf -lfftw3f -lfftw3f_omp -lm -lstdc++
   or if you have built a single-core version:
   gcc example1d1cf.c ../lib-static/libfinufft.a -o example1d1cf -lfftw3f -lm -lstdc++

   Usage: ./example1d1cf
*/
{
  int M = 1e5;            // number of nonuniform points
  int N = 1e5;            // number of modes (NB if too large lose acc in 1d)
  float acc = 1e-3;       // desired accuracy
  int j,ier,n,m,nout;
  float *x,err,aF,Fmax;
  float complex *c,*F,Ftest;

  // generate some random nonuniform points (x) and complex strengths (c):
  x = (float *)malloc(sizeof(float)*M);
  c = (float complex*)malloc(sizeof(float complex)*M);
  for (j=0; j<M; ++j) {
    x[j] = M_PI*(2*((float)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi)
    c[j] = 2*((float)rand()/RAND_MAX)-1 + I*(2*((float)rand()/RAND_MAX)-1);
  }
  // allocate complex output array for the Fourier modes
  F = (float complex*)malloc(sizeof(float complex)*N);

  nufft_opts opts;
  finufft_default_opts(&opts);            // set default opts (must do this)
  opts.debug = 2;                         // show how to override a default
  //opts.upsampfac =1.25;                 // other opts...
  
  // call the NUFFT (with iflag=+1); this is the same code as from C++:
  ier = finufft1d1(M,x,c,+1,acc,N,F,opts);

  n = 14251;   // check the answer just for this mode...
  Ftest = 0.0;
  for (j=0; j<M; ++j)
    Ftest += c[j] * cexpf(I*(float)n*x[j]);
  nout = n+N/2;       // index in output array for freq mode n
  Fmax = 0.0;       // compute inf norm of F
  for (m=0; m<N; ++m) {
    aF = cabsf(F[m]);
    if (aF>Fmax) Fmax=aF;
  }
  err = cabsf(F[nout] - Ftest)/Fmax;
  printf("1D type-1 NUFFT done. ier=%d, err in F[%d] rel to max(F) is %.3g\n",ier,n,err);

  free(x); free(c); free(F);
  return ier;
}
