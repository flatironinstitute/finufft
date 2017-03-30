// this is all you must include...
#include "../src/finufft_c.h"
// also needed for this example...
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>

int main(int argc, char* argv[])
/* Simple example of calling the FINUFFT library from C, using C complex type,
   with a math test. Note the C++ compiler is still needed. Barnett 3/10/17

   Compile with:
   g++ -fopenmp example1d1c.c ../lib/libfinufft.a -o example1d1c -lfftw3 -lfftw3_omp -lm
   or if you have built a single-core version:
   g++ example1d1c.c ../lib/libfinufft.a -o example1d1c -lfftw3 -lm

   Usage: ./example1d1c
*/
{
  int M = 1e6;            // number of nonuniform points
  int N = 1e6;            // number of modes
  double acc = 1e-9;      // desired accuracy
  int j,ier,n,nout;
  double *x,err;
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

  // call the NUFFT (with iflag=+1):
  ier = finufft1d1_c(M,x,c,+1,acc,N,F);

  n = 142519;   // check the answer just for this mode...
  Ftest = 0.0;
  for (j=0; j<M; ++j)
    Ftest += c[j] * cexp(I*(double)n*x[j]) / (double)M;
  nout = n+N/2;       // index in output array for freq mode n
  err = cabs((F[nout] - Ftest)/Ftest);
  printf("1D type-1 NUFFT done. ier=0, relative error in F[%d] is %.3g\n",ier,n,err);

  free(x); free(c); free(F);
  return ier;
}
