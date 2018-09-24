// this is all you must include...
#include "finufft.h"
#include <complex>

// also needed for this example...
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int main(int argc, char* argv[])
/* Simple example of calling the FINUFFT library from C++, using plain
   arrays of C++ complex numbers, with a math test.
   Single-precision version (must be linked with single-precision libfinufftf.a)
   See example1d1 for double-precision.
   Barnett 4/3/17. Fixed t1 prefac convention; smaller prob size 9/14/18.

   Compile with:
   g++ -fopenmp example1d1f.cpp -I ../src ../lib-static/libfinufftf.a -o example1d1f -lfftw3f -lfftw3f_omp -lm -DSINGLE
   or if you have built a single-core version:
   g++ example1d1f.cpp -I ../src ../lib-static/libfinufftf.a -o example1d1f -lfftw3f -lm -DSINGLE

   Usage: ./example1d1f
*/
{
  int M = 1e5;            // number of nonuniform points
  int N = 1e5;            // number of modes (NB if too large lose acc in 1d)
  float acc = 1e-3;       // desired accuracy
  nufft_opts opts; finufft_default_opts(&opts);
  complex<float> I = complex<float>(0.0,1.0);  // the imaginary unit
  
  // generate some random nonuniform points (x) and complex strengths (c):
  float *x = (float *)malloc(sizeof(float)*M);
  complex<float>* c = (complex<float>*)malloc(sizeof(complex<float>)*M);
  for (int j=0; j<M; ++j) {
    x[j] = M_PI*(2*((float)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi)
    c[j] = 2*((float)rand()/RAND_MAX)-1 + I*(2*((float)rand()/RAND_MAX)-1);
  }
  // allocate output array for the Fourier modes:
  complex<float>* F = (complex<float>*)malloc(sizeof(complex<float>)*N);

  // call the NUFFT (with iflag=+1): N,M will be typecast to BIGINT
  int ier = finufft1d1(M,x,c,+1,acc,N,F,opts);

  int n = 14251;   // check the answer just for this mode...
  complex<float> Ftest = complex<float>(0,0);
  for (int j=0; j<M; ++j)
    Ftest += c[j] * exp(I*(float)n*x[j]);
  int nout = n+N/2;       // index in output array for freq mode n
  float Fmax = 0.0;       // compute inf norm of F
  for (int m=0; m<N; ++m) {
    float aF = abs(F[m]);
    if (aF>Fmax) Fmax=aF;
  }
  float err = abs(F[nout] - Ftest)/Fmax;
  printf("1D type-1 NUFFT done. ier=%d, err in F[%d] rel to max(F) is %.3g\n",ier,n,err);
    
  free(x); free(c); free(F);
  return ier;
}
