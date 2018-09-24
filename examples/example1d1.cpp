// this is all you must include for the finufft lib...
#include "finufft.h"
#include <complex>

// also needed for this example...
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int main(int argc, char* argv[])
/* Simple example of calling the FINUFFT library from C++, using plain
   arrays of C++ complex numbers, with a math test. Barnett 3/10/17
   Double-precision version (see example1d1f for single-precision)

   Compile with:
   g++ -fopenmp example1d1.cpp -I ../src ../lib-static/libfinufft.a -o example1d1  -lfftw3 -lfftw3_omp -lm
   or if you have built a single-core version:
   g++ example1d1.cpp -I ../src ../lib-static/libfinufft.a -o example1d1 -lfftw3 -lm

   Usage: ./example1d1
*/
{
  int M = 1e6;            // number of nonuniform points
  int N = 1e6;            // number of modes
  double acc = 1e-9;      // desired accuracy
  nufft_opts opts; finufft_default_opts(&opts);
  complex<double> I = complex<double>(0.0,1.0);  // the imaginary unit
  
  // generate some random nonuniform points (x) and complex strengths (c):
  double *x = (double *)malloc(sizeof(double)*M);
  complex<double>* c = (complex<double>*)malloc(sizeof(complex<double>)*M);
  for (int j=0; j<M; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi)
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }
  // allocate output array for the Fourier modes:
  complex<double>* F = (complex<double>*)malloc(sizeof(complex<double>)*N);

  // call the NUFFT (with iflag=+1): note N and M are typecast to BIGINT
  int ier = finufft1d1(M,x,c,+1,acc,N,F,opts);

  int n = 142519;   // check the answer just for this mode...
  complex<double> Ftest = complex<double>(0,0);
  for (int j=0; j<M; ++j)
    Ftest += c[j] * exp(I*(double)n*x[j]);
  int nout = n+N/2;        // index in output array for freq mode n
  double Fmax = 0.0;       // compute inf norm of F
  for (int m=0; m<N; ++m) {
    double aF = abs(F[m]);
    if (aF>Fmax) Fmax=aF;
  }
  double err = abs(F[nout] - Ftest)/Fmax;
  printf("1D type-1 NUFFT done. ier=%d, err in F[%d] rel to max(F) is %.3g\n",ier,n,err);

  free(x); free(c); free(F);
  return ier;
}
