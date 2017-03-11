// this is all you need to include...
#include "../src/finufft.h"
#include <complex>
// needed for this example...
#include <stdio.h>
using namespace std;

int main(int argc, char* argv[])
/* Simple example of calling the FINUFFT library from C++, using plain
   arrays of complex numbers, with a math test. Barnett 3/10/17

   Compile with:
   g++ -std=c++11 -fopenmp example1d1.cpp ../lib/libfinufft.a -o example1d1  -lfftw3 -lfftw3_omp -lm
   or if you have built a single-core version:
   g++ -std=c++11 example1d1.cpp ../lib/libfinufft.a -o example1d1 -lfftw3 -lm

   Usage: ./example1d1
*/
{
  int M = 1e6;            // number of nonuniform points
  int N = 1e6;            // number of modes
  double acc = 1e-9;      // desired accuracy
  nufft_opts opts;        // default options struct for the library
  complex<double> I = complex<double>{0.0,1.0};  // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c):
  double *x = (double *)malloc(sizeof(double)*M);
  complex<double>* c = (complex<double>*)malloc(sizeof(complex<double>)*M);
  for (int j=0; j<M; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi)
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }
  // allocate output array for the Fourier modes:
  complex<double>* F = (complex<double>*)malloc(sizeof(complex<double>)*N);

  // call the NUFFT (with iflag=+1):
  int ier = finufft1d1(M,x,c,+1,acc,N,F,opts);

  int n = 142519;   // check the answer just for this mode...
  complex<double> Ftest = {0,0};
  for (int j=0; j<M; ++j)
    Ftest += c[j] * exp(I*(double)n*x[j]) / (double)M;
  int nout = n+N/2;       // index in output array for freq mode n
  double err = abs((F[nout] - Ftest)/Ftest);
  printf("1D type-1 NUFFT done. Relative error in F[%d] is %.3g\n",n,err);

  free(x); free(c); free(F);
  return ier;
}
