#include "../src/finufft.h"
#include <complex>
#include <stdio.h>
using namespace std;

int main(int argc, char* argv[])
/* calling the FINUFFT library from C++ using dumb inputs that raise errors.
   Barnett 3/13/17, edited from example1d1.

   Compile with:
   g++ -std=c++11 -fopenmp dumbinputs.cpp ../lib/libfinufft.a -o dumbinputs  -lfftw3 -lfftw3_omp -lm
   or if you have built a single-core version:
   g++ -std=c++11 dumbinputs.cpp ../lib/libfinufft.a -o dumbinputs -lfftw3 -lm

   Usage: ./dumbinputs
*/
{
  int M = 10;            // number of nonuniform points. Keep small
  int N = 10;            // number of modes, and also output NU pts in type 3
  double acc = 1e-6;      // desired accuracy
  nufft_opts opts;        // default options struct for the library
  complex<double> I = complex<double>{0.0,1.0};  // the imaginary unit

  int MM = M*M*M, NN = N*N*N;   // since we'll go to 3d
  // generate some random nonuniform points (x) and complex strengths (c):
  double *x = (double *)malloc(sizeof(double)*MM);
  complex<double>* c = (complex<double>*)malloc(sizeof(complex<double>)*MM);
  for (int j=0; j<MM; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);  // uniform in [-pi,pi)
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }
  // allocate output array for the Fourier modes, plus some F data too
  complex<double>* F = (complex<double>*)malloc(sizeof(complex<double>)*NN);
  for (int k=0; k<NN; ++k)
    F[k] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  double *s = (double*)malloc(sizeof(double)*NN);
  for (int k=0; k<NN; ++k) s[k] = 1e10*(2*((double)rand()/RAND_MAX)-1); // huge
  // printf("%.3g %3g\n",twonorm(N,F),twonorm(M,c));

  // 1D dumb cases
  int ier = finufft1d1(M,x,c,+1,0,N,F,opts);
  printf("1d1 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft1d1(M,x,c,+1,acc,0,F,opts);
  printf("1d1 N=0:\tier=%d\tnrm(F)=%.3g (untouched)\n",ier,twonorm(N,F));
  ier = finufft1d1(0,x,c,+1,acc,N,F,opts);
  printf("1d1 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  ier = finufft1d2(M,x,c,+1,0,N,F,opts);
  printf("1d2 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft1d2(M,x,c,+1,acc,0,F,opts);
  printf("1d2 N=0:\tier=%d\nrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft1d2(0,x,c,+1,acc,N,F,opts);
  printf("1d2 M=0:\tier=%d\n",ier);
  ier = finufft1d3(M,x,c,+1,0,N,s,F,opts);
  printf("1d3 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft1d3(M,x,c,+1,acc,0,s,F,opts);
  printf("1d3 nk=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  ier = finufft1d3(0,x,c,+1,0,N,s,F,opts);
  printf("1d3 M=0:\tier=%d\n",ier);
  ier = finufft1d3(M,x,c,+1,acc,N,s,F,opts);
  printf("1d3 XK prod too big:\tier=%d (should complain)\n",ier);

  free(x); free(c); free(F); free(s);
  return 0;
}
