#include "finufft.h"
#include <complex>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int main(int argc, char* argv[])
/* What is small-problem cost of FINUFFT library from C++, using plain
   arrays of C++ complex numbers?  Barnett 10/31/17.
   for Xi Chen question.

   g++ -fopenmp manysmallprobs.cpp ../lib-static/libfinufft.a -o manysmallprobs  -lfftw3 -lfftw3_omp -lm
   # multithreaded is much slower, due to overhead of starting threads...
   export OMP_NUM_THREADS=1
   time ./manysmallprobs

   Takes about 1.2s on single core. Ie, throughput is 3.3e6 pts/sec.
*/
{  
  int M = 2e2;            // number of nonuniform points
  int N = 2e2;            // number of modes
  int reps = 2e4;         // how many repetitions
  double acc = 1e-6;      // desired accuracy
  
  nufft_opts opts; finufft_default_opts(&opts);
  complex<double> I = complex<double>(0.0,1.0);  // the imaginary unit
  int ier;
  
  // generate some random nonuniform points (x) and complex strengths (c):
  double *x = (double *)malloc(sizeof(double)*M);
  complex<double>* c = (complex<double>*)malloc(sizeof(complex<double>)*M);
  for (int j=0; j<M; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi]
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }
  // allocate output array for the Fourier modes:
  complex<double>* F = (complex<double>*)malloc(sizeof(complex<double>)*N);

  
  for (int r=0;r<reps;++r) {    // call the NUFFT (with iflag=+1):
    //printf("rep %d\n",r);
    x[0] = M_PI*(2*((double)rand()/RAND_MAX)-1);  // one source jiggles around
    ier = finufft1d1(M,x,c,+1,acc,N,F,opts);
  }
 
  complex<double> y=F[0];
  printf("%d reps of 1d1 done (last ier=%d): F[0]=%.6g + %.6gi\n",reps,ier,real(y),imag(y));
  
  free(x); free(c); free(F);
  return ier;
}
