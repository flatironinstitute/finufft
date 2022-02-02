// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <vector>
#include <complex>
#include <cstdio>
#include <stdlib.h>
#include <cassert>
#include <omp.h>
using namespace std;

int main(int argc, char* argv[])
/* Demo single-threaded FINUFFT calls from inside a OMP parallel block.
   Adapted from simple1d1.cpp: C++, STL double complex vectors, with math test.
   Barnett 4/19/21, eg for Goran Zauhar, issue #183.

   Notes: libfinufft *must* have been built with -DFFTW_PLAN_SAFE, which needs
   FFTW >= 3.3.6. You also may not have libfftw3_omp, so I have switched to
   libfftw3_threads in this suggested compile command:

   g++ -fopenmp threadsafe1d1.cpp -I../include ../lib/libfinufft.so -o threadsafe1d1 -lfftw3 -lfftw3_threads -lm

   Usage: ./threadsafe1d1

   Expected output: multiple text lines (however many default threads), each
   reporting small error.
*/
{
  int M = 1e5;            // number of nonuniform points
  int N = 1e5;            // number of modes
  double acc = 1e-9;      // desired accuracy
  nufft_opts* opts = new nufft_opts;     // opts is pointer to struct
  finufft_default_opts(opts);
  complex<double> I = complex<double>(0.0,1.0);  // the imaginary unit
  
  // generate some random nonuniform points (x) and complex strengths (c)...
  vector<double> x(M);
  for (int j=0; j<M; ++j)
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi)

  opts->nthreads=1;       // this is *crucial* so that each call single-thread

  // Now have each thread do independent 1D type 1 on their own data:
#pragma omp parallel
  {
  // generate some complex strengths (c)... local to the thread
  vector<complex<double> > c(M);
  for (int j=0; j<M; ++j)
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);

  // allocate output array for the Fourier modes... local to the thread
  vector<complex<double> > F(N);

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1(M,&x[0],&c[0],+1,acc,N,&F[0],opts);

  int k = 42519;       // check the answer just for this mode frequency...
  assert(k>=-(double)N/2 && k<(double)N/2);
  complex<double> Ftest = complex<double>(0,0);
  for (int j=0; j<M; ++j)
    Ftest += c[j] * exp(I*(double)k*x[j]);
  double Fmax = 0.0;       // compute inf norm of F
  for (int m=0; m<N; ++m) {
    double aF = abs(F[m]);
    if (aF>Fmax) Fmax=aF;
  }
  int kout = k+N/2;        // index in output array for freq mode k
  double err = abs(F[kout] - Ftest)/Fmax;
  
  printf("[thread %2d] 1D t-1 dbl-prec NUFFT done. ier=%d, rel err in F[%d]: %.3g\n",omp_get_thread_num(),ier,k,err);
  }
  return 0;
}
