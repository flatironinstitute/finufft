// public header
#include "finufft.h"

// private access to timer
#include "finufft/utils_precindep.h"
using namespace finufft::utils;

#include <complex>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int main(int argc, char *argv[])
/* What is small-problem cost of FINUFFT library from C++, using plain
   arrays of C++ complex numbers?  Barnett 10/31/17.
   for Xi Chen question. Updated to also demo guru interface and compare speed.
   6/7/22 made deterministic changes so check answer matches both ways.

   g++ -fopenmp manysmallprobs.cpp ../lib-static/libfinufft.a -o manysmallprobs  -lfftw3
   -lfftw3_omp -lm # multithreaded is much slower, due to overhead of starting threads?...
   export OMP_NUM_THREADS=1
   time ./manysmallprobs

   simple interface: about 1.2s on single core. Ie, throughput 3.3e6 NU pts/sec.
   guru interface: about 0.24s on single core. Ie, throughput 1.7e7 NU pts/sec.

   But why is multi-thread so much slower? (thread start-up time?)
*/
{
  int M      = 2e2;                              // number of nonuniform points
  int N      = 2e2;                              // number of modes
  int reps   = 2e4;                              // how many repetitions
  double acc = 1e-6;                             // desired accuracy

  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit
  int ier;

  // generate some random nonuniform points (x) and complex strengths (c):
  double *x          = (double *)malloc(sizeof(double) * M);
  complex<double> *c = (complex<double> *)malloc(sizeof(complex<double>) * M);
  for (int j = 0; j < M; ++j) {
    x[j] = M_PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi]
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }
  // allocate output array for the Fourier modes:
  complex<double> *F = (complex<double> *)malloc(sizeof(complex<double>) * N);

  printf("repeatedly calling the simple interface: --------------------- \n");
  finufft::utils::CNTime timer;
  timer.start();
  for (int r = 0; r < reps; ++r) { // call the NUFFT (with iflag=+1):
    // printf("rep %d\n",r);
    x[0] = M_PI * (-1.0 + 2 * (double)r / (double)reps); // one source jiggles around
    c[0] = (1.0 + I) * (double)r / (double)reps;         // one coeff also jiggles
    ier  = finufft1d1(M, x, c, +1, acc, N, F, NULL);
  }
  // (note this can't use the many-vectors interface since the NU change)
  complex<double> y = F[0]; // actually use the data so not optimized away
  printf(
      "%d reps of 1d1 done in %.3g s,\t%.3g NU pts/s\t(last ier=%d)\nF[0]=%.6g + %.6gi\n",
      reps, timer.elapsedsec(), reps * M / timer.elapsedsec(), ier, real(y), imag(y));

  printf("repeatedly executing via the guru interface: -------------------\n");
  timer.restart();
  finufft_plan plan;
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.debug   = 0;
  int64_t Ns[] = {N, 1, 1};
  int ntransf  = 1;                // since we do one at a time (neq reps)
  finufft_makeplan(1, 1, Ns, +1, ntransf, acc, &plan, &opts);
  for (int r = 0; r < reps; ++r) { // set the pts and execute
    x[0] = M_PI * (-1.0 + 2 * (double)r / (double)reps); // one source jiggles around
    // (of course if most sources *were* in fact fixed, use ZGEMM for them!)
    finufft_setpts(plan, M, x, NULL, NULL, 0, NULL, NULL, NULL);
    c[0] = (1.0 + I) * (double)r / (double)reps; // one coeff also jiggles
    ier  = finufft_execute(plan, c, F);
  }
  finufft_destroy(plan);
  y = F[0];
  printf(
      "%d reps of 1d1 done in %.3g s,\t%.3g NU pts/s\t(last ier=%d)\nF[0]=%.6g + %.6gi\n",
      reps, timer.elapsedsec(), reps * M / timer.elapsedsec(), ier, real(y), imag(y));
  free(x);
  free(c);
  free(F);
  return ier;
}
