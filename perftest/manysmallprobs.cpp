#include <complex>
#include <vector>

// public header
#include "finufft.h"

// private access to timer, etc
#include "finufft/test_defs.h"
using namespace finufft::utils;

int main(int argc, char *argv[])
/* What is small-problem cost of FINUFFT library from C++, using plain
   arrays of C++ complex numbers?  Barnett 10/31/17.
   for Xi Chen question. Updated to also demo guru interface and compare speed.
   6/7/22 made deterministic changes so check answer matches both ways.

   g++ -fopenmp manysmallprobs.cpp -I../include ../lib-static/libfinufft.a \
   -o manysmallprobs -lfftw3 -lfftw3_omp -lfftw3f -lfftw3f_omp -lm

   export OMP_NUM_THREADS=1
   # multithreaded is ridiculously slow, due to overhead of starting threads?
   time ./manysmallprobs

   Old (2017) timings:
   simple interface: about 1.2s on single core. Ie, throughput 3.3e6 NU pts/sec.
   guru interface: about 0.24s on single core. Ie, throughput 1.7e7 NU pts/sec.
   Note that ZGEMM on stacked vectors is 10x faster than FINUFFT for this size!
*/
{
  int M      = 2e2;                              // number of nonuniform points
  int N      = 2e2;                              // number of modes
  int reps   = 2e4;                              // how many repetitions
  double acc = 1e-6;                             // desired accuracy

  std::complex<double> I = std::complex<double>(0.0, 1.0); // the imaginary unit
  int ier;

  // generate some random nonuniform points (x) and complex strengths (c):
  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * (static_cast<double>(std::rand()) / RAND_MAX) - 1); // uniform random in [-pi,pi]
    c[j] = 2 * (static_cast<double>(std::rand()) / RAND_MAX) - 1 +
           I * (2 * (static_cast<double>(std::rand()) / RAND_MAX) - 1);
  }
  // allocate output array for the Fourier modes:
  std::vector<std::complex<double>> F(N);

  std::printf("repeatedly calling the simple interface: --------------------- \n");
  CNTime timer;
  timer.start();
  for (int r = 0; r < reps; ++r) { // call the NUFFT (with iflag=+1):
    // printf("rep %d\n",r);
    x[0] = PI * (-1.0 + 2 * static_cast<double>(r) / static_cast<double>(reps)); // one source jiggles around
    c[0] = (1.0 + I) * static_cast<double>(r) / static_cast<double>(reps);       // one coeff also jiggles
    ier = finufft1d1(M, x.data(), c.data(), +1, acc, N, F.data(), nullptr);
  }
  // (note this can't use the many-vectors interface since the NU change)
  std::complex<double> y = F[0]; // actually use the data so not optimized away
  std::printf(
      "%d reps of 1d1 done in %.3g s,\t%.3g NU pts/s\t(last ier=%d)\nF[0]=%.6g + %.6gi\n",
      reps,
      timer.elapsedsec(),
      reps * M / timer.elapsedsec(),
      ier,
      std::real(y),
      std::imag(y));

  std::printf("repeatedly executing via the guru interface: -------------------\n");
  timer.restart();
  finufft_plan plan;
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.debug   = 0;
  int64_t Ns[] = {N, 1, 1};
  int ntransf  = 1;                // since we do one at a time (neq reps)
  finufft_makeplan(1, 1, Ns, +1, ntransf, acc, &plan, &opts);
  for (int r = 0; r < reps; ++r) { // set the pts and execute
    x[0] = PI * (-1.0 + 2 * static_cast<double>(r) / static_cast<double>(reps)); // one source jiggles around
    // (of course if most sources *were* in fact fixed, use ZGEMM for them!)
    finufft_setpts(plan, M, x.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr);
    c[0] = (1.0 + I) * static_cast<double>(r) / static_cast<double>(reps); // one coeff also jiggles
    ier = finufft_execute(plan, c.data(), F.data());
  }
  finufft_destroy(plan);
  y = F[0];
  std::printf(
      "%d reps of 1d1 done in %.3g s,\t%.3g NU pts/s\t(last ier=%d)\nF[0]=%.6g + %.6gi\n",
      reps,
      timer.elapsedsec(),
      reps * M / timer.elapsedsec(),
      ier,
      std::real(y),
      std::imag(y));
  return ier;
}
