#include <complex>
#include <mutex>
#include <vector>

#include <fftw3.h>
#include <finufft.h>
#include <omp.h>

// This file tests the user locking mechanism for multi-threaded FFTW. This
// demonstrates a user lock to prevent FFTW plan calls from interfering with
// finufft plan calls (make/destroy).
// Robert Blackwell. Based on bug identified by Jonas Krimmer (9/17/24)
// See discussion at https://github.com/ludvigak/FINUFFT.jl/issues/62

constexpr int N = 65384;

// Example user lock functions
void locker(void *lck) { reinterpret_cast<std::mutex *>(lck)->lock(); }
void unlocker(void *lck) { reinterpret_cast<std::mutex *>(lck)->unlock(); }

int main() {
  int64_t Ns[3]; // guru describes mode array by vector [N1,N2..]
  Ns[0] = N;
  std::mutex lck;

  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.nthreads        = 1;
  opts.debug           = 0;
  opts.fftw_lock_fun   = locker;
  opts.fftw_unlock_fun = unlocker;
  opts.fftw_lock_data  = reinterpret_cast<void *>(&lck);

  // random nonuniform points (x) and complex strengths (c)...
  std::vector<std::complex<double>> c(N);

  omp_set_num_threads(8);

  // init FFTW threads
  fftw_init_threads();

// FFTW and FINUFFT execution using OpenMP parallelization
#pragma omp parallel for
  for (int j = 0; j < 100; ++j) {
    // allocate output array for FFTW...
    std::vector<std::complex<double>> F1(N);

    // FFTW plan
    lck.lock();
    fftw_plan_with_nthreads(1);
    fftw_plan plan = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex *>(c.data()),
                                      reinterpret_cast<fftw_complex *>(F1.data()),
                                      FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_destroy_plan(plan);
    lck.unlock();

    // FINUFFT plan
    finufft_plan nufftplan;
    finufft_makeplan(1, 1, Ns, 1, 1, 1e-6, &nufftplan, &opts);
    finufft_destroy(nufftplan);
  }
  return 0;
}
