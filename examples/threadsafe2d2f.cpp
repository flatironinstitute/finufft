/* This is a 2D type-2 demo calling single-threaded FINUFFT inside an OpenMP
   loop, to show thread-safety with independent transforms, one per thread.
   It is based on a test code of Penfe, submitted GitHub Issue #72.
   Unlike threadsafe1d1, it does not test the math;
   it is the shell of an application from multi-coil/slice MRI reconstruction.
   Note that since the NU pts are the same in each slice, in fact a vectorized
   multithreaded transform could do all these slices together, and faster.
   Barnett, tidied 11/22/23.
   To compile, see README.  Usage: ./threadsafe2d2f
   Exit code: 0 if every per-slice call returned 0, nonzero otherwise.
*/

// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <cmath>
#include <complex>
#include <cstdio>
#include <omp.h>
#include <vector>
using namespace std;

int test_finufft(finufft_opts *opts)
// self-contained small test that one single-prec FINUFFT2D2 has no error/crash
{
  constexpr size_t n_rows = 256, n_cols = 256;   // 2d image size
  constexpr size_t n_read = 512, n_spokes = 128; // some k-space point params
  constexpr size_t M = n_read * n_spokes;        // how many k-space pts; MRI-specific
  std::vector<float> x(M);                       // bunch of zero input data
  std::vector<float> y(M);
  std::vector<std::complex<float>> img(n_rows * n_cols); // coeffs
  std::vector<std::complex<float>> ksp(M); // output array (vals @ k-space pts)

  int ier = finufftf2d2(M, x.data(), y.data(), ksp.data(), -1, 1e-3, n_rows, n_cols,
                        img.data(), opts);
  if (ier) return ier;
  for (const auto &kv : ksp)
    if (kv != std::complex<float>{}) return 1; // exact zero output; rejects NaN/Inf too
  return 0;
}

int main() {
  finufft_opts opts;
  finufftf_default_opts(&opts);
  opts.nthreads          = 1;  // *crucial* so each call single-thread; else segfaults

  constexpr int n_slices = 50; // number of transforms. parallelize over slices
  int overallstatus      = 0;
#pragma omp parallel for reduction(| : overallstatus)
  for (int i = 0; i < n_slices; i++) {
    int ier = test_finufft(&opts);
    if (ier != 0) overallstatus = 1;
  }

  if (overallstatus) {
    fprintf(stderr, "FAILED: a slice returned an error or a nonzero output\n");
    return 1;
  }
  printf("ok\n");
  return 0;
}
