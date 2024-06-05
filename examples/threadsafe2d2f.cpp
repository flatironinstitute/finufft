/* This is a 2D type-2 demo calling single-threaded FINUFFT inside an OpenMP
   loop, to show thread-safety with independent transforms, one per thread.
   It is based on a test code of Penfe,
   submitted github Issue #72. Unlike threadsafe1d1, it does not test the math;
   it is the shell of an application from multi-coil/slice MRI reconstruction.
   Note that since the NU pts are the same in each slice, in fact a vectorized
   multithreaded transform could do all these slices together, and faster.

   To compile (note uses threads rather than omp version of FFTW3):

   g++ -fopenmp threadsafe2d2f.cpp -I../include ../lib/libfinufft.so -o threadsafe2d2f -g
   -Wall

   ./threadsafe2d2f                                   <-- use all threads
   OMP_NUM_THREADS=1 ./threadsafe2d2f                 <-- sequential, 1 thread

   Expected output is 50 lines, each showing exit code 0. It's ok if they're
   mangled due to threads writing to stdout simultaneously.

   Barnett, tidied 11/22/23
*/

// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <complex>
#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

int test_finufft(finufft_opts *opts)
// self-contained small test that one single-prec FINUFFT2D2 has no error/crash
{
  size_t n_rows = 256, n_cols = 256;   // 2d image size
  size_t n_read = 512, n_spokes = 128; // some k-space point params
  size_t M = n_read * n_spokes;        // how many k-space pts; MRI-specific
  std::vector<float> x(M);             // bunch of zero input data
  std::vector<float> y(M);
  std::vector<std::complex<float>> img(n_rows * n_cols); // coeffs
  std::vector<std::complex<float>> ksp(M); // output array (vals @ k-space pts)

  int ier = finufftf2d2(M, x.data(), y.data(), ksp.data(), -1, 1e-3, n_rows, n_cols,
                        img.data(), opts);

  std::cout << "\ttest_finufft: exit code " << ier << ", thread " << omp_get_thread_num()
            << std::endl;
  return ier;
}

int main(int argc, char *argv[]) {
  finufft_opts opts;
  finufftf_default_opts(&opts);
  opts.nthreads = 1;      // *crucial* so each call single-thread; else segfaults

  int n_slices      = 50; // number of transforms. parallelize over slices
  int overallstatus = 0;
#pragma omp parallel for
  for (int i = 0; i < n_slices; i++) {
    int ier = test_finufft(&opts);
    if (ier != 0) overallstatus = 1;
  }

  return overallstatus;
}
