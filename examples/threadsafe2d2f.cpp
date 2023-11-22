/* This is a 2D type-2 demo calling single-threaded FINUFFT inside an OpenMP
   loop, to show thread-safety. It is closely based on a test code of Penfe,
   submitted github Issue #72. Unlike threadsafe1d1, it does not test the math;
   it is the shell of an application from MRI reconstruction.

   To compile (note uses threads rather than omp version of FFTW3):

   g++ -fopenmp threadsafe2d2f.cpp -I../include ../lib/libfinufft.so -o threadsafe2d2f -lfftw3 -lfftw3_threads -lm -g -Wall

*/

// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <vector>
#include <complex>
#include <iostream>
#include <omp.h>
using namespace std;

void test_finufft(finufft_opts* opts)
// self-contained small test that single-prec FINUFFT2D2 no error nor crash
{
    size_t n_rows = 8, n_cols = 8;
    size_t n_read = 16, n_spokes = 4;
    std::vector<float> x(n_read * n_spokes);       // bunch of zero data
    std::vector<float> y(n_read * n_spokes);
    std::vector<std::complex<float>> img(n_rows * n_cols);
    std::vector<std::complex<float>> ksp(n_read * n_spokes);

    int ier = finufftf2d2(n_read * n_spokes, x.data(), y.data(), ksp.data(),
                          -1, 1e-3, n_rows, n_cols, img.data(), opts);

    std::cout << "\ttest_finufft: " << ier << ", thread " << omp_get_thread_num() << std::endl;
}

int main(int argc, char* argv[])
{
  finufft_opts opts;
  finufftf_default_opts(&opts);
  opts.nthreads = 1;     // this is *crucial* so that each call single-thread

  int n_slices = 50;     // parallelize over slices
#pragma omp parallel for num_threads(8)
  for (int i = 0; i < n_slices; i++)
    {
      test_finufft(&opts);
    }
  
  return 0;
}
