/* This is a 2D type-2 demo calling FINUFFT for big number of transforms
   Then to compile (note uses threads rather than omp version of FFTW3):

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

void test_finufft(nufft_opts* opts)
// self-contained small test that single-prec FINUFFT2D2 no error nor crash
{
    size_t nj = 129*129*2;
    size_t ms = 129, mt = 129;
    size_t ntrans = 75000;
    std::vector<float> x(nj);       // bunch of zero data
    std::vector<float> y(nj);
    std::vector<std::complex<float>> cj(ntrans*nj);
    std::vector<std::complex<float>> fk(ntrans*ms*mt);

    int ier = finufftf2d2many(ntrans, nj, x.data(), y.data(), cj.data(),
                          -1, 1e-3, ms, mt, fk.data(), opts);

    std::cout << "\ttest_finufft: " << ier << std::endl;
}

int main(int argc, char* argv[])
{
  nufft_opts opts;
  finufftf_default_opts(&opts);
  test_finufft(&opts);
  return 0;
}
