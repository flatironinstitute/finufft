/* This is a 2D type-2 demo calling FINUFFT for big number of transforms, that
   results in a number of data exceeding the max signed int value of 2^31.
   This verifies correct handling via int64_t (8byte) indexing.
   It takes about 30 s to run on 8 threads, and demands about 30 GB of RAM.

   See makefile for compilation. Libin Lu 6/7/22; edits Alex Barnett.
*/

// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <complex>
#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

int test_finufft(finufft_opts *opts) {
  size_t nj = 129 * 129 * 2;
  size_t ms = 129, mt = 129;
  size_t ntrans = 75000;    // the point is: 129*129*2*75000 > 2^31 ~ 2.15e9
  std::vector<float> x(nj); // bunch of zero data
  std::vector<float> y(nj);
  std::vector<std::complex<float>> cj(ntrans * nj);
  std::vector<std::complex<float>> fk(ntrans * ms * mt);

  int ier = finufftf2d2many(ntrans, nj, x.data(), y.data(), cj.data(), -1, 1e-3, ms, mt,
                            fk.data(), opts);

  std::cout << "\tbig2d2f finufft status: " << ier << std::endl;
  return ier;
}

int main(int argc, char *argv[]) {
  finufft_opts opts;
  finufftf_default_opts(&opts);
  return test_finufft(&opts);
}
