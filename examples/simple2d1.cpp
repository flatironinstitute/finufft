// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// also needed for this example...
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

static const double PI = 3.141592653589793238462643383279502884;

int main(int argc, char *argv[]) {

  /* Simple 2D type-1 example of calling the FINUFFT library from C++, using plain
     arrays of C++ complex numbers, with a math test. Double precision version.
     To compile, see README. Usage:  ./simple2d1
  */

  int M      = 1e6;  // number of nonuniform points
  int N      = 1e6;  // approximate total number of modes (N1*N2)
  double tol = 1e-6; // desired accuracy
  finufft_opts opts;
  finufft_default_opts(&opts);
  std::complex<double> I(0.0, 1.0); // the imaginary unit

  // generate random non-uniform points on (x,y) and complex strengths (c):
  std::vector<double> x(M), y(M);
  std::vector<std::complex<double>> c(M);

  for (int i = 0; i < M; i++) {
    x[i] = PI * (2 * (double)std::rand() / RAND_MAX - 1); // uniform random in [-pi, pi)
    y[i] = PI * (2 * (double)std::rand() / RAND_MAX - 1); // uniform random in [-pi, pi)

    // each component uniform random in [-1,1]
    c[i] = 2 * ((double)std::rand() / RAND_MAX - 1) +
           I * (2 * ((double)std::rand() / RAND_MAX) - 1);
  }

  // choose numbers of output Fourier coefficients in each dimension
  int N1 = (int)std::round(2.0 * std::sqrt(N));
  int N2 = (int)std::round(N / N1);

  // output array for the Fourier modes
  std::vector<std::complex<double>> F(N1 * N2);

  // call the NUFFT (with iflag += 1): note passing in pointers...
  opts.upsampfac = 1.25;
  int ier        = finufft2d1(M, &x[0], &y[0], &c[0], 1, tol, N1, N2, &F[0], &opts);

  int k1 = (int)std::round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  int k2 = (int)std::round(-0.35 * N2);

  std::complex<double> Ftest(0, 0);
  for (int j = 0; j < M; j++)
    Ftest += c[j] * std::exp(I * ((double)k1 * x[j] + (double)k2 * y[j]));

  // compute inf norm of F
  double Fmax = 0.0;
  for (int m = 0; m < N1 * N2; m++) {
    double aF = std::abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }

  // indices in output array for this frequency pair (k1,k2)
  int k1out    = k1 + N1 / 2;
  int k2out    = k2 + N2 / 2;
  int indexOut = k1out + k2out * (N1);

  // compute relative error
  double err = std::abs(F[indexOut] - Ftest) / Fmax;
  std::cout << "2D type-1 NUFFT done. ier=" << ier << ", err in F[" << indexOut
            << "] rel to max(F) is " << std::setprecision(2) << err << std::endl;
  return ier;
}
