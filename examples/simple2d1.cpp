// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// also needed for this example...
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {

  /* Simple 2D type-1 example of calling the FINUFFT library from C++, using plain
     arrays of C++ complex numbers, with a math test. Double precision version.

     Compile multithreaded with
     g++ -fopenmp simple2d1.cpp -I ../src ../lib-static/libfinufft.a -o simple2d1 -lfftw3
     -lfftw3_omp -lm single core with: g++ simple2d1.cpp -I ../src
     ../lib-static/libfinufft.a -o simple2d1 -lfftw3 -lm

     Usage:  ./simple2d1
  */

  int M      = 1e6;  // number of nonuniform points
  int N      = 1e6;  // approximate total number of modes (N1*N2)
  double tol = 1e-6; // desired accuracy
  finufft_opts opts;
  finufft_default_opts(&opts);
  complex<double> I(0.0, 1.0); // the imaginary unit

  // generate random non-uniform points on (x,y) and complex strengths (c):
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);

  for (int i = 0; i < M; i++) {
    x[i] = M_PI * (2 * (double)rand() / RAND_MAX - 1); // uniform random in [-pi, pi)
    y[i] = M_PI * (2 * (double)rand() / RAND_MAX - 1); // uniform random in [-pi, pi)

    // each component uniform random in [-1,1]
    c[i] =
        2 * ((double)rand() / RAND_MAX - 1) + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }

  // choose numbers of output Fourier coefficients in each dimension
  int N1 = round(2.0 * sqrt(N));
  int N2 = round(N / N1);

  // output array for the Fourier modes
  vector<complex<double>> F(N1 * N2);

  // call the NUFFT (with iflag += 1): note passing in pointers...
  opts.upsampfac = 1.25;
  int ier        = finufft2d1(M, &x[0], &y[0], &c[0], 1, tol, N1, N2, &F[0], &opts);

  int k1 = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  int k2 = round(-0.35 * N2);

  complex<double> Ftest(0, 0);
  for (int j = 0; j < M; j++)
    Ftest += c[j] * exp(I * ((double)k1 * x[j] + (double)k2 * y[j]));

  // compute inf norm of F
  double Fmax = 0.0;
  for (int m = 0; m < N1 * N2; m++) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }

  // indices in output array for this frequency pair (k1,k2)
  int k1out    = k1 + N1 / 2;
  int k2out    = k2 + N2 / 2;
  int indexOut = k1out + k2out * (N1);

  // compute relative error
  double err = abs(F[indexOut] - Ftest) / Fmax;
  cout << "2D type-1 NUFFT done. ier=" << ier << ", err in F[" << indexOut
       << "] rel to max(F) is " << setprecision(2) << err << endl;
  return ier;
}
