/* Demonstrate guru FINUFFT interface performing a stack of 1d type 1
   transforms in a single execute call. See guru1d1.cpp for other guru
   features demonstrated. Barnett 11/22/23
   To compile, see README.
   Usage: ./gurumany1d1           (exit code 0 indicates success)
*/

// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// specific to this demo...
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const double PI = 3.141592653589793238462643383279502884;

int main(int argc, char *argv[]) {
  int M      = 2e5;          // number of nonuniform points
  int N      = 1e5;          // number of modes
  double tol = 1e-9;         // desired accuracy
  int ntrans = 100;          // request a bunch of transforms in the execute
  int isign  = +1;           // sign of i in the transform math definition

  int type = 1, dim = 1;     // 1d1
  int64_t Ns[3] = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  finufft_plan plan;         // creates a plan struct (NULL below: default opts)
  finufft_makeplan(type, dim, Ns, isign, ntrans, tol, &plan, nullptr);

  // generate random nonuniform points and pass to FINUFFT
  std::vector<double> x(M);
  for (int j = 0; j < M; ++j)
    x[j] = PI * (2 * ((double)std::rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
  finufft_setpts(plan, M, x.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr);

  // generate ntrans complex strength vectors each of length M (the slow bit!)
  std::vector<std::complex<double>> c(M * ntrans); // plain contiguous storage
  const std::complex<double> I(0.0, 1.0);
  for (int j = 0; j < M * ntrans; ++j)
    c[j] = 2 * ((double)std::rand() / RAND_MAX) - 1 +
           I * (2 * ((double)std::rand() / RAND_MAX) - 1);

  // alloc output array for the Fourier modes, then do the transform
  std::vector<std::complex<double>> F(N * ntrans);
  printf("guru many 1D type-1 double-prec, tol=%.3g, executing %d transforms "
         "(vectorized), each size %d NU pts to %d modes...\n",
         tol, ntrans, M, N);
  int ier = finufft_execute(plan, c.data(), F.data());

  // could now change c, do another execute, do another setpts, execute, etc...

  finufft_destroy(plan); // don't forget! we're done with transforms of this size

  // rest is math checking and reporting...
  int k     = 42519;                                // check the answer just for this mode
  int trans = 71;                                   // ...testing in just this transform
  assert(k >= -(double)N / 2 && k < (double)N / 2); // ensure meaningful test
  assert(trans >= 0 && trans < ntrans);
  std::complex<double> Ftest = std::complex<double>(0, 0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * std::exp(I * (double)k * x[j]); // c offset to trans
  double Fmax = 0.0; // compute inf norm of F for selected transform
  for (int m = 0; m < N; ++m) {
    double aF = std::abs(F[m + N * trans]);
    if (aF > Fmax) Fmax = aF;
  }
  int nout   = k + N / 2 + N * trans; // output index for freq mode k in the trans
  double err = std::abs(F[nout] - Ftest) / Fmax;
  printf("\tdone: ier=%d; for transform %d, rel err in F[%d] is %.3g\n", ier, trans, k,
         err);

  return ier;
}
