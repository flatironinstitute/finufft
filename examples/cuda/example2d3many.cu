/* This is an example of performing 2d3many
   in double precision.
*/

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include <cufinufft.h>

#include <cuda_runtime.h>

constexpr double pi = 3.14159265358979323846;

int main()
/*
 * example code for 2D Type 3 transformation.
 *
 * To compile the code:
 * nvcc example2d3many.cu -o example2d3many loc/to/cufinufft/lib-static/libcufinufft.a
 * -I/loc/to/cufinufft/include -lcudart -lcufft -lnvToolsExt
 *
 * or
 * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/loc/to/cufinufft/lib
 * nvcc example2d3many.cu -example2d3many -L/loc/to/cufinufft/lib/
 * -I/loc/to/cufinufft/include -lcufinufft
 *
 *
 */
{
  std::cout << std::scientific << std::setprecision(3);

  int ier;
  int M            = 10;
  int N            = 20;
  int ntransf      = 4;
  int maxbatchsize = 4;
  int iflag        = 1;
  double tol       = 1e-6;

  std::vector<double> x(M), y(M), s(N), t(N);
  std::vector<std::complex<double>> c(M * ntransf), fk(N * ntransf);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);

  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(y.begin(), y.end(), [&] { return upi(rng); });
  std::generate(s.begin(), s.end(), [&] { return u1(rng); });
  std::generate(t.begin(), t.end(), [&] { return u1(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  const thrust::device_vector<double> d_x = x, d_y = y, d_s = s, d_t = t;
  const auto *h_c = reinterpret_cast<const cuDoubleComplex *>(c.data());
  thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + c.size());
  thrust::device_vector<cuDoubleComplex> d_fk(fk.size());

  cufinufft_plan dplan;

  int dim           = 2;
  int64_t nmodes[3] = {N, 1, 1};
  int type          = 3;

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);

  ier = cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, &dplan, &opts);
  if (ier > 0) return ier;

  ier = cufinufft_setpts(dplan, M, thrust::raw_pointer_cast(d_x.data()),
                         thrust::raw_pointer_cast(d_y.data()), NULL, N,
                         thrust::raw_pointer_cast(d_s.data()),
                         thrust::raw_pointer_cast(d_t.data()), NULL);
  if (ier > 0) return ier;

  ier = cufinufft_execute(dplan, thrust::raw_pointer_cast(d_c.data()),
                          thrust::raw_pointer_cast(d_fk.data()));
  if (ier > 0) return ier;

  ier = cufinufft_destroy(dplan);
  if (ier > 0) return ier;

  thrust::copy(d_fk.begin(), d_fk.end(), reinterpret_cast<cuDoubleComplex *>(fk.data()));

  std::cout << std::endl << "Accuracy check:" << std::endl;
  const int tr = ntransf - 1; // check one transform, the last
  const int jt = N / 2;       // check arbitrary choice of one targ pt
  const std::complex<double> J(0, iflag * 1);
  std::vector<std::complex<double>> terms(M);
  std::transform(x.begin(), x.end(), y.begin(), terms.begin(), [&](auto xj, auto yj) {
    return std::exp(J * (s[jt] * xj + t[jt] * yj));
  });
  std::transform(terms.begin(), terms.end(), c.begin() + tr * M, terms.begin(),
                 [](auto tj, auto cj) { return tj * cj; });
  const auto fkt = std::reduce(terms.begin(), terms.end());
  double Fmax    = 0.0;
  for (const auto &v : fk) {
    const double a = std::abs(v);
    if (a > Fmax || !std::isfinite(a)) Fmax = a;
  }
  const auto err = std::abs(fk[tr * N + jt] - fkt) / Fmax;
  if (!std::isfinite(Fmax) || !(err < 10 * tol)) {
    fprintf(stderr, "FAILED: fk non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("[gpu %3d] one targ: rel err in c[%d] is %.3g\n", tr, jt, err);

  return 0;
}
