/* This is an example of performing 2d2many
   in double precision.
*/

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include <cufinufft.h>

#include <cuda_runtime.h>

constexpr double pi = 3.14159265358979323846;

int main()
/*
 * example code for 2D Type 1 transformation.
 *
 * To compile the code:
 * nvcc example2d2many.cu -o example2d2many loc/to/cufinufft/lib-static/libcufinufft.a
 * -I/loc/to/cufinufft/include -lcudart -lcufft -lnvToolsExt
 *
 * or
 * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/loc/to/cufinufft/lib
 * nvcc example2d2many.cu -L/loc/to/cufinufft/lib/ -I/loc/to/cufinufft/include -o
 * example2d1 -lcufinufft
 *
 *
 */
{
  std::cout << std::scientific << std::setprecision(3);

  int ier;
  int N1           = 128;
  int N2           = 128;
  int M            = 10;
  int ntransf      = 4;
  int maxbatchsize = 4;
  int iflag        = 1;
  double tol       = 1e-6;

  std::vector<double> x(M), y(M);
  std::vector<std::complex<double>> c(M * ntransf), fk(N1 * N2 * ntransf);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);

  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(y.begin(), y.end(), [&] { return upi(rng); });
  std::generate(fk.begin(), fk.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  const thrust::device_vector<double> d_x = x, d_y = y;
  const auto *h_fk = reinterpret_cast<const cuDoubleComplex *>(fk.data());
  thrust::device_vector<cuDoubleComplex> d_fk(h_fk, h_fk + fk.size());
  thrust::device_vector<cuDoubleComplex> d_c(c.size());

  cufinufft_plan dplan;

  int dim = 2;
  int64_t nmodes[3];
  int type  = 2;

  nmodes[0] = N1;
  nmodes[1] = N2;
  nmodes[2] = 1;

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_maxbatchsize = maxbatchsize;

  ier = cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, &dplan, &opts);
  if (ier > 0) return ier;

  ier = cufinufft_setpts(dplan, M, thrust::raw_pointer_cast(d_x.data()),
                         thrust::raw_pointer_cast(d_y.data()), NULL, 0, NULL, NULL, NULL);
  if (ier > 0) return ier;

  ier = cufinufft_execute(dplan, thrust::raw_pointer_cast(d_c.data()),
                          thrust::raw_pointer_cast(d_fk.data()));
  if (ier > 0) return ier;

  ier = cufinufft_destroy(dplan);
  if (ier > 0) return ier;

  thrust::copy(d_c.begin(), d_c.end(), reinterpret_cast<cuDoubleComplex *>(c.data()));

  const int t  = ntransf - 1; // check one transform, the last
  const int jt = M / 2;       // check arbitrary choice of one targ pt
  const std::complex<double> J(0, iflag * 1);
  std::complex<double> ct(0, 0);
  int m = 0;
  for (int m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order over F
    for (int m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
      ct += fk[t * N1 * N2 + m++] * exp(J * (m1 * x[jt] + m2 * y[jt])); // crude direct

  double cmax = 0;
  for (int i = 0; i < ntransf * M; ++i) {
    const double a = std::abs(c[i]);
    if (a > cmax || !std::isfinite(a)) cmax = a; // a NaN element also lands in cmax
  }
  const auto err = std::abs(c[t * M + jt] - ct) / cmax;
  if (!std::isfinite(cmax) || !std::isfinite(err) || !(err < 10 * tol)) {
    printf("FAILED: c non-finite or rel err %.3g >= %.3g\n", err, 10 * tol);
    return 1;
  }
  std::cout << std::endl << "Accuracy check:" << std::endl;
  printf("[gpu %3d] one targ: rel err in c[%d] is %.3g\n", t, jt, err);

  return 0;
}
