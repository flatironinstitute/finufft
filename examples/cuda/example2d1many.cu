/* This is an example of performing 2d1many
   in single precision.
*/

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include <cufinufft.h>

#include <cuda_runtime.h>

constexpr float pi = 3.14159265358979323846f;

int main()
/*
 * example code for 2D Type 1 transformation.
 *
 * To compile the code:
 * nvcc example2d1many.cu -o example2d1many -I/loc/to/cufinufft/include
 * /loc/to/cufinufft/lib-static/libcufinufft.a -lcudart -lcufft -lnvToolsExt
 *
 * or
 * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/loc/to/cufinufft/lib
 * nvcc example2d1many.cu -o example2d1many -I/loc/to/cufinufft/include
 * -L/loc/to/cufinufft/lib/ -lcufinufft
 *
 *
 */
{
  std::cout << std::scientific << std::setprecision(3);

  int ier;
  int N1      = 256;
  int N2      = 256;
  int M       = 65536;
  int ntransf = 2;
  int iflag   = 1;
  float tol   = 1e-6;

  std::vector<float> x(M), y(M);
  std::vector<std::complex<float>> c(M * ntransf), fk(N1 * N2 * ntransf);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> upi(-pi, pi), u1(-1.0f, 1.0f);

  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(y.begin(), y.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<float>{u1(rng), u1(rng)}; });

  const thrust::device_vector<float> d_x = x, d_y = y;
  const auto *h_c = reinterpret_cast<const cuFloatComplex *>(c.data());
  thrust::device_vector<cuFloatComplex> d_c(h_c, h_c + c.size());
  thrust::device_vector<cuFloatComplex> d_fk(fk.size(), thrust::no_init);

  cufinufftf_plan dplan;

  int dim = 2;
  int64_t nmodes[3];
  int type  = 1;

  nmodes[0] = N1;
  nmodes[1] = N2;
  nmodes[2] = 1;

  ier       = cufinufftf_makeplan(type, dim, nmodes, iflag, ntransf, tol, &dplan, NULL);
  if (ier > 0) return ier;

  ier =
      cufinufftf_setpts(dplan, M, thrust::raw_pointer_cast(d_x.data()),
                        thrust::raw_pointer_cast(d_y.data()), NULL, 0, NULL, NULL, NULL);
  if (ier > 0) return ier;

  ier = cufinufftf_execute(dplan, thrust::raw_pointer_cast(d_c.data()),
                           thrust::raw_pointer_cast(d_fk.data()));
  if (ier > 0) return ier;

  ier = cufinufftf_destroy(dplan);
  if (ier > 0) return ier;

  thrust::copy(d_fk.begin(), d_fk.end(), reinterpret_cast<cuFloatComplex *>(fk.data()));

  std::cout << std::endl << "Accuracy check:" << std::endl;
  const int N   = N1 * N2;
  const int i   = ntransf - 1; // check one transform, the last
  const int nt1 = (int)(0.37 * N1), nt2 = (int)(0.26 * N2); // choose some mode index to
                                                            // check
  const std::complex<float> J = std::complex<float>(0, 1) * (float)iflag;
  std::vector<std::complex<float>> terms(M);
  std::transform(x.begin(), x.end(), y.begin(), terms.begin(),
                 [&](auto xj, auto yj) { return std::exp(J * (nt1 * xj + nt2 * yj)); });
  std::transform(terms.begin(), terms.end(), c.begin() + i * M, terms.begin(),
                 [](auto tj, auto cj) { return tj * cj; });
  const auto Ft   = std::reduce(terms.begin(), terms.end()); // crude direct
  const auto Fmax = std::abs(
      *std::max_element(fk.begin() + i * N, fk.begin() + (i + 1) * N,
                        [](auto a, auto b) { return std::abs(a) < std::abs(b); }));
  const int it   = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2); // index in complex F as 1d array
  const auto err = std::abs(Ft - fk[it + i * N]) / Fmax;
  printf("[gpu %3d] one mode: rel err in F[%d,%d] is %.3g\n", i, nt1, nt2, err);

  return !(err < 10 * tol);
}
