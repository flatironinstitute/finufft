#include <cufinufft.h>

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

#include <cuda_runtime.h>

constexpr double pi = 3.14159265358979323846;

int main() {
  /* 2D type 1 guru interface example of calling the CUFINUFFT library from C++
     on the GPU, using STL double complex vectors on the host and
     thrust::device_vector to move data. Mirrors examples/guru2d1.cpp.
     Usage: ./guru2d1
  */
  constexpr int M      = 1e6;  // number of nonuniform points
  constexpr int N      = 1e6;  // approximate total number of modes (N1*N2)
  constexpr double tol = 1e-6; // desired accuracy
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  constexpr std::complex<double> I(0.0, 1.0); // the imaginary unit

  // generate random non-uniform points on (x,y) in [-pi,pi] and complex strengths (c)
  // with components in [-1,1], from a fixed-seed generator:
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  std::vector<double> x(M), y(M);
  std::vector<std::complex<double>> c(M);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(y.begin(), y.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  // choose numbers of output Fourier coefficients in each dimension
  const int N1 = std::round(2.0 * std::sqrt(N));
  const int N2 = std::round(N / N1);

  // output array for the Fourier modes
  std::vector<std::complex<double>> F(N1 * N2);

  constexpr int type = 1, dim = 2, ntrans = 1; // you could also do ntrans>1
  const int64_t Ns[] = {N1, N2};               // N1,N2 as 64-bit int array
  // step 1: make a plan...
  cufinufft_plan plan;
  int ier = cufinufft_makeplan(type, dim, Ns, +1, ntrans, tol, &plan, &opts);
  if (ier > 1) return ier;

  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  const thrust::device_vector<double> d_x = x, d_y = y;
  const auto *h_c = reinterpret_cast<const cuDoubleComplex *>(c.data());
  thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + M);
  thrust::device_vector<cuDoubleComplex> d_F(F.size());

  // step 2: send in M nonuniform points (just x, y in this case)...
  ier = cufinufft_setpts(plan, M, thrust::raw_pointer_cast(d_x.data()),
                         thrust::raw_pointer_cast(d_y.data()), nullptr, 0, nullptr,
                         nullptr, nullptr);
  if (ier > 1) return ier;
  // step 3: do the planned transform to the c strength data, output to F...
  ier = cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                          thrust::raw_pointer_cast(d_F.data()));
  // ... you could now send in new points, and/or do transforms with new c data
  // ...
  if (ier > 1 || cudaDeviceSynchronize() != cudaSuccess) return ier > 1 ? ier : 1;
  thrust::copy(d_F.begin(), d_F.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));
  // step 4: free the memory used by the plan...
  cufinufft_destroy(plan);

  const int k1 = std::round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2 = std::round(-0.35 * N2);

  std::vector<std::complex<double>> terms(M);
  std::transform(x.begin(), x.end(), y.begin(), terms.begin(), [&](auto xj, auto yj) {
    return std::exp(I * (double(k1) * xj + double(k2) * yj));
  });
  std::transform(terms.begin(), terms.end(), c.begin(), terms.begin(),
                 [](auto tj, auto cj) { return tj * cj; });
  const auto Ftest = std::reduce(terms.begin(), terms.end());
  double Fmax      = 0.0;
  for (const auto &v : F) {
    const double a = std::abs(v);
    if (a > Fmax || !std::isfinite(a)) Fmax = a;
  }
  // index in output array for this frequency pair (k1,k2)
  const int indexOut = k1 + N1 / 2 + (k2 + N2 / 2) * N1;
  // compute relative error
  const auto err     = std::abs(F[indexOut] - Ftest) / Fmax;
  if (!std::isfinite(Fmax) || !(err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  std::cout << "2D type-1 NUFFT done (GPU). ier=" << ier << ", err in F[" << indexOut
            << "] rel to max(F) is " << std::setprecision(2) << err << std::endl;
  return 0;
}
