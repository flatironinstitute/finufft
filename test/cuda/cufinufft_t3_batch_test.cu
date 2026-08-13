// Type 3 must give the same answer whatever batchsize it runs at, including the
// buffers its batch loop shares (CpBatch doubles as the inner type 2's fw).

#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <cufinufft.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <finufft_common/common.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename T>
std::vector<std::complex<T>> run(
    int maxbatchsize, int dim, int M, int nk, int ntransf,
    const thrust::host_vector<T> &x, const thrust::host_vector<T> &y,
    const thrust::host_vector<T> &s, const thrust::host_vector<T> &t,
    const thrust::host_vector<thrust::complex<T>> &c) {
  thrust::device_vector<T> d_x(x), d_y(y), d_s(s), d_t(t);
  thrust::device_vector<thrust::complex<T>> d_c(c), d_fk(size_t(nk) * ntransf);

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_maxbatchsize = maxbatchsize;

  const T tol = std::is_same_v<T, float> ? 1e-5 : 1e-12;
  const int64_t nmodes[3] = {8, 8, 1}; // unused by type 3, but makeplan takes it
  typename std::conditional_t<std::is_same_v<T, float>, cufinufftf_plan, cufinufft_plan>
      plan;
  int ier;
  if constexpr (std::is_same_v<T, float>) {
    ier = cufinufftf_makeplan(3, dim, nmodes, 1, ntransf, tol, &plan, &opts);
    if (ier) return {};
    ier = cufinufftf_setpts(plan, M, d_x.data().get(), d_y.data().get(), nullptr, nk,
                            d_s.data().get(), d_t.data().get(), nullptr);
    if (!ier)
      ier = cufinufftf_execute(plan, (cuFloatComplex *)d_c.data().get(),
                               (cuFloatComplex *)d_fk.data().get());
    cufinufftf_destroy(plan);
  } else {
    ier = cufinufft_makeplan(3, dim, nmodes, 1, ntransf, tol, &plan, &opts);
    if (ier) return {};
    ier = cufinufft_setpts(plan, M, d_x.data().get(), d_y.data().get(), nullptr, nk,
                           d_s.data().get(), d_t.data().get(), nullptr);
    if (!ier)
      ier = cufinufft_execute(plan, (cuDoubleComplex *)d_c.data().get(),
                              (cuDoubleComplex *)d_fk.data().get());
    cufinufft_destroy(plan);
  }
  if (ier) return {};

  thrust::host_vector<thrust::complex<T>> h_fk(d_fk);
  std::vector<std::complex<T>> out(h_fk.size());
  for (size_t i = 0; i < h_fk.size(); ++i) out[i] = {h_fk[i].real(), h_fk[i].imag()};
  return out;
}

template<typename T> int run_prec(const char *label) {
  const int dim = 2, M = 2000, nk = 500, ntransf = 7;
  std::default_random_engine eng(42);
  std::uniform_real_distribution<T> uni(-1, 1);

  thrust::host_vector<T> x(M), y(M), s(nk), t(nk);
  thrust::host_vector<thrust::complex<T>> c(size_t(M) * ntransf);
  for (int j = 0; j < M; ++j) {
    x[j] = T(finufft::common::PI) * uni(eng);
    y[j] = T(finufft::common::PI) * uni(eng);
  }
  for (int k = 0; k < nk; ++k) {
    s[k] = T(12) * uni(eng);
    t[k] = T(12) * uni(eng);
  }
  // Distinct data per transform, so a batch-indexing slip cannot cancel out.
  for (size_t i = 0; i < c.size(); ++i) {
    const T re = uni(eng), im = uni(eng);
    c[i] = thrust::complex<T>(re, im);
  }

  // batchsize 1 is the reference: one transform per batch, nothing shared across a
  // batch boundary.
  const auto ref = run<T>(1, dim, M, nk, ntransf, x, y, s, t, c);
  if (ref.empty()) {
    std::cerr << label << ": reference run failed\n";
    return 1;
  }
  T refnorm = 0;
  for (const auto &v : ref) refnorm = std::max(refnorm, std::abs(v));

  const auto diff = [&](const std::vector<std::complex<T>> &a) {
    T e = 0;
    for (size_t i = 0; i < ref.size(); ++i) e = std::max(e, std::abs(a[i] - ref[i]));
    return e / refnorm;
  };

  // The spreader accumulates with atomics, so even a repeat of the reference differs.
  // That noise floor, not a fixed constant, is what a batchsize may move the answer by.
  const T ctrl = diff(run<T>(1, dim, M, nk, ntransf, x, y, s, t, c));
  // Floor scales with the precision: a deterministic control must not demand bit
  // equality of a batch that only reassociates the sum.
  const T checktol = std::max(T(8) * ctrl, T(100) * std::numeric_limits<T>::epsilon());
  std::cout << label << ": bs=1 repeat (control) " << ctrl << ", tol " << checktol
            << "\n";

  int fails = 0;
  for (int bs : {0, 2, 3, 4, 7, 8}) { // 0 = auto, 8 > ntransf (must clamp)
    const auto got = run<T>(bs, dim, M, nk, ntransf, x, y, s, t, c);
    if (got.size() != ref.size()) {
      std::cerr << label << " maxbatchsize=" << bs << ": run failed\n";
      ++fails;
      continue;
    }
    const T err = diff(got);
    std::cout << label << " maxbatchsize=" << bs << ": rel err vs bs=1 " << err << "\n";
    if (!(err <= checktol)) {
      std::cerr << label << " maxbatchsize=" << bs << ": FAILED (tol " << checktol
                << ")\n";
      ++fails;
    }
  }
  return fails;
}

int main() {
  int fails = run_prec<float>("float") + run_prec<double>("double");
  std::cout << (fails ? "FAILED\n" : "PASSED\n");
  return fails != 0;
}
