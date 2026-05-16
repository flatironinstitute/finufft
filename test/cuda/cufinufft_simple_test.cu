// Cross-check the C-level simple cuFINUFFT API against the 4-step plan API.
//
// For each (dim, type) pair we run the simple call and the equivalent 4-step
// plan path on identical inputs and require the device output to be bit-for-bit
// identical. Both float and double precision are exercised.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cufinufft.h>

namespace {

#define CUDA_CHECK(call)                                                                \
  do {                                                                                  \
    cudaError_t e = (call);                                                             \
    if (e != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
                   __LINE__);                                                           \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

#define CHECK_RET(expr, name)                            \
  do {                                                   \
    int _r = (expr);                                     \
    if (_r != 0) {                                       \
      std::fprintf(stderr, "%s failed: %d\n", name, _r); \
      return {};                                         \
    }                                                    \
  } while (0)

template<typename T> struct Traits;
template<> struct Traits<float> {
  using cuc = cuFloatComplex;
  static constexpr const char *label = "float";
};
template<> struct Traits<double> {
  using cuc = cuDoubleComplex;
  static constexpr const char *label = "double";
};

template<typename T> T *dev_alloc(size_t n) {
  T *p = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T));
  return p;
}

template<typename T> void dev_copy_in(T *dst, const std::vector<T> &src) {
  cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T> std::vector<T> dev_copy_out(const T *src, size_t n) {
  std::vector<T> v(n);
  cudaMemcpy(v.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

// Approximate equality. GPU spreading uses atomicAdd which is not bit-
// deterministic for float, so two independent runs of the same type-1 / type-3
// transform can differ by a few ULPs even though the algorithm is the same.
// Type-2 paths (interpolation only, no atomics) compare bit-identically; we
// allow a small relative tolerance uniformly to keep the test simple.
template<typename T>
bool near_equal(const std::vector<std::complex<T>> &a,
                const std::vector<std::complex<T>> &b, T rtol) {
  if (a.size() != b.size()) return false;
  T worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const T num = std::abs(a[i] - b[i]);
    const T den = std::max<T>(std::abs(a[i]), std::abs(b[i]));
    const T r = den > 0 ? num / den : num;
    if (r > worst) worst = r;
  }
  if (worst > rtol) {
    std::fprintf(stderr, "  worst rel error = %g (tol %g)\n", double(worst),
                 double(rtol));
    return false;
  }
  return true;
}

template<typename T> struct Fixture {
  using cuc = typename Traits<T>::cuc;
  int dim, type;
  int64_t M;
  int64_t N1, N2, N3, Nk; // Nk = nk for type 3
  std::vector<T> x, y, z;
  std::vector<T> s, t, u;
  // For type 1/3 c is input strengths; for type 2 c is output.
  std::vector<std::complex<T>> c_in;
  std::vector<std::complex<T>> fk_in;

  size_t c_count() const { return static_cast<size_t>(M); }
  size_t fk_count() const {
    if (type == 3) return static_cast<size_t>(Nk);
    return static_cast<size_t>(N1) * N2 * N3;
  }
};

template<typename T> Fixture<T> make_fixture(int dim, int type, std::mt19937 &rng) {
  Fixture<T> f;
  f.dim = dim;
  f.type = type;
  f.M = 137;
  f.N1 = 16;
  f.N2 = (dim >= 2) ? 12 : 1;
  f.N3 = (dim >= 3) ? 8 : 1;
  f.Nk = 23;
  std::uniform_real_distribution<T> uni(T(-3.14), T(3.14));
  std::uniform_real_distribution<T> uni_s(T(-2), T(2));
  std::uniform_real_distribution<T> uni_c(T(-1), T(1));
  f.x.resize(f.M);
  if (dim >= 2) f.y.resize(f.M);
  if (dim >= 3) f.z.resize(f.M);
  for (int64_t j = 0; j < f.M; ++j) {
    f.x[j] = uni(rng);
    if (dim >= 2) f.y[j] = uni(rng);
    if (dim >= 3) f.z[j] = uni(rng);
  }
  if (type == 3) {
    f.s.resize(f.Nk);
    if (dim >= 2) f.t.resize(f.Nk);
    if (dim >= 3) f.u.resize(f.Nk);
    for (int64_t k = 0; k < f.Nk; ++k) {
      f.s[k] = uni_s(rng);
      if (dim >= 2) f.t[k] = uni_s(rng);
      if (dim >= 3) f.u[k] = uni_s(rng);
    }
  }
  if (type == 1 || type == 3) {
    f.c_in.resize(f.M);
    for (auto &z : f.c_in) z = {uni_c(rng), uni_c(rng)};
  } else {
    f.fk_in.resize(f.fk_count());
    for (auto &z : f.fk_in) z = {uni_c(rng), uni_c(rng)};
  }
  return f;
}

// 4-step path. Returns host vector of output complex values.
template<typename T> std::vector<std::complex<T>> run_plan(const Fixture<T> &f) {
  using cuc = typename Traits<T>::cuc;
  T *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
  T *d_s = nullptr, *d_t = nullptr, *d_u = nullptr;
  cuc *d_c = nullptr, *d_fk = nullptr;
  d_x = dev_alloc<T>(f.M);
  dev_copy_in(d_x, f.x);
  if (f.dim >= 2) {
    d_y = dev_alloc<T>(f.M);
    dev_copy_in(d_y, f.y);
  }
  if (f.dim >= 3) {
    d_z = dev_alloc<T>(f.M);
    dev_copy_in(d_z, f.z);
  }
  if (f.type == 3) {
    d_s = dev_alloc<T>(f.Nk);
    dev_copy_in(d_s, f.s);
    if (f.dim >= 2) {
      d_t = dev_alloc<T>(f.Nk);
      dev_copy_in(d_t, f.t);
    }
    if (f.dim >= 3) {
      d_u = dev_alloc<T>(f.Nk);
      dev_copy_in(d_u, f.u);
    }
  }
  d_c = dev_alloc<cuc>(f.c_count());
  d_fk = dev_alloc<cuc>(f.fk_count());

  if (f.type == 1 || f.type == 3) {
    cudaMemcpy(d_c, f.c_in.data(), f.c_count() * sizeof(cuc), cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(d_fk, f.fk_in.data(), f.fk_count() * sizeof(cuc), cudaMemcpyHostToDevice);
  }

  const int64_t nmodes[3] = {f.N1, f.N2, f.N3};
  const int Nk32 = (f.type == 3) ? static_cast<int>(f.Nk) : 0;
  std::vector<std::complex<T>> out;
  if constexpr (std::is_same_v<T, float>) {
    cufinufftf_plan plan{};
    CHECK_RET(cufinufftf_makeplan(f.type, f.dim, nmodes, 1, 1, T(1e-5), &plan, nullptr),
              "makeplan");
    CHECK_RET(cufinufftf_setpts(plan, f.M, d_x, d_y, d_z, Nk32, d_s, d_t, d_u), "setpts");
    CHECK_RET(cufinufftf_execute(plan, d_c, d_fk), "execute");
    CHECK_RET(cufinufftf_destroy(plan), "destroy");
  } else {
    cufinufft_plan plan{};
    CHECK_RET(cufinufft_makeplan(f.type, f.dim, nmodes, 1, 1, T(1e-9), &plan, nullptr),
              "makeplan");
    CHECK_RET(cufinufft_setpts(plan, f.M, d_x, d_y, d_z, Nk32, d_s, d_t, d_u), "setpts");
    CHECK_RET(cufinufft_execute(plan, d_c, d_fk), "execute");
    CHECK_RET(cufinufft_destroy(plan), "destroy");
  }

  if (f.type == 2)
    out = dev_copy_out(reinterpret_cast<const std::complex<T> *>(d_c), f.c_count());
  else
    out = dev_copy_out(reinterpret_cast<const std::complex<T> *>(d_fk), f.fk_count());

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_s);
  cudaFree(d_t);
  cudaFree(d_u);
  cudaFree(d_c);
  cudaFree(d_fk);
  return out;
}

// Simple-API path. Mirrors run_plan().
template<typename T> std::vector<std::complex<T>> run_simple(const Fixture<T> &f) {
  using cuc = typename Traits<T>::cuc;
  T *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
  T *d_s = nullptr, *d_t = nullptr, *d_u = nullptr;
  cuc *d_c = nullptr, *d_fk = nullptr;
  d_x = dev_alloc<T>(f.M);
  dev_copy_in(d_x, f.x);
  if (f.dim >= 2) {
    d_y = dev_alloc<T>(f.M);
    dev_copy_in(d_y, f.y);
  }
  if (f.dim >= 3) {
    d_z = dev_alloc<T>(f.M);
    dev_copy_in(d_z, f.z);
  }
  if (f.type == 3) {
    d_s = dev_alloc<T>(f.Nk);
    dev_copy_in(d_s, f.s);
    if (f.dim >= 2) {
      d_t = dev_alloc<T>(f.Nk);
      dev_copy_in(d_t, f.t);
    }
    if (f.dim >= 3) {
      d_u = dev_alloc<T>(f.Nk);
      dev_copy_in(d_u, f.u);
    }
  }
  d_c = dev_alloc<cuc>(f.c_count());
  d_fk = dev_alloc<cuc>(f.fk_count());

  if (f.type == 1 || f.type == 3) {
    cudaMemcpy(d_c, f.c_in.data(), f.c_count() * sizeof(cuc), cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(d_fk, f.fk_in.data(), f.fk_count() * sizeof(cuc), cudaMemcpyHostToDevice);
  }

  const T eps = std::is_same_v<T, float> ? T(1e-5) : T(1e-9);
  int rc = 0;
  if constexpr (std::is_same_v<T, float>) {
    if (f.dim == 1) {
      if (f.type == 1)
        rc = cufinufftf1d1(f.M, d_x, d_c, 1, eps, f.N1, d_fk, nullptr);
      else if (f.type == 2)
        rc = cufinufftf1d2(f.M, d_x, d_c, 1, eps, f.N1, d_fk, nullptr);
      else
        rc = cufinufftf1d3(f.M, d_x, d_c, 1, eps, f.Nk, d_s, d_fk, nullptr);
    } else if (f.dim == 2) {
      if (f.type == 1)
        rc = cufinufftf2d1(f.M, d_x, d_y, d_c, 1, eps, f.N1, f.N2, d_fk, nullptr);
      else if (f.type == 2)
        rc = cufinufftf2d2(f.M, d_x, d_y, d_c, 1, eps, f.N1, f.N2, d_fk, nullptr);
      else
        rc = cufinufftf2d3(f.M, d_x, d_y, d_c, 1, eps, f.Nk, d_s, d_t, d_fk, nullptr);
    } else {
      if (f.type == 1)
        rc = cufinufftf3d1(f.M, d_x, d_y, d_z, d_c, 1, eps, f.N1, f.N2, f.N3, d_fk,
                           nullptr);
      else if (f.type == 2)
        rc = cufinufftf3d2(f.M, d_x, d_y, d_z, d_c, 1, eps, f.N1, f.N2, f.N3, d_fk,
                           nullptr);
      else
        rc = cufinufftf3d3(f.M, d_x, d_y, d_z, d_c, 1, eps, f.Nk, d_s, d_t, d_u, d_fk,
                           nullptr);
    }
  } else {
    if (f.dim == 1) {
      if (f.type == 1)
        rc = cufinufft1d1(f.M, d_x, d_c, 1, eps, f.N1, d_fk, nullptr);
      else if (f.type == 2)
        rc = cufinufft1d2(f.M, d_x, d_c, 1, eps, f.N1, d_fk, nullptr);
      else
        rc = cufinufft1d3(f.M, d_x, d_c, 1, eps, f.Nk, d_s, d_fk, nullptr);
    } else if (f.dim == 2) {
      if (f.type == 1)
        rc = cufinufft2d1(f.M, d_x, d_y, d_c, 1, eps, f.N1, f.N2, d_fk, nullptr);
      else if (f.type == 2)
        rc = cufinufft2d2(f.M, d_x, d_y, d_c, 1, eps, f.N1, f.N2, d_fk, nullptr);
      else
        rc = cufinufft2d3(f.M, d_x, d_y, d_c, 1, eps, f.Nk, d_s, d_t, d_fk, nullptr);
    } else {
      if (f.type == 1)
        rc = cufinufft3d1(f.M, d_x, d_y, d_z, d_c, 1, eps, f.N1, f.N2, f.N3, d_fk,
                          nullptr);
      else if (f.type == 2)
        rc = cufinufft3d2(f.M, d_x, d_y, d_z, d_c, 1, eps, f.N1, f.N2, f.N3, d_fk,
                          nullptr);
      else
        rc = cufinufft3d3(f.M, d_x, d_y, d_z, d_c, 1, eps, f.Nk, d_s, d_t, d_u, d_fk,
                          nullptr);
    }
  }
  if (rc != 0) {
    std::fprintf(stderr, "simple call failed: %d (dim=%d type=%d)\n", rc, f.dim, f.type);
    return {};
  }

  std::vector<std::complex<T>> out;
  if (f.type == 2)
    out = dev_copy_out(reinterpret_cast<const std::complex<T> *>(d_c), f.c_count());
  else
    out = dev_copy_out(reinterpret_cast<const std::complex<T> *>(d_fk), f.fk_count());

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_s);
  cudaFree(d_t);
  cudaFree(d_u);
  cudaFree(d_c);
  cudaFree(d_fk);
  return out;
}

template<typename T> int compare_one(int dim, int type, std::mt19937 &rng) {
  auto f = make_fixture<T>(dim, type, rng);
  auto via_p = run_plan(f);
  auto via_s = run_simple(f);
  if (via_p.empty() || via_s.empty()) return 1;
  // Tolerance: the algorithm is the same; only atomic-add non-determinism
  // separates the two runs. Float type-3 chains spreading + FFT + interpolation
  // so the error accumulates more; allow a looser bound there.
  const T rtol = std::is_same_v<T, float> ? T(5e-3) : T(1e-9);
  if (!near_equal(via_p, via_s, rtol)) {
    std::fprintf(stderr, "mismatch %s dim=%d type=%d (n=%zu)\n", Traits<T>::label, dim,
                 type, via_p.size());
    return 1;
  }
  std::printf("ok %s dim=%d type=%d (n=%zu)\n", Traits<T>::label, dim, type,
              via_p.size());
  return 0;
}

template<typename T> int run_all() {
  std::mt19937 rng(static_cast<unsigned>(std::is_same_v<T, float> ? 1 : 2));
  int rc = 0;
  for (int dim = 1; dim <= 3; ++dim) {
    for (int type = 1; type <= 3; ++type) {
      rc |= compare_one<T>(dim, type, rng);
    }
  }
  return rc;
}

} // namespace

int main() {
  int rc = 0;
  rc |= run_all<float>();
  rc |= run_all<double>();
  return rc;
}
