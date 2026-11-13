// Simple-API dumb-input test for cuFINUFFT (issue #947), double precision only.
// Empty point and mode sets must behave like the CPU library: a no-op or a
// zero-filled output. Bad point counts must give FINUFFT_ERR_NUM_NU_PTS_INVALID.
// Each output is pre-poisoned so a missing zero-fill shows up. Barbone 11/26

#include <cstdint>
#include <cstdio>

#include <cufinufft.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

namespace {

int nfail = 0;

void check_ier(const char *name, int ier, int want = 0) {
  if (ier != want) {
    std::printf("%s: ier=%d (want %d)\n", name, ier, want);
    ++nfail;
  }
}

void check_zero(const char *name, const thrust::device_vector<cuDoubleComplex> &v,
                int64_t n) {
  const thrust::host_vector<cuDoubleComplex> h(v.begin(), v.begin() + n);
  for (const auto &z : h) {
    if (z.x != 0.0 || z.y != 0.0) {
      std::printf("%s: nonzero output\n", name);
      ++nfail;
      return;
    }
  }
}

} // namespace

int main() {
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  const int iflag           = 1;
  const double eps          = 1e-6;
  const cuDoubleComplex pzn = {1e30, -1e30};

  const int64_t M = 4, ms = 5, nk = 3;
  thrust::device_vector<double> x(M, 0.1), y(M, 0.2), s(nk, 0.5);
  thrust::device_vector<cuDoubleComplex> c(M), f(ms);
  double *xp = x.data().get(), *yp = y.data().get(), *sp = s.data().get();
  auto *cp = c.data().get();
  auto *fp = f.data().get();

  // type 1 with no points zero-fills f
  thrust::fill(f.begin(), f.end(), pzn);
  check_ier("t1 M=0", cufinufft1d1(0, xp, cp, iflag, eps, ms, fp, &opts));
  check_zero("t1 M=0", f, ms);

  // type 1 with no modes, and type 2 with no points, are no-ops
  check_ier("t1 N1=0", cufinufft1d1(M, xp, cp, iflag, eps, 0, fp, &opts));
  check_ier("t2 M=0", cufinufft1d2(0, xp, cp, iflag, eps, ms, fp, &opts));

  // type 2 with no modes zero-fills c
  thrust::fill(c.begin(), c.end(), pzn);
  check_ier("t2 N1=0", cufinufft1d2(M, xp, cp, iflag, eps, 0, fp, &opts));
  check_zero("t2 N1=0", c, M);

  // type 3 with no source points zero-fills f; with no targets it is a no-op
  thrust::fill(f.begin(), f.begin() + nk, pzn);
  check_ier("t3 nj=0", cufinufft1d3(0, xp, cp, iflag, eps, nk, sp, fp, &opts));
  check_zero("t3 nj=0", f, nk);
  check_ier("t3 nk=0", cufinufft1d3(M, xp, cp, iflag, eps, 0, sp, fp, &opts));

  // a zero mode count in a non-leading dimension (2D type 2) zero-fills c
  thrust::fill(c.begin(), c.end(), pzn);
  check_ier("t2d2 mt=0", cufinufft2d2(M, xp, yp, cp, iflag, eps, ms, 0, fp, &opts));
  check_zero("t2d2 mt=0", c, M);

  // spreadinterponly with an empty mode set zero-fills c
  {
    cufinufft_opts o       = opts;
    o.gpu_spreadinterponly = 1;
    thrust::fill(c.begin(), c.end(), pzn);
    check_ier("t2 N1=0 io", cufinufft1d2(M, xp, cp, iflag, eps, 0, fp, &o));
    check_zero("t2 N1=0 io", c, M);
  }

  // bad point counts give FINUFFT_ERR_NUM_NU_PTS_INVALID, as on the CPU
  check_ier("t1 M<0", cufinufft1d1(-1, xp, cp, iflag, eps, ms, fp, &opts),
            FINUFFT_ERR_NUM_NU_PTS_INVALID);
  check_ier("t1 M huge", cufinufft1d1(int64_t(1e16), xp, cp, iflag, eps, ms, fp, &opts),
            FINUFFT_ERR_NUM_NU_PTS_INVALID);

  std::printf("cufinufft_dumbinputs %s\n", nfail ? "FAILED" : "passed");
  return nfail ? 1 : 0;
}
