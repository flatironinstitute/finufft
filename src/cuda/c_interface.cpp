#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

#include <cufinufft.h>
#include <cufinufft/cufinufft_plan_t.hpp>
#include <finufft_common/safe_call.h>

using finufft::common::safe_finufft_call;

static inline bool is_invalid_mode_array(int type, int dim, const int64_t *modes64,
                                         int32_t modes32[3]) {
  if (type == 3) {
    modes32[0] = modes32[1] = modes32[2] = 1;
    return false;
  }

  int64_t tot_size = 1;
  for (int i = 0; i < dim; ++i) {
    if (modes64[i] > std::numeric_limits<int32_t>::max()) return true;
    if (modes64[i] <= 0) return true;
    modes32[i] = modes64[i];
    tot_size *= modes64[i];
  }
  for (int i = dim; i < 3; ++i) modes32[i] = 1;

  return tot_size > std::numeric_limits<int32_t>::max();
}

extern "C" {
int cufinufftf_makeplan(int type, int dim, const int64_t *nmodes, int iflag, int ntransf,
                        float tol, cufinufftf_plan *d_plan_ptr,
                        const cufinufft_opts *opts) {
  return safe_finufft_call([&]() -> int {
    if (dim < 1 || dim > 3) {
      fprintf(stderr, "[%s] Invalid dim (%d), should be 1, 2 or 3.\n", __func__, dim);
      throw finufft::exception(FINUFFT_ERR_DIM_NOTVALID);
    }

    int nmodes32[3];
    if (is_invalid_mode_array(type, dim, nmodes, nmodes32))
      throw finufft::exception(FINUFFT_ERR_NDATA_NOTVALID);

    cufinufft_opts planopts;
    if (opts)
      planopts = *opts;
    else
      cufinufft_default_opts(&planopts);

    auto res =
        new cufinufft_plan_t<float>(type, dim, nmodes32, iflag, ntransf, tol, planopts);
    *d_plan_ptr = (cufinufftf_plan)res;
    return res->eps_too_small ? FINUFFT_WARN_EPS_TOO_SMALL : 0;
  });
}

int cufinufft_makeplan(int type, int dim, const int64_t *nmodes, int iflag, int ntransf,
                       double tol, cufinufft_plan *d_plan_ptr,
                       const cufinufft_opts *opts) {
  return safe_finufft_call([&]() -> int {
    if (dim < 1 || dim > 3) {
      fprintf(stderr, "[%s] Invalid dim (%d), should be 1, 2 or 3.\n", __func__, dim);
      throw finufft::exception(FINUFFT_ERR_DIM_NOTVALID);
    }

    int nmodes32[3];
    if (is_invalid_mode_array(type, dim, nmodes, nmodes32))
      throw finufft::exception(FINUFFT_ERR_NDATA_NOTVALID);

    cufinufft_opts planopts;
    if (opts)
      planopts = *opts;
    else
      cufinufft_default_opts(&planopts);

    auto res =
        new cufinufft_plan_t<double>(type, dim, nmodes32, iflag, ntransf, tol, planopts);
    *d_plan_ptr = (cufinufft_plan)res;
    return res->eps_too_small ? FINUFFT_WARN_EPS_TOO_SMALL : 0;
  });
}

int cufinufftf_setpts(cufinufftf_plan d_plan, const int64_t M, const float *d_x,
                      const float *d_y, const float *d_z, int N, const float *d_s,
                      const float *d_t, const float *d_u) {
  return safe_finufft_call([&]() {
    if (M > std::numeric_limits<int32_t>::max())
      throw finufft::exception(FINUFFT_ERR_NDATA_NOTVALID);

    ((cufinufft_plan_t<float> *)d_plan)->setpts((int)M, d_x, d_y, d_z, N, d_s, d_t, d_u);
  });
}

int cufinufft_setpts(cufinufft_plan d_plan, const int64_t M, const double *d_x,
                     const double *d_y, const double *d_z, int N, const double *d_s,
                     const double *d_t, const double *d_u) {
  return safe_finufft_call([&]() {
    if (M > std::numeric_limits<int32_t>::max())
      throw finufft::exception(FINUFFT_ERR_NDATA_NOTVALID);

    ((cufinufft_plan_t<double> *)d_plan)->setpts((int)M, d_x, d_y, d_z, N, d_s, d_t, d_u);
  });
}

int cufinufftf_execute(cufinufftf_plan d_plan, cuFloatComplex *d_c,
                       cuFloatComplex *d_fk) {
  return safe_finufft_call(
      [&]() { ((cufinufft_plan_t<float> *)d_plan)->execute(d_c, d_fk); });
}

int cufinufft_execute(cufinufft_plan d_plan, cuDoubleComplex *d_c,
                      cuda_complex<double> *d_fk) {
  return safe_finufft_call(
      [&]() { ((cufinufft_plan_t<double> *)d_plan)->execute(d_c, d_fk); });
}

int cufinufftf_destroy(cufinufftf_plan d_plan) {
  return safe_finufft_call([&]() {
    if (!d_plan) throw finufft::exception(FINUFFT_ERR_PLAN_NOTVALID);
    delete ((cufinufft_plan_t<float> *)d_plan);
  });
}

int cufinufft_destroy(cufinufft_plan d_plan) {
  return safe_finufft_call([&]() {
    if (!d_plan) throw finufft::exception(FINUFFT_ERR_PLAN_NOTVALID);
    delete ((cufinufft_plan_t<double> *)d_plan);
  });
}

void cufinufft_default_opts(cufinufft_opts *opts)
/*
    Sets the default options in cufinufft_opts. This must be called
    before the user changes any options from default values.
    The resulting struct may then be passed (instead of NULL) to the last
    argument of cufinufft_plan().

    Notes:
    1) Values set in this function for different type and dimensions are preferable
    based on experiments. User can experiment with different settings by
    changing them after calling this function.
    2) Sphinx sucks the below code block into the web docs, hence keep it clean.

    Melody Shih 07/25/19; Barnett 2/5/21, tidied for sphinx 7/2/24.
    Barbone Jan/29/26: tweaked np default to 32. Increases performance by 15-21%.
*/
{
  // sphinx tag (don't remove): @gpu_defopts_start
  // data handling opts...
  opts->modeord              = 0;
  opts->gpu_device_id        = 0;
  opts->gpu_spreadinterponly = 0;

  // algorithm performance opts...
  opts->gpu_method         = 0;
  opts->gpu_sort           = 1;
  opts->gpu_kerevalmeth    = 1;
  opts->upsampfac          = 0.0;
  opts->gpu_maxsubprobsize = 1024;
  opts->gpu_obinsizex      = 0;
  opts->gpu_obinsizey      = 0;
  opts->gpu_obinsizez      = 0;
  opts->gpu_binsizex       = 0;
  opts->gpu_binsizey       = 0;
  opts->gpu_binsizez       = 0;
  opts->gpu_maxbatchsize   = 0;
  opts->gpu_np             = 0;
  opts->debug              = 0;
  opts->gpu_stream         = cudaStreamDefault;
  // sphinx tag (don't remove): @gpu_defopts_end
}
} // extern "C"

/* ---------------------------------------------------------------------------
   The 18 simple interfaces (= 3 dims * 3 types * {singlecall,many}) to
   cuFINUFFT, mirroring the CPU c_interface.cpp layout. Pointers are device
   pointers; behavior mirrors the 4-step plan API above.
   --------------------------------------------------------------------------- */

namespace { // helpers local to this TU

using i64 = int64_t;
using ci64 = const int64_t;

template<typename T>
int simple_guru(
    int n_dims, int type, int n_transf, i64 nj, const std::array<const T *, 3> &xyz,
    cuda_complex<T> *cj, int iflag, T eps, const std::array<ci64, 3> &n_modes, i64 nk,
    const std::array<const T *, 3> &stu, cuda_complex<T> *fk, const cufinufft_opts *popts)
// Helper layer between simple interfaces and the underlying plan methods.
// Plan is heap-allocated (it owns CUDA streams / cuFFT plans / device buffers
// whose destruction order matters) and managed via unique_ptr so error paths
// still free GPU resources.
{
  if (n_dims < 1 || n_dims > 3) return FINUFFT_ERR_DIM_NOTVALID;
  if (nj > std::numeric_limits<int32_t>::max()) return FINUFFT_ERR_NDATA_NOTVALID;
  if (nk > std::numeric_limits<int32_t>::max()) return FINUFFT_ERR_NDATA_NOTVALID;

  int nmodes32[3];
  if (is_invalid_mode_array(type, n_dims, n_modes.data(), nmodes32))
    return FINUFFT_ERR_NDATA_NOTVALID;

  cufinufft_opts planopts;
  if (popts)
    planopts = *popts;
  else
    cufinufft_default_opts(&planopts);

  auto plan = std::make_unique<cufinufft_plan_t<T>>(type, n_dims, nmodes32, iflag,
                                                    n_transf, eps, planopts);
  const int warn = plan->eps_too_small ? FINUFFT_WARN_EPS_TOO_SMALL : 0;
  plan->setpts((int)nj, xyz[0], xyz[1], xyz[2], (int)nk, stu[0], stu[1], stu[2]);
  plan->execute(cj, fk);
  return warn;
}

template<typename T>
int guru13(int n_dims, int type, int n_transf, i64 nj,
           const std::array<const T *, 3> &xyz, const cuda_complex<T> *cj, int iflag,
           T eps, const std::array<ci64, 3> &n_modes, i64 nk,
           const std::array<const T *, 3> &stu, cuda_complex<T> *fk,
           const cufinufft_opts *popts) {
  return safe_finufft_call([&]() -> int {
    return simple_guru<T>(n_dims, type, n_transf, nj, xyz,
                          const_cast<cuda_complex<T> *>(cj), iflag, eps, n_modes, nk, stu,
                          fk, popts);
  });
}

template<typename T>
int guru2(int n_dims, int type, int n_transf, i64 nj, const std::array<const T *, 3> &xyz,
          cuda_complex<T> *cj, int iflag, T eps, const std::array<ci64, 3> &n_modes,
          i64 nk, const std::array<const T *, 3> &stu, const cuda_complex<T> *fk,
          const cufinufft_opts *popts) {
  return safe_finufft_call([&]() -> int {
    return simple_guru<T>(n_dims, type, n_transf, nj, xyz, cj, iflag, eps, n_modes, nk,
                          stu, const_cast<cuda_complex<T> *>(fk), popts);
  });
}

} // anonymous namespace

extern "C" {

// Dimension 1111111111111111111111111111111111111111111111111111111111111111

int cufinufft1d1many(int n_transf, i64 nj, const double *xj, const cuDoubleComplex *cj,
                     int iflag, double eps, i64 ms, cuDoubleComplex *fk,
                     const cufinufft_opts *opts) {
  return guru13<double>(1, 1, n_transf, nj, {xj}, cj, iflag, eps, {ms, 1, 1}, 0, {}, fk,
                        opts);
}
int cufinufftf1d1many(int n_transf, i64 nj, const float *xj, const cuFloatComplex *cj,
                      int iflag, float eps, i64 ms, cuFloatComplex *fk,
                      const cufinufft_opts *opts) {
  return guru13<float>(1, 1, n_transf, nj, {xj}, cj, iflag, eps, {ms, 1, 1}, 0, {}, fk,
                       opts);
}
int cufinufft1d1(i64 nj, const double *xj, const cuDoubleComplex *cj, int iflag,
                 double eps, i64 ms, cuDoubleComplex *fk, const cufinufft_opts *opts) {
  return cufinufft1d1many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}
int cufinufftf1d1(i64 nj, const float *xj, const cuFloatComplex *cj, int iflag, float eps,
                  i64 ms, cuFloatComplex *fk, const cufinufft_opts *opts) {
  return cufinufftf1d1many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}

int cufinufft1d2many(int n_transf, i64 nj, const double *xj, cuDoubleComplex *cj,
                     int iflag, double eps, i64 ms, const cuDoubleComplex *fk,
                     const cufinufft_opts *opts) {
  return guru2<double>(1, 2, n_transf, nj, {xj}, cj, iflag, eps, {ms, 1, 1}, 0, {}, fk,
                       opts);
}
int cufinufftf1d2many(int n_transf, i64 nj, const float *xj, cuFloatComplex *cj,
                      int iflag, float eps, i64 ms, const cuFloatComplex *fk,
                      const cufinufft_opts *opts) {
  return guru2<float>(1, 2, n_transf, nj, {xj}, cj, iflag, eps, {ms, 1, 1}, 0, {}, fk,
                      opts);
}
int cufinufft1d2(i64 nj, const double *xj, cuDoubleComplex *cj, int iflag, double eps,
                 i64 ms, const cuDoubleComplex *fk, const cufinufft_opts *opts) {
  return cufinufft1d2many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}
int cufinufftf1d2(i64 nj, const float *xj, cuFloatComplex *cj, int iflag, float eps,
                  i64 ms, const cuFloatComplex *fk, const cufinufft_opts *opts) {
  return cufinufftf1d2many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}

int cufinufft1d3many(int n_transf, i64 nj, const double *xj, const cuDoubleComplex *cj,
                     int iflag, double eps, i64 nk, const double *s, cuDoubleComplex *fk,
                     const cufinufft_opts *opts) {
  return guru13<double>(1, 3, n_transf, nj, {xj}, cj, iflag, eps, {0, 0, 0}, nk, {s}, fk,
                        opts);
}
int cufinufftf1d3many(int n_transf, i64 nj, const float *xj, const cuFloatComplex *cj,
                      int iflag, float eps, i64 nk, const float *s, cuFloatComplex *fk,
                      const cufinufft_opts *opts) {
  return guru13<float>(1, 3, n_transf, nj, {xj}, cj, iflag, eps, {0, 0, 0}, nk, {s}, fk,
                       opts);
}
int cufinufft1d3(i64 nj, const double *xj, const cuDoubleComplex *cj, int iflag,
                 double eps, i64 nk, const double *s, cuDoubleComplex *fk,
                 const cufinufft_opts *opts) {
  return cufinufft1d3many(1, nj, xj, cj, iflag, eps, nk, s, fk, opts);
}
int cufinufftf1d3(i64 nj, const float *xj, const cuFloatComplex *cj, int iflag, float eps,
                  i64 nk, const float *s, cuFloatComplex *fk,
                  const cufinufft_opts *opts) {
  return cufinufftf1d3many(1, nj, xj, cj, iflag, eps, nk, s, fk, opts);
}

// Dimension 22222222222222222222222222222222222222222222222222222222222222222

int cufinufft2d1many(int n_transf, i64 nj, const double *xj, const double *yj,
                     const cuDoubleComplex *cj, int iflag, double eps, i64 ms, i64 mt,
                     cuDoubleComplex *fk, const cufinufft_opts *opts) {
  return guru13<double>(2, 1, n_transf, nj, {xj, yj}, cj, iflag, eps, {ms, mt, 1}, 0, {},
                        fk, opts);
}
int cufinufftf2d1many(int n_transf, i64 nj, const float *xj, const float *yj,
                      const cuFloatComplex *cj, int iflag, float eps, i64 ms, i64 mt,
                      cuFloatComplex *fk, const cufinufft_opts *opts) {
  return guru13<float>(2, 1, n_transf, nj, {xj, yj}, cj, iflag, eps, {ms, mt, 1}, 0, {},
                       fk, opts);
}
int cufinufft2d1(i64 nj, const double *xj, const double *yj, const cuDoubleComplex *cj,
                 int iflag, double eps, i64 ms, i64 mt, cuDoubleComplex *fk,
                 const cufinufft_opts *opts) {
  return cufinufft2d1many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}
int cufinufftf2d1(i64 nj, const float *xj, const float *yj, const cuFloatComplex *cj,
                  int iflag, float eps, i64 ms, i64 mt, cuFloatComplex *fk,
                  const cufinufft_opts *opts) {
  return cufinufftf2d1many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}

int cufinufft2d2many(int n_transf, i64 nj, const double *xj, const double *yj,
                     cuDoubleComplex *cj, int iflag, double eps, i64 ms, i64 mt,
                     const cuDoubleComplex *fk, const cufinufft_opts *opts) {
  return guru2<double>(2, 2, n_transf, nj, {xj, yj}, cj, iflag, eps, {ms, mt, 1}, 0, {},
                       fk, opts);
}
int cufinufftf2d2many(int n_transf, i64 nj, const float *xj, const float *yj,
                      cuFloatComplex *cj, int iflag, float eps, i64 ms, i64 mt,
                      const cuFloatComplex *fk, const cufinufft_opts *opts) {
  return guru2<float>(2, 2, n_transf, nj, {xj, yj}, cj, iflag, eps, {ms, mt, 1}, 0, {},
                      fk, opts);
}
int cufinufft2d2(i64 nj, const double *xj, const double *yj, cuDoubleComplex *cj,
                 int iflag, double eps, i64 ms, i64 mt, const cuDoubleComplex *fk,
                 const cufinufft_opts *opts) {
  return cufinufft2d2many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}
int cufinufftf2d2(i64 nj, const float *xj, const float *yj, cuFloatComplex *cj, int iflag,
                  float eps, i64 ms, i64 mt, const cuFloatComplex *fk,
                  const cufinufft_opts *opts) {
  return cufinufftf2d2many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}

int cufinufft2d3many(int n_transf, i64 nj, const double *xj, const double *yj,
                     const cuDoubleComplex *cj, int iflag, double eps, i64 nk,
                     const double *s, const double *t, cuDoubleComplex *fk,
                     const cufinufft_opts *opts) {
  return guru13<double>(2, 3, n_transf, nj, {xj, yj}, cj, iflag, eps, {0, 0, 0}, nk,
                        {s, t}, fk, opts);
}
int cufinufftf2d3many(int n_transf, i64 nj, const float *xj, const float *yj,
                      const cuFloatComplex *cj, int iflag, float eps, i64 nk,
                      const float *s, const float *t, cuFloatComplex *fk,
                      const cufinufft_opts *opts) {
  return guru13<float>(2, 3, n_transf, nj, {xj, yj}, cj, iflag, eps, {0, 0, 0}, nk,
                       {s, t}, fk, opts);
}
int cufinufft2d3(i64 nj, const double *xj, const double *yj, const cuDoubleComplex *cj,
                 int iflag, double eps, i64 nk, const double *s, const double *t,
                 cuDoubleComplex *fk, const cufinufft_opts *opts) {
  return cufinufft2d3many(1, nj, xj, yj, cj, iflag, eps, nk, s, t, fk, opts);
}
int cufinufftf2d3(i64 nj, const float *xj, const float *yj, const cuFloatComplex *cj,
                  int iflag, float eps, i64 nk, const float *s, const float *t,
                  cuFloatComplex *fk, const cufinufft_opts *opts) {
  return cufinufftf2d3many(1, nj, xj, yj, cj, iflag, eps, nk, s, t, fk, opts);
}

// Dimension 3333333333333333333333333333333333333333333333333333333333333333

int cufinufft3d1many(int n_transf, i64 nj, const double *xj, const double *yj,
                     const double *zj, const cuDoubleComplex *cj, int iflag, double eps,
                     i64 ms, i64 mt, i64 mu, cuDoubleComplex *fk,
                     const cufinufft_opts *opts) {
  return guru13<double>(3, 1, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {ms, mt, mu}, 0,
                        {}, fk, opts);
}
int cufinufftf3d1many(int n_transf, i64 nj, const float *xj, const float *yj,
                      const float *zj, const cuFloatComplex *cj, int iflag, float eps,
                      i64 ms, i64 mt, i64 mu, cuFloatComplex *fk,
                      const cufinufft_opts *opts) {
  return guru13<float>(3, 1, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {ms, mt, mu}, 0,
                       {}, fk, opts);
}
int cufinufft3d1(i64 nj, const double *xj, const double *yj, const double *zj,
                 const cuDoubleComplex *cj, int iflag, double eps, i64 ms, i64 mt, i64 mu,
                 cuDoubleComplex *fk, const cufinufft_opts *opts) {
  return cufinufft3d1many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}
int cufinufftf3d1(i64 nj, const float *xj, const float *yj, const float *zj,
                  const cuFloatComplex *cj, int iflag, float eps, i64 ms, i64 mt, i64 mu,
                  cuFloatComplex *fk, const cufinufft_opts *opts) {
  return cufinufftf3d1many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}

int cufinufft3d2many(int n_transf, i64 nj, const double *xj, const double *yj,
                     const double *zj, cuDoubleComplex *cj, int iflag, double eps, i64 ms,
                     i64 mt, i64 mu, const cuDoubleComplex *fk,
                     const cufinufft_opts *opts) {
  return guru2<double>(3, 2, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {ms, mt, mu}, 0,
                       {}, fk, opts);
}
int cufinufftf3d2many(int n_transf, i64 nj, const float *xj, const float *yj,
                      const float *zj, cuFloatComplex *cj, int iflag, float eps, i64 ms,
                      i64 mt, i64 mu, const cuFloatComplex *fk,
                      const cufinufft_opts *opts) {
  return guru2<float>(3, 2, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {ms, mt, mu}, 0,
                      {}, fk, opts);
}
int cufinufft3d2(i64 nj, const double *xj, const double *yj, const double *zj,
                 cuDoubleComplex *cj, int iflag, double eps, i64 ms, i64 mt, i64 mu,
                 const cuDoubleComplex *fk, const cufinufft_opts *opts) {
  return cufinufft3d2many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}
int cufinufftf3d2(i64 nj, const float *xj, const float *yj, const float *zj,
                  cuFloatComplex *cj, int iflag, float eps, i64 ms, i64 mt, i64 mu,
                  const cuFloatComplex *fk, const cufinufft_opts *opts) {
  return cufinufftf3d2many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}

int cufinufft3d3many(int n_transf, i64 nj, const double *xj, const double *yj,
                     const double *zj, const cuDoubleComplex *cj, int iflag, double eps,
                     i64 nk, const double *s, const double *t, const double *u,
                     cuDoubleComplex *fk, const cufinufft_opts *opts) {
  return guru13<double>(3, 3, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {0, 0, 0}, nk,
                        {s, t, u}, fk, opts);
}
int cufinufftf3d3many(int n_transf, i64 nj, const float *xj, const float *yj,
                      const float *zj, const cuFloatComplex *cj, int iflag, float eps,
                      i64 nk, const float *s, const float *t, const float *u,
                      cuFloatComplex *fk, const cufinufft_opts *opts) {
  return guru13<float>(3, 3, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {0, 0, 0}, nk,
                       {s, t, u}, fk, opts);
}
int cufinufft3d3(i64 nj, const double *xj, const double *yj, const double *zj,
                 const cuDoubleComplex *cj, int iflag, double eps, i64 nk,
                 const double *s, const double *t, const double *u, cuDoubleComplex *fk,
                 const cufinufft_opts *opts) {
  return cufinufft3d3many(1, nj, xj, yj, zj, cj, iflag, eps, nk, s, t, u, fk, opts);
}
int cufinufftf3d3(i64 nj, const float *xj, const float *yj, const float *zj,
                  const cuFloatComplex *cj, int iflag, float eps, i64 nk, const float *s,
                  const float *t, const float *u, cuFloatComplex *fk,
                  const cufinufft_opts *opts) {
  return cufinufftf3d3many(1, nj, xj, yj, zj, cj, iflag, eps, nk, s, t, u, fk, opts);
}

} // extern "C"
