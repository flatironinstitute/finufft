#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include <cufinufft.h>
#include <cufinufft/impl.h>

inline bool is_invalid_mode_array(int dim, const int64_t *modes64, int32_t modes32[3]) {
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
                        float tol, cufinufftf_plan *d_plan_ptr, cufinufft_opts *opts) {
  if (dim < 1 || dim > 3) {
    fprintf(stderr, "[%s] Invalid dim (%d), should be 1, 2 or 3.\n", __func__, dim);
    return FINUFFT_ERR_DIM_NOTVALID;
  }

  int nmodes32[3];
  if (is_invalid_mode_array(dim, nmodes, nmodes32)) return FINUFFT_ERR_NDATA_NOTVALID;

  return cufinufft_makeplan_impl(type, dim, nmodes32, iflag, ntransf, tol,
                                 (cufinufft_plan_t<float> **)d_plan_ptr, opts);
}

int cufinufft_makeplan(int type, int dim, const int64_t *nmodes, int iflag, int ntransf,
                       double tol, cufinufft_plan *d_plan_ptr, cufinufft_opts *opts) {
  if (dim < 1 || dim > 3) {
    fprintf(stderr, "[%s] Invalid dim (%d), should be 1, 2 or 3.\n", __func__, dim);
    return FINUFFT_ERR_DIM_NOTVALID;
  }

  int nmodes32[3];
  if (is_invalid_mode_array(dim, nmodes, nmodes32)) return FINUFFT_ERR_NDATA_NOTVALID;

  return cufinufft_makeplan_impl(type, dim, nmodes32, iflag, ntransf, tol,
                                 (cufinufft_plan_t<double> **)d_plan_ptr, opts);
}

int cufinufftf_setpts(cufinufftf_plan d_plan, const int64_t M, float *d_x, float *d_y,
                      float *d_z, int N, float *d_s, float *d_t, float *d_u) {
  if (M > std::numeric_limits<int32_t>::max()) return FINUFFT_ERR_NDATA_NOTVALID;

  return cufinufft_setpts_impl((int)M, d_x, d_y, d_z, N, d_s, d_t, d_u,
                               (cufinufft_plan_t<float> *)d_plan);
}

int cufinufft_setpts(cufinufft_plan d_plan, const int64_t M, double *d_x, double *d_y,
                     double *d_z, int N, double *d_s, double *d_t, double *d_u) {
  if (M > std::numeric_limits<int32_t>::max()) return FINUFFT_ERR_NDATA_NOTVALID;

  return cufinufft_setpts_impl((int)M, d_x, d_y, d_z, N, d_s, d_t, d_u,
                               (cufinufft_plan_t<double> *)d_plan);
}

int cufinufftf_execute(cufinufftf_plan d_plan, cuFloatComplex *d_c,
                       cuFloatComplex *d_fk) {
  return cufinufft_execute_impl<float>(d_c, d_fk, (cufinufft_plan_t<float> *)d_plan);
}

int cufinufft_execute(cufinufft_plan d_plan, cuDoubleComplex *d_c,
                      cuda_complex<double> *d_fk) {
  return cufinufft_execute_impl<double>(d_c, d_fk, (cufinufft_plan_t<double> *)d_plan);
}

int cufinufftf_destroy(cufinufftf_plan d_plan) {
  return cufinufft_destroy_impl<float>((cufinufft_plan_t<float> *)d_plan);
}

int cufinufft_destroy(cufinufft_plan d_plan) {
  return cufinufft_destroy_impl<double>((cufinufft_plan_t<double> *)d_plan);
}

void cufinufft_default_opts(cufinufft_opts *opts)
/*
    Sets the default options in cufinufft_opts. This must be called
    before the user changes any options from default values.
    The resulting struct may then be passed (instead of NULL) to the last
    argument of cufinufft_plan().

    Options with prefix "gpu_" are used for gpu code.

    Notes:
    1) Values set in this function for different type and dimensions are preferable
    based on experiments. User can experiment with different settings by
    changing them after calling this function.
    2) Sphinx sucks the below code block into the web docs, hence keep it clean.

    Melody Shih 07/25/19; Barnett 2/5/21, tidied for sphinx 7/2/24.
*/
{
  // sphinx tag (don't remove): @gpu_defopts_start
  // data handling opts...
  opts->modeord       = 0;
  opts->gpu_device_id = 0;

  // diagnostic opts...
  opts->gpu_spreadinterponly = 0;

  // algorithm performance opts...
  opts->gpu_method         = 0;
  opts->gpu_sort           = 1;
  opts->gpu_kerevalmeth    = 1;
  opts->upsampfac          = 2.0;
  opts->gpu_maxsubprobsize = 1024;
  opts->gpu_obinsizex      = 0;
  opts->gpu_obinsizey      = 0;
  opts->gpu_obinsizez      = 0;
  opts->gpu_binsizex       = 0;
  opts->gpu_binsizey       = 0;
  opts->gpu_binsizez       = 0;
  opts->gpu_maxbatchsize   = 0;
  opts->gpu_stream         = cudaStreamDefault;
  // sphinx tag (don't remove): @gpu_defopts_end
}
}
