// Defines the C++/C user interface to CUFINUFFT library.
#include <cufft.h>

#include <stdint.h>

#include <finufft_common/defines.h>
#include <cufinufft_opts.h>
#include <finufft_errors.h>

typedef struct cufinufft_plan_s *cufinufft_plan;
typedef struct cufinufft_fplan_s *cufinufftf_plan;

#ifdef __cplusplus
extern "C" {
#endif
FINUFFT_EXPORT void cufinufft_default_opts(cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft_makeplan(int type, int dim, const int64_t *n_modes,
                                      int iflag, int ntr, double eps,
                                      cufinufft_plan *d_plan_ptr, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf_makeplan(int type, int dim, const int64_t *n_modes,
                                       int iflag, int ntr, float eps,
                                       cufinufftf_plan *d_plan_ptr, const cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft_setpts(cufinufft_plan d_plan, int64_t M, const double *d_x,
                                    const double *d_y, const double *d_z, int N, const double *d_s,
                                    const double *d_t, const double *d_u);
FINUFFT_EXPORT int cufinufftf_setpts(cufinufftf_plan d_plan, int64_t M, const float *d_x,
                                     const float *d_y, const float *d_z, int N, const float *d_s,
                                     const float *d_t, const float *d_u);

FINUFFT_EXPORT int cufinufft_execute(cufinufft_plan d_plan, cuDoubleComplex *d_c,
                                     cuDoubleComplex *d_fk);
FINUFFT_EXPORT int cufinufftf_execute(cufinufftf_plan d_plan, cuFloatComplex *d_c,
                                      cuFloatComplex *d_fk);

FINUFFT_EXPORT int cufinufft_destroy(cufinufft_plan d_plan);
FINUFFT_EXPORT int cufinufftf_destroy(cufinufftf_plan d_plan);
#ifdef __cplusplus
}
#endif
