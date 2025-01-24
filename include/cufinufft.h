// Defines the C++/C user interface to CUFINUFFT library.
#include <cufft.h>

#include <stdint.h>

#include <cufinufft_opts.h>
#include <finufft_errors.h>

typedef struct cufinufft_plan_s *cufinufft_plan;
typedef struct cufinufft_fplan_s *cufinufftf_plan;

#ifdef __cplusplus
extern "C" {
#endif
void cufinufft_default_opts(cufinufft_opts *opts);

int cufinufft_makeplan(int type, int dim, const int64_t *n_modes, int iflag, int ntr,
                       double eps, cufinufft_plan *d_plan_ptr, cufinufft_opts *opts);
int cufinufftf_makeplan(int type, int dim, const int64_t *n_modes, int iflag, int ntr,
                        float eps, cufinufftf_plan *d_plan_ptr, cufinufft_opts *opts);

int cufinufft_setpts(cufinufft_plan d_plan, int64_t M, double *d_x, double *d_y,
                     double *d_z, int N, double *d_s, double *d_t, double *d_u);
int cufinufftf_setpts(cufinufftf_plan d_plan, int64_t M, float *d_x, float *d_y,
                      float *d_z, int N, float *d_s, float *d_t, float *d_u);

int cufinufft_execute(cufinufft_plan d_plan, cuDoubleComplex *d_c, cuDoubleComplex *d_fk);
int cufinufftf_execute(cufinufftf_plan d_plan, cuFloatComplex *d_c, cuFloatComplex *d_fk);

int cufinufft_destroy(cufinufft_plan d_plan);
int cufinufftf_destroy(cufinufftf_plan d_plan);
#ifdef __cplusplus
}
#endif
