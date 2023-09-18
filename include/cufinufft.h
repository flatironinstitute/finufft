// Defines the C++/C user interface to CUFINUFFT library.
#include <cufft.h>

#include <cufinufft_opts.h>

typedef struct cufinufft_plan_s *cufinufft_plan;
typedef struct cufinufft_fplan_s *cufinufftf_plan;

#ifdef __cplusplus
extern "C" {
#endif
void cufinufft_default_opts(cufinufft_opts *opts);

int cufinufft_makeplan(int type, int dim, int64_t *n_modes, int iflag, int ntr, double eps, cufinufft_plan *d_plan_ptr,
                       cufinufft_opts *opts);
int cufinufftf_makeplan(int type, int dim, int64_t *n_modes, int iflag, int ntr, float eps, cufinufftf_plan *d_plan_ptr,
                        cufinufft_opts *opts);

int cufinufft_setpts(cufinufft_plan d_plan, int M, double *h_kx, double *h_ky, double *h_kz, int N, double *h_s,
                     double *h_t, double *h_u);
int cufinufftf_setpts(cufinufftf_plan d_plan, int M, float *h_kx, float *h_ky, float *h_kz, int N, float *h_s,
                      float *h_t, float *h_u);

int cufinufft_execute(cufinufft_plan d_plan, cuDoubleComplex *h_c, cuDoubleComplex *h_fk);
int cufinufftf_execute(cufinufftf_plan d_plan, cuFloatComplex *h_c, cuFloatComplex *h_fk);

int cufinufft_destroy(cufinufft_plan d_plan);
int cufinufftf_destroy(cufinufftf_plan d_plan);
#ifdef __cplusplus
}
#endif
