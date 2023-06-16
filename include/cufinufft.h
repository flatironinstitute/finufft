// Defines the C++/C user interface to CUFINUFFT library.
#include <cufft.h>

#include <cufinufft_errors.h>
#include <cufinufft_opts.h>

typedef struct cufinufft_plan_s *cufinufft_plan;
typedef struct cufinufft_fplan_s *cufinufftf_plan;

#ifdef __cplusplus
extern "C" {
#endif
int cufinufft_default_opts(int type, int dim, cufinufft_opts *opts);

int cufinufft_makeplan(int type, int dim, int *n_modes, int iflag, int ntransf, double tol, int maxbatchsize,
                       cufinufft_plan *d_plan_ptr, cufinufft_opts *opts);
int cufinufft_setpts(int M, double *h_kx, double *h_ky, double *h_kz, int N, double *h_s, double *h_t, double *h_u,
                     cufinufft_plan d_plan);
int cufinufft_execute(cuDoubleComplex *h_c, cuDoubleComplex *h_fk, cufinufft_plan d_plan);
int cufinufft_destroy(cufinufft_plan d_plan);

int cufinufftf_makeplan(int type, int dim, int *n_modes, int iflag, int ntransf, float tol, int maxbatchsize,
                        cufinufftf_plan *d_plan_ptr, cufinufft_opts *opts);
int cufinufftf_setpts(int M, float *h_kx, float *h_ky, float *h_kz, int N, float *h_s, float *h_t, float *h_u,
                      cufinufftf_plan d_plan);
int cufinufftf_execute(cuFloatComplex *h_c, cuFloatComplex *h_fk, cufinufftf_plan d_plan);
int cufinufftf_destroy(cufinufftf_plan d_plan);
#ifdef __cplusplus
}
#endif
