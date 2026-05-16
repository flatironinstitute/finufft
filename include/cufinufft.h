// Defines the C++/C user interface to CUFINUFFT library.
#include <cufft.h>

#include <stdint.h>

#include <cufinufft_opts.h>
#include <finufft_common/defines.h>
#include <finufft_errors.h>

typedef struct cufinufft_plan_s *cufinufft_plan;
typedef struct cufinufft_fplan_s *cufinufftf_plan;

#ifdef __cplusplus
extern "C" {
#endif
FINUFFT_EXPORT void cufinufft_default_opts(cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft_makeplan(
    int type, int dim, const int64_t *n_modes, int iflag, int ntr, double eps,
    cufinufft_plan *d_plan_ptr, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf_makeplan(
    int type, int dim, const int64_t *n_modes, int iflag, int ntr, float eps,
    cufinufftf_plan *d_plan_ptr, const cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft_setpts(
    cufinufft_plan d_plan, int64_t M, const double *d_x, const double *d_y,
    const double *d_z, int N, const double *d_s, const double *d_t, const double *d_u);
FINUFFT_EXPORT int cufinufftf_setpts(
    cufinufftf_plan d_plan, int64_t M, const float *d_x, const float *d_y,
    const float *d_z, int N, const float *d_s, const float *d_t, const float *d_u);

FINUFFT_EXPORT int cufinufft_execute(cufinufft_plan d_plan, cuDoubleComplex *d_c,
                                     cuDoubleComplex *d_fk);
FINUFFT_EXPORT int cufinufftf_execute(cufinufftf_plan d_plan, cuFloatComplex *d_c,
                                      cuFloatComplex *d_fk);

FINUFFT_EXPORT int cufinufft_destroy(cufinufft_plan d_plan);
FINUFFT_EXPORT int cufinufftf_destroy(cufinufftf_plan d_plan);

// Simple (one-shot) interfaces. Pointers are device pointers. Behavior matches
// the 4-step plan API above. 36 entry points (3 dims x 3 types x {single,many}
// x {double,float}).

// Dimension 1111111111111111111111111111111111111111111111111111111111111111
FINUFFT_EXPORT int cufinufft1d1many(
    int n_transf, int64_t nj, const double *xj, const cuDoubleComplex *cj, int iflag,
    double eps, int64_t ms, cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf1d1many(
    int n_transf, int64_t nj, const float *xj, const cuFloatComplex *cj, int iflag,
    float eps, int64_t ms, cuFloatComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft1d1(int64_t nj, const double *xj, const cuDoubleComplex *cj,
                                int iflag, double eps, int64_t ms, cuDoubleComplex *fk,
                                const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf1d1(int64_t nj, const float *xj, const cuFloatComplex *cj,
                                 int iflag, float eps, int64_t ms, cuFloatComplex *fk,
                                 const cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft1d2many(
    int n_transf, int64_t nj, const double *xj, cuDoubleComplex *cj, int iflag,
    double eps, int64_t ms, const cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf1d2many(
    int n_transf, int64_t nj, const float *xj, cuFloatComplex *cj, int iflag, float eps,
    int64_t ms, const cuFloatComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft1d2(int64_t nj, const double *xj, cuDoubleComplex *cj,
                                int iflag, double eps, int64_t ms,
                                const cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf1d2(int64_t nj, const float *xj, cuFloatComplex *cj,
                                 int iflag, float eps, int64_t ms,
                                 const cuFloatComplex *fk, const cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft1d3many(int n_transf, int64_t nj, const double *xj,
                                    const cuDoubleComplex *cj, int iflag, double eps,
                                    int64_t nk, const double *s, cuDoubleComplex *fk,
                                    const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf1d3many(int n_transf, int64_t nj, const float *xj,
                                     const cuFloatComplex *cj, int iflag, float eps,
                                     int64_t nk, const float *s, cuFloatComplex *fk,
                                     const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft1d3(int64_t nj, const double *xj, const cuDoubleComplex *cj,
                                int iflag, double eps, int64_t nk, const double *s,
                                cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf1d3(int64_t nj, const float *xj, const cuFloatComplex *cj,
                                 int iflag, float eps, int64_t nk, const float *s,
                                 cuFloatComplex *fk, const cufinufft_opts *opts);

// Dimension 22222222222222222222222222222222222222222222222222222222222222222
FINUFFT_EXPORT int cufinufft2d1many(int n_transf, int64_t nj, const double *xj,
                                    const double *yj, const cuDoubleComplex *cj,
                                    int iflag, double eps, int64_t ms, int64_t mt,
                                    cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf2d1many(int n_transf, int64_t nj, const float *xj,
                                     const float *yj, const cuFloatComplex *cj, int iflag,
                                     float eps, int64_t ms, int64_t mt,
                                     cuFloatComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft2d1(
    int64_t nj, const double *xj, const double *yj, const cuDoubleComplex *cj, int iflag,
    double eps, int64_t ms, int64_t mt, cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf2d1(
    int64_t nj, const float *xj, const float *yj, const cuFloatComplex *cj, int iflag,
    float eps, int64_t ms, int64_t mt, cuFloatComplex *fk, const cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft2d2many(
    int n_transf, int64_t nj, const double *xj, const double *yj, cuDoubleComplex *cj,
    int iflag, double eps, int64_t ms, int64_t mt, const cuDoubleComplex *fk,
    const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf2d2many(
    int n_transf, int64_t nj, const float *xj, const float *yj, cuFloatComplex *cj,
    int iflag, float eps, int64_t ms, int64_t mt, const cuFloatComplex *fk,
    const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft2d2(int64_t nj, const double *xj, const double *yj,
                                cuDoubleComplex *cj, int iflag, double eps, int64_t ms,
                                int64_t mt, const cuDoubleComplex *fk,
                                const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf2d2(int64_t nj, const float *xj, const float *yj,
                                 cuFloatComplex *cj, int iflag, float eps, int64_t ms,
                                 int64_t mt, const cuFloatComplex *fk,
                                 const cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft2d3many(
    int n_transf, int64_t nj, const double *xj, const double *yj,
    const cuDoubleComplex *cj, int iflag, double eps, int64_t nk, const double *s,
    const double *t, cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf2d3many(
    int n_transf, int64_t nj, const float *xj, const float *yj, const cuFloatComplex *cj,
    int iflag, float eps, int64_t nk, const float *s, const float *t, cuFloatComplex *fk,
    const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft2d3(int64_t nj, const double *xj, const double *yj,
                                const cuDoubleComplex *cj, int iflag, double eps,
                                int64_t nk, const double *s, const double *t,
                                cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf2d3(int64_t nj, const float *xj, const float *yj,
                                 const cuFloatComplex *cj, int iflag, float eps,
                                 int64_t nk, const float *s, const float *t,
                                 cuFloatComplex *fk, const cufinufft_opts *opts);

// Dimension 3333333333333333333333333333333333333333333333333333333333333333
FINUFFT_EXPORT int cufinufft3d1many(
    int n_transf, int64_t nj, const double *xj, const double *yj, const double *zj,
    const cuDoubleComplex *cj, int iflag, double eps, int64_t ms, int64_t mt, int64_t mu,
    cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf3d1many(
    int n_transf, int64_t nj, const float *xj, const float *yj, const float *zj,
    const cuFloatComplex *cj, int iflag, float eps, int64_t ms, int64_t mt, int64_t mu,
    cuFloatComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft3d1(int64_t nj, const double *xj, const double *yj,
                                const double *zj, const cuDoubleComplex *cj, int iflag,
                                double eps, int64_t ms, int64_t mt, int64_t mu,
                                cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf3d1(int64_t nj, const float *xj, const float *yj,
                                 const float *zj, const cuFloatComplex *cj, int iflag,
                                 float eps, int64_t ms, int64_t mt, int64_t mu,
                                 cuFloatComplex *fk, const cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft3d2many(
    int n_transf, int64_t nj, const double *xj, const double *yj, const double *zj,
    cuDoubleComplex *cj, int iflag, double eps, int64_t ms, int64_t mt, int64_t mu,
    const cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf3d2many(
    int n_transf, int64_t nj, const float *xj, const float *yj, const float *zj,
    cuFloatComplex *cj, int iflag, float eps, int64_t ms, int64_t mt, int64_t mu,
    const cuFloatComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft3d2(int64_t nj, const double *xj, const double *yj,
                                const double *zj, cuDoubleComplex *cj, int iflag,
                                double eps, int64_t ms, int64_t mt, int64_t mu,
                                const cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf3d2(int64_t nj, const float *xj, const float *yj,
                                 const float *zj, cuFloatComplex *cj, int iflag,
                                 float eps, int64_t ms, int64_t mt, int64_t mu,
                                 const cuFloatComplex *fk, const cufinufft_opts *opts);

FINUFFT_EXPORT int cufinufft3d3many(
    int n_transf, int64_t nj, const double *xj, const double *yj, const double *zj,
    const cuDoubleComplex *cj, int iflag, double eps, int64_t nk, const double *s,
    const double *t, const double *u, cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf3d3many(
    int n_transf, int64_t nj, const float *xj, const float *yj, const float *zj,
    const cuFloatComplex *cj, int iflag, float eps, int64_t nk, const float *s,
    const float *t, const float *u, cuFloatComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufft3d3(
    int64_t nj, const double *xj, const double *yj, const double *zj,
    const cuDoubleComplex *cj, int iflag, double eps, int64_t nk, const double *s,
    const double *t, const double *u, cuDoubleComplex *fk, const cufinufft_opts *opts);
FINUFFT_EXPORT int cufinufftf3d3(
    int64_t nj, const float *xj, const float *yj, const float *zj,
    const cuFloatComplex *cj, int iflag, float eps, int64_t nk, const float *s,
    const float *t, const float *u, cuFloatComplex *fk, const cufinufft_opts *opts);
#ifdef __cplusplus
}
#endif
