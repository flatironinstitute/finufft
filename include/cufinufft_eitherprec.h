// Switchable-precision interface template for cufinufft. Used by cufinufft.h
// Internal use only: users should link to cufinufft.h

#include "cufinufft/types.h"
#if 0 && (!defined(__CUFINUFFT_H__) && !defined(CUFINUFFT_SINGLE)) ||                                                  \
    (!defined(__CUFINUFFTF_H__) && defined(CUFINUFFT_SINGLE))
// (note we entered one level of conditional until the end of this header)
// Make sure we don't include double or single headers more than once...

#include <assert.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>

#include <finufft_spread_opts.h>
#include <cufinufft_errors.h>
#include <cufinufft_opts.h>
#include <cufinufft_types.h>

#ifndef CUFINUFFT_SINGLE
#define __CUFINUFFT_H__
#else
#define __CUFINUFFTF_H__
#endif

/* Undefine things so we don't get warnings/errors later */
#undef CUFINUFFT_DEFAULT_OPTS
#undef CUFINUFFT_MAKEPLAN
#undef CUFINUFFT_SETPTS
#undef CUFINUFFT_EXECUTE
#undef CUFINUFFT_DESTROY
#undef CUFINUFFT1D1_EXEC
#undef CUFINUFFT1D2_EXEC
#undef CUFINUFFT2D1_EXEC
#undef CUFINUFFT2D2_EXEC
#undef CUFINUFFT3D1_EXEC
#undef CUFINUFFT3D2_EXEC
#undef SETUP_BINSIZE
/* memtransfer.h */
#undef ALLOCGPUMEM1D_PLAN
#undef ALLOCGPUMEM1D_NUPTS
#undef FREEGPUMEMORY1D
#undef ALLOCGPUMEM2D_PLAN
#undef ALLOCGPUMEM2D_NUPTS
#undef FREEGPUMEMORY2D
#undef ALLOCGPUMEM3D_PLAN
#undef ALLOCGPUMEM3D_NUPTS
#undef FREEGPUMEMORY3D
/* spreading and interp only*/
#undef CUFINUFFT_SPREAD1D
#undef CUFINUFFT_SPREAD2D
#undef CUFINUFFT_SPREAD3D
#undef CUFINUFFT_INTERP1D
#undef CUFINUFFT_INTERP2D
#undef CUFINUFFT_INTERP3D
/* spreading 1D */
#undef CUSPREAD1D
#undef CUSPREAD1D_NUPTSDRIVEN_PROP
#undef CUSPREAD1D_NUPTSDRIVEN
#undef CUSPREAD1D_SUBPROB_PROP
#undef CUSPREAD1D_SUBPROB
/* spreading 2D */
#undef CUSPREAD2D
#undef CUSPREAD2D_NUPTSDRIVEN_PROP
#undef CUSPREAD2D_NUPTSDRIVEN
#undef CUSPREAD2D_SUBPROB_PROP
#undef CUSPREAD2D_SUBPROB
#undef CUSPREAD2D_PAUL
#undef CUSPREAD2D_PAUL_PROP
/* spreading 3d */
#undef CUSPREAD3D
#undef CUSPREAD3D_NUPTSDRIVEN_PROP
#undef CUSPREAD3D_NUPTSDRIVEN
#undef CUSPREAD3D_BLOCKGATHER_PROP
#undef CUSPREAD3D_BLOCKGATHER
#undef CUSPREAD3D_SUBPROB_PROP
#undef CUSPREAD3D_SUBPROB
/* interp */
#undef CUINTERP1D
#undef CUINTERP2D
#undef CUINTERP3D
#undef CUINTERP1D_NUPTSDRIVEN
#undef CUINTERP2D_NUPTSDRIVEN
#undef CUINTERP2D_SUBPROB
#undef CUINTERP3D_NUPTSDRIVEN
#undef CUINTERP3D_SUBPROB
/* deconvolve */
#undef CUDECONVOLVE1D
#undef CUDECONVOLVE2D
#undef CUDECONVOLVE3D
/* structs */
#undef CUFINUFFT_PLAN_S
#undef CUFINUFFT_PLAN
/* fseries kernel */
#undef CUFSERIESKERNELCOMPUTE

#ifdef CUFINUFFT_SINGLE

#define CUFINUFFT_DEFAULT_OPTS cufinufftf_default_opts
#define CUFINUFFT_MAKEPLAN cufinufftf_makeplan
#define CUFINUFFT_SETPTS cufinufftf_setpts
#define CUFINUFFT_EXECUTE cufinufftf_execute
#define CUFINUFFT_DESTROY cufinufftf_destroy
#define CUFINUFFT1D1_EXEC cufinufftf1d1_exec
#define CUFINUFFT1D2_EXEC cufinufftf1d2_exec
#define CUFINUFFT2D1_EXEC cufinufftf2d1_exec
#define CUFINUFFT2D2_EXEC cufinufftf2d2_exec
#define CUFINUFFT3D1_EXEC cufinufftf3d1_exec
#define CUFINUFFT3D2_EXEC cufinufftf3d2_exec
#define SETUP_BINSIZE setup_binsizef
/* memtransfer.h */
#define ALLOCGPUMEM1D_PLAN allocgpumem1df_plan
#define ALLOCGPUMEM1D_NUPTS allocgpumem1df_nupts
#define FREEGPUMEMORY1D freegpumemory1df
#define ALLOCGPUMEM2D_PLAN allocgpumem2df_plan
#define ALLOCGPUMEM2D_NUPTS allocgpumem2df_nupts
#define FREEGPUMEMORY2D freegpumemory2df
#define ALLOCGPUMEM3D_PLAN allocgpumem3df_plan
#define ALLOCGPUMEM3D_NUPTS allocgpumem3df_nupts
#define FREEGPUMEMORY3D freegpumemory3df
/* spreading and interp only*/
#define CUFINUFFT_SPREAD1D cufinufft_spread1df
#define CUFINUFFT_SPREAD2D cufinufft_spread2df
#define CUFINUFFT_SPREAD3D cufinufft_spread3df
#define CUFINUFFT_INTERP1D cufinufft_interp1df
#define CUFINUFFT_INTERP2D cufinufft_interp2df
#define CUFINUFFT_INTERP3D cufinufft_interp3df
/* spreading 1D */
#define CUSPREAD1D cuspread1df
#define CUSPREAD1D_NUPTSDRIVEN_PROP cuspread1df_nuptsdriven_prop
#define CUSPREAD1D_NUPTSDRIVEN cuspread1df_nuptsdriven
#define CUSPREAD1D_SUBPROB_PROP cuspread1df_subprob_prop
#define CUSPREAD1D_SUBPROB cuspread1df_subprob
/* spreading 2D */
#define CUSPREAD2D cuspread2df
#define CUSPREAD2D_NUPTSDRIVEN_PROP cuspread2df_nuptsdriven_prop
#define CUSPREAD2D_NUPTSDRIVEN cuspread2df_nuptsdriven
#define CUSPREAD2D_SUBPROB_PROP cuspread2df_subprob_prop
#define CUSPREAD2D_SUBPROB cuspread2df_subprob
#define CUSPREAD2D_PAUL cuspread2df_paul
#define CUSPREAD2D_PAUL_PROP cuspread2df_paul_prop
/* spreading 3d */
#define CUSPREAD3D cuspread3df
#define CUSPREAD3D_NUPTSDRIVEN_PROP cuspread3df_nuptsdriven_prop
#define CUSPREAD3D_NUPTSDRIVEN cuspread3df_nuptsdriven
#define CUSPREAD3D_BLOCKGATHER_PROP cuspread3df_blockgather_prop
#define CUSPREAD3D_BLOCKGATHER cuspread3df_blockgather
#define CUSPREAD3D_SUBPROB_PROP cuspread3df_subprob_prop
#define CUSPREAD3D_SUBPROB cuspread3df_subprob
/* interp */
#define CUINTERP1D cuinterp1df
#define CUINTERP2D cuinterp2df
#define CUINTERP3D cuinterp3df
#define CUINTERP1D_NUPTSDRIVEN cuinterp1df_nuptsdriven
#define CUINTERP2D_NUPTSDRIVEN cuinterp2df_nuptsdriven
#define CUINTERP2D_SUBPROB cuinterp2df_subprob
#define CUINTERP3D_NUPTSDRIVEN cuinterp3df_nuptsdriven
#define CUINTERP3D_SUBPROB cuinterp3df_subprob
/* deconvolve */
#define CUDECONVOLVE1D cudeconvolve1df
#define CUDECONVOLVE2D cudeconvolve2df
#define CUDECONVOLVE3D cudeconvolve3df
/* structs */
#define CUFINUFFT_PLAN_S cufinufftf_plan_s
#define CUFINUFFT_PLAN cufinufftf_plan
/* fseries kernel */
#define CUFSERIESKERNELCOMPUTE cufserieskernelcompute_f

#else

#define CUFINUFFT_DEFAULT_OPTS cufinufft_default_opts
#define CUFINUFFT_MAKEPLAN cufinufft_makeplan
#define CUFINUFFT_SETPTS cufinufft_setpts
#define CUFINUFFT_EXECUTE cufinufft_execute
#define CUFINUFFT_DESTROY cufinufft_destroy
#define CUFINUFFT1D1_EXEC cufinufft1d1_exec
#define CUFINUFFT1D2_EXEC cufinufft1d2_exec
#define CUFINUFFT2D1_EXEC cufinufft2d1_exec
#define CUFINUFFT2D2_EXEC cufinufft2d2_exec
#define CUFINUFFT3D1_EXEC cufinufft3d1_exec
#define CUFINUFFT3D2_EXEC cufinufft3d2_exec
#define SETUP_BINSIZE setup_binsize
/* memtransfer.h */
#define ALLOCGPUMEM1D_PLAN allocgpumem1d_plan
#define ALLOCGPUMEM1D_NUPTS allocgpumem1d_nupts
#define FREEGPUMEMORY1D freegpumemory1d
#define ALLOCGPUMEM2D_PLAN allocgpumem2d_plan
#define ALLOCGPUMEM2D_NUPTS allocgpumem2d_nupts
#define FREEGPUMEMORY2D freegpumemory2d
#define ALLOCGPUMEM3D_PLAN allocgpumem3d_plan
#define ALLOCGPUMEM3D_NUPTS allocgpumem3d_nupts
#define FREEGPUMEMORY3D freegpumemory3d
/* spreading and interp only*/
#define CUFINUFFT_SPREAD1D cufinufft_spread1d
#define CUFINUFFT_SPREAD2D cufinufft_spread2d
#define CUFINUFFT_SPREAD3D cufinufft_spread3d
#define CUFINUFFT_INTERP1D cufinufft_interp1d
#define CUFINUFFT_INTERP2D cufinufft_interp2d
#define CUFINUFFT_INTERP3D cufinufft_interp3d
/* spreading 1D */
#define CUSPREAD1D cuspread1d
#define CUSPREAD1D_NUPTSDRIVEN_PROP cuspread1d_nuptsdriven_prop
#define CUSPREAD1D_NUPTSDRIVEN cuspread1d_nuptsdriven
#define CUSPREAD1D_SUBPROB_PROP cuspread1d_subprob_prop
#define CUSPREAD1D_SUBPROB cuspread1d_subprob
/* spreading 2D */
#define CUSPREAD2D cuspread2d
#define CUSPREAD2D_NUPTSDRIVEN_PROP cuspread2d_nuptsdriven_prop
#define CUSPREAD2D_NUPTSDRIVEN cuspread2d_nuptsdriven
#define CUSPREAD2D_SUBPROB_PROP cuspread2d_subprob_prop
#define CUSPREAD2D_SUBPROB cuspread2d_subprob
#define CUSPREAD2D_PAUL cuspread2d_paul
#define CUSPREAD2D_PAUL_PROP cuspread2d_paul_prop
/* spreading 3d */
#define CUSPREAD3D cuspread3d
#define CUSPREAD3D_NUPTSDRIVEN_PROP cuspread3d_nuptsdriven_prop
#define CUSPREAD3D_NUPTSDRIVEN cuspread3d_nuptsdriven
#define CUSPREAD3D_BLOCKGATHER_PROP cuspread3d_blockgather_prop
#define CUSPREAD3D_BLOCKGATHER cuspread3d_blockgather
#define CUSPREAD3D_SUBPROB_PROP cuspread3d_subprob_prop
#define CUSPREAD3D_SUBPROB cuspread3d_subprob
/* interp */
#define CUINTERP1D cuinterp1d
#define CUINTERP2D cuinterp2d
#define CUINTERP3D cuinterp3d
#define CUINTERP1D_NUPTSDRIVEN cuinterp1d_nuptsdriven
#define CUINTERP2D_NUPTSDRIVEN cuinterp2d_nuptsdriven
#define CUINTERP2D_SUBPROB cuinterp2d_subprob
#define CUINTERP3D_NUPTSDRIVEN cuinterp3d_nuptsdriven
#define CUINTERP3D_SUBPROB cuinterp3d_subprob
/* deconvolve */
#define CUDECONVOLVE1D cudeconvolve1d
#define CUDECONVOLVE2D cudeconvolve2d
#define CUDECONVOLVE3D cudeconvolve3d
/* structs */
#define CUFINUFFT_PLAN_S cufinufft_plan_s
#define CUFINUFFT_PLAN cufinufft_plan
/* fseries kernel */
#define CUFSERIESKERNELCOMPUTE cufserieskernelcompute

#endif

typedef struct CUFINUFFT_PLAN_S {
    cufinufft_opts opts;
    finufft_spread_opts spopts;

    int type;
    int dim;
    int M;
    int nf1;
    int nf2;
    int nf3;
    int ms;
    int mt;
    int mu;
    int ntransf;
    int maxbatchsize;
    int iflag;

    int totalnumsubprob;
    int byte_now;
    CUFINUFFT_FLT *fwkerhalf1;
    CUFINUFFT_FLT *fwkerhalf2;
    CUFINUFFT_FLT *fwkerhalf3;

    CUFINUFFT_FLT *kx;
    CUFINUFFT_FLT *ky;
    CUFINUFFT_FLT *kz;
    CUCPX *c;
    CUCPX *fw;
    CUCPX *fk;

    // Arrays that used in subprob method
    int *idxnupts;        // length: #nupts, index of the nupts in the bin-sorted order
    int *sortidx;         // length: #nupts, order inside the bin the nupt belongs to
    int *numsubprob;      // length: #bins,  number of subproblems in each bin
    int *binsize;         // length: #bins, number of nonuniform ponits in each bin
    int *binstartpts;     // length: #bins, exclusive scan of array binsize
    int *subprob_to_bin;  // length: #subproblems, the bin the subproblem works on
    int *subprobstartpts; // length: #bins, exclusive scan of array numsubprob

    // Extra arrays for Paul's method
    int *finegridsize;
    int *fgstartpts;

    // Arrays for 3d (need to sort out)
    int *numnupts;
    int *subprob_to_nupts;

    cufftHandle fftplan;
    cudaStream_t *streams;

} CUFINUFFT_PLAN_S;

// The plan that is passed around is a pointer to a struct.
// makeplan will utilize a double pointer.
// This encourages bindings to treat the struct as opaque.
typedef struct CUFINUFFT_PLAN_S *CUFINUFFT_PLAN;

#define checkCufftErrors(call)

#ifdef __cplusplus
extern "C" {
#endif
int cufinufft_default_opts(int type, int dim, cufinufft_opts *opts);

int cufinufft_makeplan(int type, int dim, int *n_modes, int iflag, int ntransf, double tol, int maxbatchsize,
                       cufinufftf_plan *d_plan_ptr, cufinufft_opts *opts);
int cufinufft_setpts(int M, double *h_kx, double *h_ky, double *h_kz, int N, double *h_s,
                     double *h_t, double *h_u, cufinufftf_plan d_plan);
int cufinufft_execute(cuDoubleComplex *h_c, cuDoubleComplex *h_fk, cufinufftf_plan d_plan);
int cufinufft_destroy(cufinufftf_plan d_plan);

    
int cufinufft_makeplanf(int type, int dim, int *n_modes, int iflag, int ntransf, double tol, int maxbatchsize,
                        cufinufftf_plan *d_plan_ptr, cufinufft_opts *opts);
int cufinufft_setptsf(int M, float *h_kx, float *h_ky, float *h_kz, int N, float *h_s,
                      float *h_t, float *h_u, cufinufftf_plan d_plan);
int cufinufft_executef(cuFloatComplex *h_c, cuFloatComplex *h_fk, cufinufftf_plan d_plan);
int cufinufft_destroyf(cufinufftf_plan d_plan);
#ifdef __cplusplus
}
#endif

// 1d
template <typename T>
int cufinufft1d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, CUFINUFFT_PLAN d_plan);
template <typename T>
int cufinufft1d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, CUFINUFFT_PLAN d_plan);

// 2d
template <typename T>
int cufinufft2d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, CUFINUFFT_PLAN d_plan);
template <typename T>
int cufinufft2d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, CUFINUFFT_PLAN d_plan);

// 3d
template <typename T>
int cufinufft3d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, CUFINUFFT_PLAN d_plan);
template <typename T>
int cufinufft3d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, CUFINUFFT_PLAN d_plan);

#endif
