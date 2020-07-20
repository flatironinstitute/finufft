// Switchable-precision interface template for cufinufft. Used by cufinufft.h
// Internal use only: users should link to cufinufft.h

#if (!defined(__CUFINUFFT_H__) && !defined(SINGLE)) || \
  (!defined(__CUFINUFFTF_H__) && defined(SINGLE))
// (note we entered one level of conditional until the end of this header)
// Make sure we don't include double or single headers more than once...

#include <cstdlib>
#include <cufft.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cufinufft_opts.h"
#include "../src/precision_independent.h"
#include "cufinufft_errors.h"

#include "../contrib/utils.h"
#include "../contrib/dataTypes.h"
#include "../contrib/spreadinterp.h"
#include "../contrib/utils_fp.h"


#ifndef SINGLE
#define __CUFINUFFT_H__
#else
#define __CUFINUFFTF_H__
#endif


/* Undefine things so we don't get warnings/errors later */
#undef CUFINUFFT_DEFAULT_OPTS
#undef CUFINUFFT_MAKEPLAN
#undef CUFINUFFT_SETPTS
#undef CUFINUFFT_EXEC
#undef CUFINUFFT_DESTROY
#undef CUFINUFFT2D1_EXEC
#undef CUFINUFFT2D2_EXEC
#undef CUFINUFFT3D1_EXEC
#undef CUFINUFFT3D2_EXEC
#undef SETUP_BINSIZE
/* extern c interface */
#undef CUFINUFFTC_DEFAULT_OPTS
#undef CUFINUFFTC_MAKEPLAN
#undef CUFINUFFTC_SETPTS
#undef CUFINUFFTC_EXEC
#undef CUFINUFFTC_DESTROY
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
#undef CUINTERP2D
#undef CUINTERP3D
#undef CUINTERP2D_NUPTSDRIVEN
#undef CUINTERP2D_SUBPROB
#undef CUINTERP3D_NUPTSDRIVEN
#undef CUINTERP3D_SUBPROB
/* deconvolve */
#undef CUDECONVOLVE2D
#undef CUDECONVOLVE3D
/* structs */
#undef CUFINUFFT_PLAN


#ifdef SINGLE

#define CUFINUFFT_DEFAULT_OPTS cufinufftf_default_opts
#define CUFINUFFT_MAKEPLAN cufinufftf_makeplan
#define CUFINUFFT_SETPTS cufinufftf_setpts
#define CUFINUFFT_EXEC cufinufftf_exec
#define CUFINUFFT_DESTROY cufinufftf_destroy
#define CUFINUFFT2D1_EXEC cufinufftf2d1_exec
#define CUFINUFFT2D2_EXEC cufinufftf2d2_exec
#define CUFINUFFT3D1_EXEC cufinufftf3d1_exec
#define CUFINUFFT3D2_EXEC cufinufftf3d2_exec
#define SETUP_BINSIZE setup_binsizef
/* extern c interface */
#define CUFINUFFTC_DEFAULT_OPTS cufinufftcf_default_opts
#define CUFINUFFTC_MAKEPLAN cufinufftcf_makeplan
#define CUFINUFFTC_SETPTS cufinufftcf_setpts
#define CUFINUFFTC_EXEC cufinufftcf_exec
#define CUFINUFFTC_DESTROY cufinufftcf_destroy
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
#define CUINTERP2D cuinterp2df
#define CUINTERP3D cuinterp3df
#define CUINTERP2D_NUPTSDRIVEN cuinterp2df_nuptsdriven
#define CUINTERP2D_SUBPROB cuinterp2df_subprob
#define CUINTERP3D_NUPTSDRIVEN cuinterp3df_nuptsdriven
#define CUINTERP3D_SUBPROB cuinterp3df_subprob
/* deconvolve */
#define CUDECONVOLVE2D cudeconvolve2df
#define CUDECONVOLVE3D cudeconvolve3df
/* structs */
#define CUFINUFFT_PLAN cufinufftf_plan

#else

#define CUFINUFFT_DEFAULT_OPTS cufinufft_default_opts
#define CUFINUFFT_MAKEPLAN cufinufft_makeplan
#define CUFINUFFT_SETPTS cufinufft_setpts
#define CUFINUFFT_EXEC cufinufft_exec
#define CUFINUFFT_DESTROY cufinufft_destroy
#define CUFINUFFT2D1_EXEC cufinufft2d1_exec
#define CUFINUFFT2D2_EXEC cufinufft2d2_exec
#define CUFINUFFT3D1_EXEC cufinufft3d1_exec
#define CUFINUFFT3D2_EXEC cufinufft3d2_exec
#define SETUP_BINSIZE setup_binsize
/* extern c interface */
#define CUFINUFFTC_DEFAULT_OPTS cufinufftc_default_opts
#define CUFINUFFTC_MAKEPLAN cufinufftc_makeplan
#define CUFINUFFTC_SETPTS cufinufftc_setpts
#define CUFINUFFTC_EXEC cufinufftc_exec
#define CUFINUFFTC_DESTROY cufinufftc_destroy
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
#define CUINTERP2D cuinterp2d
#define CUINTERP3D cuinterp3d
#define CUINTERP2D_NUPTSDRIVEN cuinterp2d_nuptsdriven
#define CUINTERP2D_SUBPROB cuinterp2d_subprob
#define CUINTERP3D_NUPTSDRIVEN cuinterp3d_nuptsdriven
#define CUINTERP3D_SUBPROB cuinterp3d_subprob
/* deconvolve */
#define CUDECONVOLVE2D cudeconvolve2d
#define CUDECONVOLVE3D cudeconvolve3d
/* structs */
#define CUFINUFFT_PLAN cufinufft_plan

#endif

typedef struct {
	cufinufft_opts  opts;
	SPREAD_OPTS     spopts;

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
	FLT *fwkerhalf1;
	FLT *fwkerhalf2;
	FLT *fwkerhalf3;

	FLT *kx;
	FLT *ky;
	FLT *kz;
	CUCPX *c;
	CUCPX *fw;
	CUCPX *fk;

	// Arrays that used in subprob method
	int *idxnupts;
	int *sortidx;
	int *numsubprob;
	int *binsize;
	int *binstartpts;
	int *subprob_to_bin;
	int *subprobstartpts;

	// Extra arrays for Paul's method
	int *finegridsize;
	int *fgstartpts;

	// Arrays for 3d (need to sort out)
	int *numnupts;
	int *subprob_to_nupts;

	cufftHandle fftplan;
	cudaStream_t *streams;

} CUFINUFFT_PLAN;


/* We include common.h here because it depends on SPREAD_OPTS and
   CUFINUFFT_PLAN structs being completely defined first. */
#include "../contrib/common.h"

#define checkCufftErrors(call)

int CUFINUFFT_DEFAULT_OPTS(int type, int dim, cufinufft_opts *opts);
int CUFINUFFT_MAKEPLAN(int type, int dim, int *n_modes, int iflag,
		       int ntransf, FLT tol, int maxbatchsize,
		       CUFINUFFT_PLAN *d_plan, cufinufft_opts *opts);
int CUFINUFFT_SETPTS(int M, FLT* h_kx, FLT* h_ky, FLT* h_kz, int N, FLT *h_s,
	FLT *h_t, FLT *h_u, CUFINUFFT_PLAN *d_plan);
int CUFINUFFT_EXEC(CUCPX* h_c, CUCPX* h_fk, CUFINUFFT_PLAN *d_plan);
int CUFINUFFT_DESTROY(CUFINUFFT_PLAN *d_plan);

// 2d
int CUFINUFFT2D1_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN *d_plan);
int CUFINUFFT2D2_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN *d_plan);

// 3d
int CUFINUFFT3D1_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN *d_plan);
int CUFINUFFT3D2_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN *d_plan);

#endif
