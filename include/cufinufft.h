#ifndef __CUFINUFFT_H__
#define __CUFINUFFT_H__

#include <cufft.h>
#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include "../contrib/utils.h"
#include "../contrib/spreadinterp.h"
#include "../contrib/common.h"
#include "../src/precision_independent.h"

#ifdef SINGLE

#define CUFINUFFT_DEFAULT_OPTS cufinufftf_default_opts
#define CUFINUFFT_MAKEPLAN cufinufftf_makeplan
#define CUFINUFFT_SETNUPTS cufinufftf_setNUpts
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
#define CUFINUFFTC_SETNUPTS cufinufftcf_setnupts
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

#else

#define CUFINUFFT_DEFAULT_OPTS cufinufft_default_opts
#define CUFINUFFT_MAKEPLAN cufinufft_makeplan
#define CUFINUFFT_SETNUPTS cufinufft_setNUpts
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
#define CUFINUFFTC_SETNUPTS cufinufftc_setnupts
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

#endif

typedef struct cufinufft_opts{   // see cufinufft_default_opts() for defaults
	FLT upsampfac;   // upsampling ratio sigma, only 2.0 (standard) is implemented
	/* following options are for gpu */
	int gpu_method;
	int gpu_sort; // used for 3D nupts driven method

	int gpu_binsizex; // used for 2D, 3D subproblem method
	int gpu_binsizey;
	int gpu_binsizez;

	int gpu_obinsizex; // used for 3D spread block gather method
	int gpu_obinsizey;
	int gpu_obinsizez;

	int gpu_maxsubprobsize;
	int gpu_nstreams; 
	int gpu_kerevalmeth;	// 0: direct exp(sqrt()), 1: Horner ppval
}cufinufft_opts;

typedef struct {
	cufinufft_opts  opts; 
	spread_opts     spopts;

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

}cufinufft_plan;

// For error checking (where should this function be??)
static const char* _cufftGetErrorEnum(cufftResult_t error)
{
	switch(error)
	{
		case CUFFT_SUCCESS:
			return "cufft_success";
		case CUFFT_INVALID_PLAN:
			return "cufft_invalid_plan";
		case CUFFT_ALLOC_FAILED:
			return "cufft_alloc_failed";
		case CUFFT_INVALID_TYPE:
			return "cufft_invalid_type";
		case CUFFT_INVALID_VALUE:
			return "cufft_invalid_value";
		case CUFFT_INTERNAL_ERROR:
			return "cufft_internal_error";
		case CUFFT_EXEC_FAILED:
			return "cufft_exec_failed";
		case CUFFT_SETUP_FAILED:
			return "cufft_setup_failed";
		case CUFFT_INVALID_SIZE:
			return "cufft_invalid_size";
		case CUFFT_UNALIGNED_DATA:
			return "cufft_unaligned data";
		case CUFFT_INCOMPLETE_PARAMETER_LIST:
			return "cufft_incomplete_parameter_list";
		case CUFFT_INVALID_DEVICE:
			return "cufft_invalid_device";
		case CUFFT_PARSE_ERROR:
			return "cufft_parse_error";
		case CUFFT_NO_WORKSPACE:
			return "cufft_no_workspace";
		case CUFFT_NOT_IMPLEMENTED:
			return "cufft_not_implemented";
		case CUFFT_LICENSE_ERROR:
			return "cufft_license_error";
		case CUFFT_NOT_SUPPORTED:
			return "cufft_not_supported";
	}
	return "<unknown>";
}

#define checkCufftErrors(call)

int CUFINUFFT_DEFAULT_OPTS(int type, int dim, cufinufft_opts *opts);
int CUFINUFFT_MAKEPLAN(int type, int dim, int *n_modes, int iflag, 
	int ntransf, FLT tol, int maxbatchsize, cufinufft_plan *d_plan);
int CUFINUFFT_SETNUPTS(int M, FLT* h_kx, FLT* h_ky, FLT* h_kz, int N, FLT *h_s, 
	FLT *h_t, FLT *h_u, cufinufft_plan *d_plan);
int CUFINUFFT_EXEC(CUCPX* h_c, CUCPX* h_fk, cufinufft_plan *d_plan);
int CUFINUFFT_DESTROY(cufinufft_plan *d_plan);
 
// 2d
int CUFINUFFT2D1_EXEC(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan);
int CUFINUFFT2D2_EXEC(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan);

// 3d
int CUFINUFFT3D1_EXEC(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan);
int CUFINUFFT3D2_EXEC(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan);

#endif
