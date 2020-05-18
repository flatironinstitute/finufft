#ifndef __CUFINUFFT_H__
#define __CUFINUFFT_H__

#include <cufft.h>
#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include "../finufft/utils.h"
#include "../finufft/spreadinterp.h"
//#include "../finufft/finufft.h"
#include "../finufft/common.h"

enum finufft_type {type1,type2,type3};

struct nufft_opts {   // see common/finufft_default_opts() for defaults
	FLT upsampfac;      // upsampling ratio sigma, only 2.0 (standard) is implemented
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
};

typedef struct {
	finufft_type    type;
	nufft_opts      opts; 
	spread_opts     spopts;

	int dim;
	int M;
	int nf1;
	int nf2;
	int nf3;
	int ms;
	int mt;
	int mu;
	int ntransf;
	int ntransfcufftplan;
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
int cufinufft_default_opts(finufft_type type, int dim, nufft_opts &opts);
int cufinufft_makeplan(finufft_type type, int dim, int *n_modes, int iflag, 
	int ntransf, FLT tol, int ntransfcufftplan, cufinufft_plan *d_plan);
int cufinufft_setNUpts(int M, FLT* h_kx, FLT* h_ky, FLT* h_kz, int N, FLT *h_s, 
	FLT *h_t, FLT *h_u, cufinufft_plan *d_plan);
int cufinufft_exec(CUCPX* h_c, CUCPX* h_fk, cufinufft_plan *d_plan);
int cufinufft_destroy(cufinufft_plan *d_plan);

// 2d
int cufinufft2d1_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan);
int cufinufft2d2_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan);

// 3d
int cufinufft3d1_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan);
int cufinufft3d2_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan);
#endif
