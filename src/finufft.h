// minimum definitions needed for interface to FINUFFT library, from C++ or C

#ifndef FINUFFT_H
#define FINUFFT_H

#include <dataTypes.h>
#include <spreadinterp.h>


enum finufft_type { type1, type2, type3};

// ------------------- the user input options struct ------------------------
typedef struct {      // Note: defaults in common/finufft_default_opts()
  int debug;          // 0: silent, 1: text basic timing output
  int spread_debug;   // passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)
  int spread_sort;    // passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)
  int spread_kerevalmeth; // "     spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)
  int spread_kerpad;  // passed to spread_opts, 0: don't pad to mult of 4, 1: do
  int chkbnds;        // 0: don't check if input NU pts in [-3pi,3pi], 1: do
  int fftw;           // 0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan but faster)
  int modeord;        // 0: CMCL-style increasing mode ordering (neg to pos), or
                      // 1: FFT-style mode ordering (affects type-1,2 only)
  FLT upsampfac;      // upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)
} nufft_opts;



#include <fftw_defs.h>

typedef struct {

  finufft_type type;
  int n_dims;
  int n_transf;
  int M; 

  BIGINT ms;
  BIGINT mt;
  BIGINT mu;
  
  BIGINT nf1;
  BIGINT nf2;
  BIGINT nf3; 
  
  int iflag; 

  FLT * fwker; //fourier coefficients of spreading kernel for all dims
  FFTW_CPX * fw; //fourier coefficients for all dims //need to free
  
  BIGINT *sortIndices; //FREE ME
  bool didSort;
  
  FLT * targetFreqs; //type 3 only 

  FLT *X;
  FLT *Y;
  FLT *Z; 
  
  fftw_plan fftwPlan;
  
  nufft_opts opts;
  spread_opts spopts;
}finufft_plan;







// ------------------ library provides ------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif

// ------------------ Guru Interface ------------------------------------
int make_finufft_plan(finufft_type type, int n_dims, BIGINT* n_modes, int iflag, int n_transf, FLT tol, finufft_plan *plan );
int setNUpoints(finufft_plan * plan , BIGINT M, FLT *Xpts, FLT *Ypts, FLT *Zpts, CPX *targetFreqs); 
int finufft_exec(finufft_plan * plan ,  CPX *weights, CPX * result);
int finufft_destroy(finufft_plan * plan);

// ------------------ Deprecate Interface ------------------------------------
void finufft_default_opts(nufft_opts *o);
int finufft1d1(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts opts);
int finufft1d2(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts opts);
int finufft1d3(BIGINT nj,FLT* x,CPX* c,int iflag,FLT eps,BIGINT nk, FLT* s, CPX* f, nufft_opts opts);

int finufft2d1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d1many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d2many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d3(BIGINT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,BIGINT nk, FLT* s, FLT* t, CPX* fk, nufft_opts opts);

int finufft3d1(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);
int finufft3d2(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);
int finufft3d3(BIGINT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
	       FLT eps,BIGINT nk,FLT* s, FLT* t, FLT *u,
	       CPX* fk, nufft_opts opts);
#ifdef __cplusplus
}
#endif


#endif   // FINUFFT_H
