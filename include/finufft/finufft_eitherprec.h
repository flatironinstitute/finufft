// Switchable-precision interface template for FINUFFT. Used by finufft.h
// (Not for public/direct use: users should include only the header finufft.h)

/* Devnotes.
 1)  Since everything here is exposed to the public interface, macros must be
 safe, eg FINUFFTsomething.
 2)  They will clobber any prior macros starting FINUFFT*, so in the lib/test
 sources finufft.h must be included before defs.h
 3) for debug, see finufft.h
*/

// Local precision-switching macros to make the public interface...
#ifdef FINUFFT_SINGLE
// macro to prepend finufft or finufftf to a string without a space.
// The 2nd level of indirection is needed for safety, see:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-token-pasting
#define FINUFFTIFY_UNSAFE(x) finufftf##x
#define FINUFFT_FLT float
#else
#define FINUFFTIFY_UNSAFE(x) finufft##x
#define FINUFFT_FLT double
#endif
#define FINUFFTIFY(x) FINUFFTIFY_UNSAFE(x)

// decide which kind of complex numbers FINUFFT_CPX is (four options)
#ifdef __cplusplus
#define _USE_MATH_DEFINES
#include <complex>          // C++ type
#define FINUFFT_COMPLEXIFY(X) std::complex<X>
#else
#include <complex.h>        // C99 type
#define FINUFFT_COMPLEXIFY(X) X complex
#endif
#define FINUFFT_CPX FINUFFT_COMPLEXIFY(FINUFFT_FLT)


//////////////////////////////////////////////////////////////////////
// finufft_plan struct (plan C-style) for this precision...
#define FINUFFT_PLAN FINUFFTIFY(_plan)
#define FINUFFT_PLAN_S FINUFFTIFY(_plan_s)
#define FINUFFT_TYPE3PARAMS FINUFFTIFY(_type3Params)

// the plan handle that we pass around is just a pointer to the struct that
// contains all the info
typedef struct FINUFFT_PLAN_S * FINUFFT_PLAN;

// group together a bunch of type 3 rescaling/centering/phasing parameters:
typedef struct {
  FINUFFT_FLT X1,C1,D1,h1,gam1;  // x dim: X=halfwid C=center D=freqcen h,gam=rescale
  FINUFFT_FLT X2,C2,D2,h2,gam2;  // y
  FINUFFT_FLT X3,C3,D3,h3,gam3;  // z
} FINUFFT_TYPE3PARAMS;

// only the aspects of FFTW needed in this public-facing header...
#include <fftw3.h>          // (must come after complex.h)
// (following were typedefed in v<=2.0.4, which seems bad for plain macros)
#ifdef FINUFFT_SINGLE
#define FINUFFT_FFTW_CPX fftwf_complex
#define FINUFFT_FFTW_PLAN fftwf_plan
#else
#define FINUFFT_FFTW_CPX fftw_complex
#define FINUFFT_FFTW_PLAN fftw_plan
#endif

typedef struct FINUFFT_PLAN_S {  // the main plan struct; note C-compatible struct
  
  int type;        // transform type (Rokhlin naming): 1,2 or 3
  int dim;         // overall dimension: 1,2 or 3
  int ntrans;      // how many transforms to do at once (vector or "many" mode)
  FINUFFT_BIGINT nj;  // num of NU pts in type 1,2 (for type 3, num input x pts)
  FINUFFT_BIGINT nk;  // number of NU freq pts (type 3 only)
  FINUFFT_FLT tol; // relative user tolerance
  int batchSize;   // # strength vectors to group together for FFTW, etc
  int nbatch;      // how many batches done to cover all ntrans vectors
  
  FINUFFT_BIGINT ms; // number of modes in x (1) dir (historical CMCL name) = N1
  FINUFFT_BIGINT mt; // number of modes in y (2) direction = N2
  FINUFFT_BIGINT mu; // number of modes in z (3) direction = N3
  FINUFFT_BIGINT N;  // total # modes (prod of above three)
  
  FINUFFT_BIGINT nf1;   // size of internal fine grid in x (1) direction
  FINUFFT_BIGINT nf2;   // " y
  FINUFFT_BIGINT nf3;   // " z
  FINUFFT_BIGINT nf;    // total # fine grid points (product of the above three)
  
  int fftSign;     // sign in exponential for NUFFT defn, guaranteed to be +-1

  FINUFFT_FLT* phiHat1;    // FT of kernel in t1,2, on x-axis mode grid
  FINUFFT_FLT* phiHat2;    // " y-axis.
  FINUFFT_FLT* phiHat3;    // " z-axis.
  
  FINUFFT_FFTW_CPX* fwBatch;    // (batches of) fine grid(s) for FFTW to plan
                                // & act on. Usually the largest working array
  
  FINUFFT_BIGINT *sortIndices;  // precomputed NU pt permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FINUFFT_FLT *X, *Y, *Z;  // for t1,2: ptr to user-supplied NU pts (no new allocs).
                   // for t3: allocated as "primed" (scaled) src pts x'_j, etc

  // type 3 specific
  FINUFFT_FLT *S, *T, *U;  // pointers to user's target NU pts arrays (no new allocs)
  FINUFFT_CPX* prephase;   // pre-phase, for all input NU pts
  FINUFFT_CPX* deconv;     // reciprocal of kernel FT, phase, all output NU pts
  FINUFFT_CPX* CpBatch;    // working array of prephased strengths
  FINUFFT_FLT *Sp, *Tp, *Up;  // internal primed targs (s'_k, etc), allocated
  FINUFFT_TYPE3PARAMS t3P; // groups together type 3 shift, scale, phase, parameters
  FINUFFT_PLAN innerT2plan;   // ptr used for type 2 in step 2 of type 3
  
  // other internal structs; each is C-compatible of course
  FINUFFT_FFTW_PLAN fftwPlan;
  finufft_opts opts;     // this and spopts could be made ptrs
  finufft_spread_opts spopts;
  
} FINUFFT_PLAN_S;



////////////////////////////////////////////////////////////////////
// PUBLIC METHOD INTERFACES. All are C-style even when used from C++...
#ifdef __cplusplus
extern "C"
{
#endif

// ------------------ the guru interface ------------------------------------
// (sources in finufft.cpp)
  
  void FINUFFTIFY(_default_opts)(finufft_opts *o);
  int FINUFFTIFY(_makeplan)(int type, int dim, FINUFFT_BIGINT* n_modes, int iflag, int n_transf, FINUFFT_FLT tol, FINUFFT_PLAN* plan, finufft_opts* o);
  int FINUFFTIFY(_setpts)(FINUFFT_PLAN plan , FINUFFT_BIGINT M, FINUFFT_FLT *xj, FINUFFT_FLT *yj, FINUFFT_FLT *zj, FINUFFT_BIGINT N, FINUFFT_FLT *s, FINUFFT_FLT *t, FINUFFT_FLT *u); 
  int FINUFFTIFY(_execute)(FINUFFT_PLAN plan, FINUFFT_CPX* weights, FINUFFT_CPX* result);
  int FINUFFTIFY(_destroy)(FINUFFT_PLAN plan);


// ----------------- the 18 simple interfaces -------------------------------
// (sources in simpleinterfaces.cpp)

  int FINUFFTIFY(1d1)(FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,FINUFFT_BIGINT ms,
                      FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(1d1many)(int ntransf, FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,FINUFFT_BIGINT ms,
                         FINUFFT_CPX* fk, finufft_opts *opts);

  int FINUFFTIFY(1d2)(FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,FINUFFT_BIGINT ms,
                      FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(1d2many)(int ntransf, FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,FINUFFT_BIGINT ms,
                          FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(1d3)(FINUFFT_BIGINT nj,FINUFFT_FLT* x,FINUFFT_CPX* c,int iflag,FINUFFT_FLT eps,FINUFFT_BIGINT nk, FINUFFT_FLT* s, FINUFFT_CPX* f, finufft_opts *opts);
  int FINUFFTIFY(1d3many)(int ntransf, FINUFFT_BIGINT nj,FINUFFT_FLT* x,FINUFFT_CPX* c,int iflag,FINUFFT_FLT eps,FINUFFT_BIGINT nk, FINUFFT_FLT* s, FINUFFT_CPX* f, finufft_opts *opts);
  int FINUFFTIFY(2d1)(FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_FLT *yj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,
	       FINUFFT_BIGINT ms, FINUFFT_BIGINT mt, FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(2d1many)(int ndata, FINUFFT_BIGINT nj, FINUFFT_FLT* xj, FINUFFT_FLT *yj, FINUFFT_CPX* c, int iflag,
                   FINUFFT_FLT eps, FINUFFT_BIGINT ms, FINUFFT_BIGINT mt, FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(2d2)(FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_FLT *yj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,
	       FINUFFT_BIGINT ms, FINUFFT_BIGINT mt, FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(2d2many)(int ndata, FINUFFT_BIGINT nj, FINUFFT_FLT* xj, FINUFFT_FLT *yj, FINUFFT_CPX* c, int iflag,
                   FINUFFT_FLT eps, FINUFFT_BIGINT ms, FINUFFT_BIGINT mt, FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(2d3)(FINUFFT_BIGINT nj,FINUFFT_FLT* x,FINUFFT_FLT *y,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,FINUFFT_BIGINT nk, FINUFFT_FLT* s, FINUFFT_FLT* t, FINUFFT_CPX* fk, finufft_opts *opts);

  int FINUFFTIFY(2d3many)(int ntransf, FINUFFT_BIGINT nj,FINUFFT_FLT* x,FINUFFT_FLT *y,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,FINUFFT_BIGINT nk, FINUFFT_FLT* s, FINUFFT_FLT* t, FINUFFT_CPX* fk, finufft_opts *opts);

  int FINUFFTIFY(3d1)(FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_FLT *yj,FINUFFT_FLT *zj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,
	       FINUFFT_BIGINT ms, FINUFFT_BIGINT mt, FINUFFT_BIGINT mu, FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(3d1many)(int ntransfs, FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_FLT *yj,FINUFFT_FLT *zj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,
	       FINUFFT_BIGINT ms, FINUFFT_BIGINT mt, FINUFFT_BIGINT mu, FINUFFT_CPX* fk, finufft_opts *opts);

  int FINUFFTIFY(3d2)(FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_FLT *yj,FINUFFT_FLT *zj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,
	       FINUFFT_BIGINT ms, FINUFFT_BIGINT mt, FINUFFT_BIGINT mu, FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(3d2many)(int ntransf, FINUFFT_BIGINT nj,FINUFFT_FLT* xj,FINUFFT_FLT *yj,FINUFFT_FLT *zj,FINUFFT_CPX* cj,int iflag,FINUFFT_FLT eps,
	       FINUFFT_BIGINT ms, FINUFFT_BIGINT mt, FINUFFT_BIGINT mu, FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(3d3)(FINUFFT_BIGINT nj,FINUFFT_FLT* x,FINUFFT_FLT *y,FINUFFT_FLT *z, FINUFFT_CPX* cj,int iflag,
	       FINUFFT_FLT eps,FINUFFT_BIGINT nk,FINUFFT_FLT* s, FINUFFT_FLT* t, FINUFFT_FLT *u,
	       FINUFFT_CPX* fk, finufft_opts *opts);
  int FINUFFTIFY(3d3many)(int ntransf, FINUFFT_BIGINT nj,FINUFFT_FLT* x,FINUFFT_FLT *y,FINUFFT_FLT *z, FINUFFT_CPX* cj,int iflag,
	       FINUFFT_FLT eps,FINUFFT_BIGINT nk,FINUFFT_FLT* s, FINUFFT_FLT* t, FINUFFT_FLT *u,
	       FINUFFT_CPX* fk, finufft_opts *opts);
  
#ifdef __cplusplus
}
#endif


// clean up things that were purely local to this file
#undef FINUFFT_COMPLEXIFY
#undef FINUFFTIFY_UNSAFE
#undef FINUFFTIFY
#undef FINUFFT_FLT
#undef FINUFFT_CPX
#undef FINUFFT_PLAN
#undef FINUFFT_PLAN_S
#undef FINUFFT_TYPE3PARAMS
#undef FINUFFT_FFTW_CPX
#undef FINUFFT_FFTW_PLAN
