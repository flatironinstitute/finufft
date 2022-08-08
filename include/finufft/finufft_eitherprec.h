// Switchable-precision interface template for FINUFFT. Used by finufft.h
// (Not for public/direct use: users should include only the header finufft.h)

/* Devnotes.
 1)  Since everything here is exposed to the public interface, macros must be
 safe, eg FINUFFTsomething.
 2)  They will clobber any prior macros starting FINUFFT*, so in the lib/test
 sources finufft.h must be included before defs.h
 3) for debug of header macros, see finufft.h
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

// opaque pointer to finufft_plan private object, for this precision...
#define FINUFFT_PLAN FINUFFTIFY(_plan)
// the plan object pointed to... (doesn't need to be even defined here)
#define FINUFFT_PLAN_S FINUFFTIFY(_plan_s)


////////////////////////////////////////////////////////////////////
// PUBLIC METHOD INTERFACES. All are C-style even when used from C++...
#ifdef __cplusplus
extern "C"
{
#endif

// ----------------- the plan ----------------------------------------------- 
// the plan handle that we pass around is just a pointer to the plan object
// that contains all the info. The latter is invisible to the public user.
typedef struct FINUFFT_PLAN_S * FINUFFT_PLAN;

  
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
