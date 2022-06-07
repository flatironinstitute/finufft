// Library private definitions & macros; also used by some test routines.
// If SINGLE defined, chooses single prec, otherwise double prec.
// Must be #included *after* finufft.h which clobbers FINUFFT* macros
// (see discussion at end of this file).

// Split out by Joakim Anden, Alex Barnett 9/20/18-9/24/18.
// Merged in dataTypes, private/public header split, clean. Barnett 6/7/22.

/* Devnotes:
   1) Only need work in C++ since that's how compiled.
   (But we use C-style templating, following fftw, etc.)
*/

#ifndef DEFS_H
#define DEFS_H



// ---------- Global error/warning output codes for the library ---------------
// (it could be argued these belong in finufft.h, but to avoid polluting
//  user's name space we keep them here)
// NB: if change these numbers, also must regen test/results/dumbinputs.refout
#define WARN_EPS_TOO_SMALL       1
// this means that a fine grid array dim exceeded MAX_NF; no malloc tried...
#define ERR_MAXNALLOC            2
#define ERR_SPREAD_BOX_SMALL     3
#define ERR_SPREAD_PTS_OUT_RANGE 4
#define ERR_SPREAD_ALLOC         5
#define ERR_SPREAD_DIR           6
#define ERR_UPSAMPFAC_TOO_SMALL  7
#define ERR_HORNER_WRONG_BETA    8
#define ERR_NTRANS_NOTVALID      9
#define ERR_TYPE_NOTVALID        10
// some generic internal allocation failure...
#define ERR_ALLOC                11
#define ERR_DIM_NOTVALID         12
#define ERR_SPREAD_THREAD_NOTVALID 13


// --------------- Private data types for compilation in either prec ---------
// Devnote: must match those in relevant prec of public finufft.h interface!

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday.
#define BIGINT int64_t
// Precision-independent real and complex types, for private lib/test compile
#ifdef SINGLE
  #define FLT float
#else
  #define FLT double
#endif
// next line possibly obsolete...
#define _USE_MATH_DEFINES
#include <complex>          // we define C++ complex type only
#define CPX std::complex<FLT>



// ------------- Library-wide algorithm parameter settings ----------------

// Library version (is a string)
#define FINUFFT_VER "2.1.0"

// Largest possible kernel spread width per dimension, in fine grid points
// (used only in spreadinterp.cpp)
#define MAX_NSPREAD 16

// Fraction growth cut-off in utils:arraywidcen, sets when translate in type-3
#define ARRAYWIDCEN_GROWFRAC 0.1

// Max number of positive quadr nodes for kernel FT (used only in common.cpp)
#define MAX_NQUAD 100

// Internal (nf1 etc) array allocation size that immediately raises error.
// (Note: next235 takes 1s for this size, so it is also to prevent hang here.)
// Increase this if you need >1TB RAM... (used only in common.cpp)
#define MAX_NF    (BIGINT)1e11



// -------------- Math consts (not in math.h) and useful math macros ----------
#include <math.h>

// either-precision unit imaginary number...
#define IMA (CPX(0.0,1.0))
// using namespace std::complex_literals;  // needs C++14, provides 1i, 1if
#ifndef M_PI                     // Windows apparently doesn't have this const
  #define M_PI    3.14159265358979329
#endif
#define M_1_2PI 0.159154943091895336
#define M_2PI   6.28318530717958648
// to avoid mixed precision operators in eg i*pi, an either-prec PI...
#define PI (FLT)M_PI

// machine epsilon for decisions of achievable tolerance...
#ifdef SINGLE
  #define EPSILON (float)6e-08
#else
  #define EPSILON (double)1.1e-16
#endif


// Random numbers: crappy unif random number generator in [0,1).
// These macros should probably be replaced by modern C++ std lib or random123.
// (RAND_MAX is in stdlib.h)
#include <stdlib.h>
//#define rand01() (((FLT)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((FLT)rand()/RAND_MAX)
// unif[-1,1]:
#define randm11() (2*rand01() - (FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11() (randm11() + IMA*randm11())

// Thread-safe seed-carrying versions of above (x is ptr to seed)...
#define rand01r(x) ((FLT)rand_r(x)/RAND_MAX)
// unif[-1,1]:
#define randm11r(x) (2*rand01r(x) - (FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11r(x) (randm11r(x) + IMA*randm11r(x))


// ----- OpenMP (and FFTW omp) macros which also work when omp not present -----
// Allows compile-time switch off of openmp, so compilation without any openmp
// is done (Note: _OPENMP is automatically set by -fopenmp compile flag)
#ifdef _OPENMP
  #include <omp.h>
  // point to actual omp utils
  #define MY_OMP_GET_NUM_THREADS() omp_get_num_threads()
  #define MY_OMP_GET_MAX_THREADS() omp_get_max_threads()
  #define MY_OMP_GET_THREAD_NUM() omp_get_thread_num()
  #define MY_OMP_SET_NUM_THREADS(x) omp_set_num_threads(x)
#else
  // non-omp safe dummy versions of omp utils, and dummy fftw threads calls...
  #define MY_OMP_GET_NUM_THREADS() 1
  #define MY_OMP_GET_MAX_THREADS() 1
  #define MY_OMP_GET_THREAD_NUM() 0
  #define MY_OMP_SET_NUM_THREADS(x)
  #undef FFTW_INIT
  #define FFTW_INIT()
  #undef FFTW_PLAN_TH
  #define FFTW_PLAN_TH(x)
  #undef FFTW_CLEANUP_THREADS
  #define FFTW_CLEANUP_THREADS()
#endif


// Prec-switching name macros (respond to SINGLE), used in lib & test sources:
// Note: crucially, these are now indep of macros used to gen public finufft.h!
// However, some overlap in name (FINUFFTIFY*, FINUFFT_PLAN*), meaning
// finufft.h cannot be included after defs.h since it undefs these overlaps :(
#ifdef SINGLE
// macro to prepend finufft or finufftf to a string without a space.
// The 2nd level of indirection is needed for safety, see:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-token-pasting
#define FINUFFTIFY_UNSAFE(x) finufftf##x
#else
#define FINUFFTIFY_UNSAFE(x) finufft##x
#endif
#define FINUFFTIFY(x) FINUFFTIFY_UNSAFE(x)
// Now use the tool to set up 2020-style macros used in tester source...
#define FINUFFT_PLAN FINUFFTIFY(_plan)
#define FINUFFT_PLAN_S FINUFFTIFY(_plan_s)
#define FINUFFT_DEFAULT_OPTS FINUFFTIFY(_default_opts)
#define FINUFFT_MAKEPLAN FINUFFTIFY(_makeplan)
#define FINUFFT_SETPTS FINUFFTIFY(_setpts)
#define FINUFFT_EXECUTE FINUFFTIFY(_execute)
#define FINUFFT_DESTROY FINUFFTIFY(_destroy)
#define FINUFFT1D1 FINUFFTIFY(1d1)
#define FINUFFT1D2 FINUFFTIFY(1d2)
#define FINUFFT1D3 FINUFFTIFY(1d3)
#define FINUFFT2D1 FINUFFTIFY(2d1)
#define FINUFFT2D2 FINUFFTIFY(2d2)
#define FINUFFT2D3 FINUFFTIFY(2d3)
#define FINUFFT3D1 FINUFFTIFY(3d1)
#define FINUFFT3D2 FINUFFTIFY(3d2)
#define FINUFFT3D3 FINUFFTIFY(3d3)
#define FINUFFT1D1MANY FINUFFTIFY(1d1many)
#define FINUFFT1D2MANY FINUFFTIFY(1d2many)
#define FINUFFT1D3MANY FINUFFTIFY(1d3many)
#define FINUFFT2D1MANY FINUFFTIFY(2d1many)
#define FINUFFT2D2MANY FINUFFTIFY(2d2many)
#define FINUFFT2D3MANY FINUFFTIFY(2d3many)
#define FINUFFT3D1MANY FINUFFTIFY(3d1many)
#define FINUFFT3D2MANY FINUFFTIFY(3d2many)
#define FINUFFT3D3MANY FINUFFTIFY(3d3many)


#endif  // DEFS_H
