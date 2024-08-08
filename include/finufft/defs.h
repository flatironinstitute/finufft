// Library private definitions & macros; also used by some test routines.
// If SINGLE defined, chooses single prec, otherwise double prec.
// Must be #included *after* finufft.h which clobbers FINUFFT* macros
// (see discussion near line 145 of this file).

// Split out by Joakim Anden, Alex Barnett 9/20/18-9/24/18.
// Merged in dataTypes, private/public header split, clean. Barnett 6/7/22.
// finufft_plan_s made private, Wenda's idea.  Barnett 8/8/22.

/* Devnotes:
   1) Only need work for C++ since that's how compiled, including f_plan_s.
   (But we use C-style templating, following fftw, etc.)
*/

#ifndef DEFS_H
#define DEFS_H

// public header gives access to f_opts, f_spread_opts, f_plan...
// (and clobbers FINUFFT* macros; watch out!)
#include <finufft.h>

// --------------- Private data types for compilation in either prec ---------
// Devnote: must match those in relevant prec of public finufft.h interface!

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday.
#define BIGINT  int64_t
#define UBIGINT uint64_t
// Precision-independent real and complex types, for private lib/test compile
#ifdef SINGLE
#define FLT float
#else
#define FLT double
#endif
// next line possibly obsolete...
#define _USE_MATH_DEFINES
#include <complex> // we define C++ complex type only
#define CPX std::complex<FLT>

// inline macro, to force inlining of small functions
// this avoids the use of macros to implement functions
#if defined(_MSC_VER)
#define FINUFFT_ALWAYS_INLINE __forceinline inline
#define FINUFFT_NEVER_INLINE  __declspec(noinline)
#define FINUFFT_RESTRICT      __restrict
#define FINUFFT_UNREACHABLE   __assume(0)
#define FINUFFT_UNLIKELY(x)   (x)
#define FINUFFT_LIKELY(x)     (x)
#elif defined(__GNUC__) || defined(__clang__)
#define FINUFFT_ALWAYS_INLINE __attribute__((always_inline)) inline
#define FINUFFT_NEVER_INLINE  __attribute__((noinline))
#define FINUFFT_RESTRICT      __restrict__
#define FINUFFT_UNREACHABLE   __builtin_unreachable()
#define FINUFFT_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#define FINUFFT_LIKELY(x)     __builtin_expect(!!(x), 1)
#else
#define FINUFFT_ALWAYS_INLINE inline
#define FINUFFT_NEVER_INLINE
#define FINUFFT_RESTRICT
#define FINUFFT_UNREACHABLE
#define FINUFFT_UNLIKELY(x) (x)
#define FINUFFT_LIKELY(x)   (x)
#endif

// ------------- Library-wide algorithm parameter settings ----------------

// Library version (is a string)
#define FINUFFT_VER          "2.3.0-rc1"

// Smallest possible kernel spread width per dimension, in fine grid points
// (used only in spreadinterp.cpp)
#define MIN_NSPREAD          2

// Largest possible kernel spread width per dimension, in fine grid points
// (used only in spreadinterp.cpp)
#define MAX_NSPREAD          16

// Fraction growth cut-off in utils:arraywidcen, sets when translate in type-3
#define ARRAYWIDCEN_GROWFRAC 0.1

// Max number of positive quadr nodes for kernel FT (used only in common.cpp)
#define MAX_NQUAD            100

// Internal (nf1 etc) array allocation size that immediately raises error.
// (Note: next235 takes 1s for 1e11, so it is also to prevent hang here.)
// Increase this if you need >10TB (!) RAM...
#define MAX_NF               (BIGINT)1e12

// Maximum allowed number M of NU points; useful to catch incorrectly cast int32
// values for M = nj (also nk in type 3)...
#define MAX_NU_PTS           (BIGINT)1e14

// -------------- Math consts (not in math.h) and useful math macros ----------
#include <math.h>

// either-precision unit imaginary number...
#define IMA (CPX(0.0, 1.0))
// using namespace std::complex_literals;  // needs C++14, provides 1i, 1if
#ifndef M_PI // Windows apparently doesn't have this const
#define M_PI 3.14159265358979329
#endif
#define M_1_2PI 0.159154943091895336
#define M_2PI   6.28318530717958648
// to avoid mixed precision operators in eg i*pi, an either-prec PI...
#define PI      (FLT) M_PI

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
// #define rand01() (((FLT)(rand()%RAND_MAX))/RAND_MAX)
#define rand01()     ((FLT)rand() / (FLT)RAND_MAX)
// unif[-1,1]:
#define randm11()    (2 * rand01() - (FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11()   (randm11() + IMA * randm11())

// Thread-safe seed-carrying versions of above (x is ptr to seed)...
#define rand01r(x)   ((FLT)rand_r(x) / (FLT)RAND_MAX)
// unif[-1,1]:
#define randm11r(x)  (2 * rand01r(x) - (FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11r(x) (randm11r(x) + IMA * randm11r(x))

// ----- OpenMP macros which also work when omp not present -----
// Allows compile-time switch off of openmp, so compilation without any openmp
// is done (Note: _OPENMP is automatically set by -fopenmp compile flag)
#ifdef _OPENMP
#include <omp.h>
// point to actual omp utils
#define MY_OMP_GET_NUM_THREADS()  omp_get_num_threads()
#define MY_OMP_GET_MAX_THREADS()  omp_get_max_threads()
#define MY_OMP_GET_THREAD_NUM()   omp_get_thread_num()
#define MY_OMP_SET_NUM_THREADS(x) omp_set_num_threads(x)
#else
// non-omp safe dummy versions of omp utils...
#define MY_OMP_GET_NUM_THREADS() 1
#define MY_OMP_GET_MAX_THREADS() 1
#define MY_OMP_GET_THREAD_NUM()  0
#define MY_OMP_SET_NUM_THREADS(x)
#endif

// Prec-switching name macros (respond to SINGLE), used in lib & test sources
// and the plan object below.
// Note: crucially, these are now indep of macros used to gen public finufft.h!
// However, some overlap in name (FINUFFTIFY*, FINUFFT_PLAN*), meaning
// finufft.h cannot be included after defs.h since it undefs these overlaps :(
#ifdef SINGLE
// a macro to prepend finufft or finufftf to a string without a space.
// The 2nd level of indirection is needed for safety, see:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-token-pasting
#define FINUFFTIFY_UNSAFE(x) finufftf##x
#else
#define FINUFFTIFY_UNSAFE(x) finufft##x
#endif
#define FINUFFTIFY(x)        FINUFFTIFY_UNSAFE(x)
// Now use the above tool to set up 2020-style macros used in tester source...
#define FINUFFT_PLAN         FINUFFTIFY(_plan)
#define FINUFFT_PLAN_S       FINUFFTIFY(_plan_s)
#define FINUFFT_DEFAULT_OPTS FINUFFTIFY(_default_opts)
#define FINUFFT_MAKEPLAN     FINUFFTIFY(_makeplan)
#define FINUFFT_SETPTS       FINUFFTIFY(_setpts)
#define FINUFFT_EXECUTE      FINUFFTIFY(_execute)
#define FINUFFT_DESTROY      FINUFFTIFY(_destroy)
#define FINUFFT1D1           FINUFFTIFY(1d1)
#define FINUFFT1D2           FINUFFTIFY(1d2)
#define FINUFFT1D3           FINUFFTIFY(1d3)
#define FINUFFT2D1           FINUFFTIFY(2d1)
#define FINUFFT2D2           FINUFFTIFY(2d2)
#define FINUFFT2D3           FINUFFTIFY(2d3)
#define FINUFFT3D1           FINUFFTIFY(3d1)
#define FINUFFT3D2           FINUFFTIFY(3d2)
#define FINUFFT3D3           FINUFFTIFY(3d3)
#define FINUFFT1D1MANY       FINUFFTIFY(1d1many)
#define FINUFFT1D2MANY       FINUFFTIFY(1d2many)
#define FINUFFT1D3MANY       FINUFFTIFY(1d3many)
#define FINUFFT2D1MANY       FINUFFTIFY(2d1many)
#define FINUFFT2D2MANY       FINUFFTIFY(2d2many)
#define FINUFFT2D3MANY       FINUFFTIFY(2d3many)
#define FINUFFT3D1MANY       FINUFFTIFY(3d1many)
#define FINUFFT3D2MANY       FINUFFTIFY(3d2many)
#define FINUFFT3D3MANY       FINUFFTIFY(3d3many)

// --------  FINUFFT's plan object, prec-switching version ------------------
// NB: now private (the public C++ or C etc user sees an opaque pointer to it)

#include <finufft/fft.h> // (must come after complex.h)

// group together a bunch of type 3 rescaling/centering/phasing parameters:
#define TYPE3PARAMS FINUFFTIFY(_type3Params)
typedef struct {
  FLT X1, C1, D1, h1, gam1; // x dim: X=halfwid C=center D=freqcen h,gam=rescale
  FLT X2, C2, D2, h2, gam2; // y
  FLT X3, C3, D3, h3, gam3; // z
} TYPE3PARAMS;

typedef struct FINUFFT_PLAN_S { // the main plan object, fully C++

  int type;                     // transform type (Rokhlin naming): 1,2 or 3
  int dim;                      // overall dimension: 1,2 or 3
  int ntrans;          // how many transforms to do at once (vector or "many" mode)
  BIGINT nj;           // num of NU pts in type 1,2 (for type 3, num input x pts)
  BIGINT nk;           // number of NU freq pts (type 3 only)
  FLT tol;             // relative user tolerance
  int batchSize;       // # strength vectors to group together for FFTW, etc
  int nbatch;          // how many batches done to cover all ntrans vectors

  BIGINT ms;           // number of modes in x (1) dir (historical CMCL name) = N1
  BIGINT mt;           // number of modes in y (2) direction = N2
  BIGINT mu;           // number of modes in z (3) direction = N3
  BIGINT N;            // total # modes (prod of above three)

  BIGINT nf1;          // size of internal fine grid in x (1) direction
  BIGINT nf2;          // " y (2)
  BIGINT nf3;          // " z (3)
  BIGINT nf;           // total # fine grid points (product of the above three)

  int fftSign;         // sign in exponential for NUFFT defn, guaranteed to be +-1

  FLT *phiHat1;        // FT of kernel in t1,2, on x-axis mode grid
  FLT *phiHat2;        // " y-axis.
  FLT *phiHat3;        // " z-axis.

  CPX *fwBatch;        // (batches of) fine grid(s) for FFTW to plan
                       // & act on. Usually the largest working array

  BIGINT *sortIndices; // precomputed NU pt permutation, speeds spread/interp
  bool didSort;        // whether binsorting used (false: identity perm used)

  FLT *X, *Y, *Z;      // for t1,2: ptr to user-supplied NU pts (no new allocs).
                       // for t3: allocated as "primed" (scaled) src pts x'_j, etc

  // type 3 specific
  FLT *S, *T, *U;           // pointers to user's target NU pts arrays (no new allocs)
  CPX *prephase;            // pre-phase, for all input NU pts
  CPX *deconv;              // reciprocal of kernel FT, phase, all output NU pts
  CPX *CpBatch;             // working array of prephased strengths
  FLT *Sp, *Tp, *Up;        // internal primed targs (s'_k, etc), allocated
  TYPE3PARAMS t3P;          // groups together type 3 shift, scale, phase, parameters
  FINUFFT_PLAN innerT2plan; // ptr used for type 2 in step 2 of type 3

  // other internal structs; each is C-compatible of course
#ifndef FINUFFT_USE_DUCC0
  FFTW_PLAN fftwPlan;
#endif
  finufft_opts opts; // this and spopts could be made ptrs
  finufft_spread_opts spopts;

} FINUFFT_PLAN_S;

#undef TYPE3PARAMS

#endif // DEFS_H
