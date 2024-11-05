#ifndef FINUFFT_CORE_H
#define FINUFFT_CORE_H

/* IMPORTANT: for Windows compilers, you should add a line
        #define FINUFFT_DLL
   here if you are compiling/using FINUFFT as a DLL,
   in order to do the proper importing/exporting, or
   alternatively compile with -DFINUFFT_DLL or the equivalent
   command-line flag.  This is not necessary under MinGW/Cygwin, where
   libtool does the imports/exports automatically.
   Alternatively use include(GenerateExportHeader) and
   generate_export_header(finufft) to auto generate an header containing
   these defines.The main reason is that if msvc changes the way it deals
   with it in the future we just need to update cmake for it to work
   instead of having a check on the msvc version. */
#if defined(FINUFFT_DLL) && (defined(_WIN32) || defined(__WIN32__))
#if defined(dll_EXPORTS)
#define FINUFFT_EXPORT __declspec(dllexport)
#else
#define FINUFFT_EXPORT __declspec(dllimport)
#endif
#else
#define FINUFFT_EXPORT
#endif

/* specify calling convention (Windows only)
   The cdecl calling convention is actually not the default in all but a very
   few C/C++ compilers.
   If the user code changes the default compiler calling convention, may need
   this when generating DLL. */
#if defined(_WIN32) || defined(__WIN32__)
#define FINUFFT_CDECL __cdecl
#else
#define FINUFFT_CDECL
#endif

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

#include <finufft_errors.h>
#include <memory>
#include <xsimd/xsimd.hpp>

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday.
using BIGINT  = int64_t;
using UBIGINT = uint64_t;

// ------------- Library-wide algorithm parameter settings ----------------

// Library version (is a string)
#define FINUFFT_VER "2.3.1"

// Smallest possible kernel spread width per dimension, in fine grid points
// (used only in spreadinterp.cpp)
inline constexpr int MIN_NSPREAD = 2;

// Largest possible kernel spread width per dimension, in fine grid points
// (used only in spreadinterp.cpp)
inline constexpr int MAX_NSPREAD = 16;

// Fraction growth cut-off in utils:arraywidcen, sets when translate in type-3
inline constexpr double ARRAYWIDCEN_GROWFRAC = 0.1;

// Max number of positive quadr nodes for kernel FT (used only in common.cpp)
inline constexpr int MAX_NQUAD = 100;

// Internal (nf1 etc) array allocation size that immediately raises error.
// (Note: next235 takes 1s for 1e11, so it is also to prevent hang here.)
// Increase this if you need >10TB (!) RAM...
inline constexpr BIGINT MAX_NF = BIGINT(1e12);

// Maximum allowed number M of NU points; useful to catch incorrectly cast int32
// values for M = nj (also nk in type 3)...
inline constexpr BIGINT MAX_NU_PTS = BIGINT(1e14);

// We define our own PI here because M_PI is not actually part of standard C++
inline constexpr double PI      = 3.14159265358979329;
inline constexpr double INV_2PI = 0.159154943091895336;

// ----- OpenMP macros which also work when omp not present -----
// Allows compile-time switch off of openmp, so compilation without any openmp
// is done (Note: _OPENMP is automatically set by -fopenmp compile flag)
#ifdef _OPENMP
#include <omp.h>
// point to actual omp utils
static inline int MY_OMP_GET_NUM_THREADS [[maybe_unused]] () {
  return omp_get_num_threads();
}
static inline int MY_OMP_GET_MAX_THREADS [[maybe_unused]] () {
  return omp_get_max_threads();
}
static inline int MY_OMP_GET_THREAD_NUM [[maybe_unused]] () {
  return omp_get_thread_num();
}
static inline void MY_OMP_SET_NUM_THREADS [[maybe_unused]] (int x) {
  omp_set_num_threads(x);
}
#else
// non-omp safe dummy versions of omp utils...
static inline int MY_OMP_GET_NUM_THREADS [[maybe_unused]] () { return 1; }
static inline int MY_OMP_GET_MAX_THREADS [[maybe_unused]] () { return 1; }
static inline int MY_OMP_GET_THREAD_NUM [[maybe_unused]] () { return 0; }
static inline void MY_OMP_SET_NUM_THREADS [[maybe_unused]] (int) {}
#endif

#include <finufft/fft.h> // (must come after complex.h)
#include <finufft_opts.h>
#include <finufft_spread_opts.h>

// group together a bunch of type 3 rescaling/centering/phasing parameters:
template<typename T> struct type3params {
  T X1, C1, D1, h1, gam1; // x dim: X=halfwid C=center D=freqcen h,gam=rescale
  T X2, C2, D2, h2, gam2; // y
  T X3, C3, D3, h3, gam3; // z
};

template<typename TF> struct FINUFFT_PLAN_T { // the main plan class, fully C++

  using TC = std::complex<TF>;

  // These default and delete specifications just state the obvious,
  // but are here to silence compiler warnings.
  FINUFFT_PLAN_T(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans, TF tol,
                 finufft_opts *opts, int &ier);
  // Copy construction and assignent are already deleted implicitly
  // because of the unique_ptr member.
  FINUFFT_PLAN_T(const FINUFFT_PLAN_T &)            = delete;
  FINUFFT_PLAN_T &operator=(const FINUFFT_PLAN_T &) = delete;
  ~FINUFFT_PLAN_T();

  int type;                // transform type (Rokhlin naming): 1,2 or 3
  int dim;                 // overall dimension: 1,2 or 3
  int ntrans;              // how many transforms to do at once (vector or "many" mode)
  BIGINT nj;               // num of NU pts in type 1,2 (for type 3, num input x pts)
  BIGINT nk;               // number of NU freq pts (type 3 only)
  TF tol;                  // relative user tolerance
  int batchSize;           // # strength vectors to group together for FFTW, etc
  int nbatch;              // how many batches done to cover all ntrans vectors

  BIGINT ms;               // number of modes in x (1) dir (historical CMCL name) = N1
  BIGINT mt;               // number of modes in y (2) direction = N2
  BIGINT mu;               // number of modes in z (3) direction = N3
  BIGINT N;                // total # modes (prod of above three)

  BIGINT nf1 = 1;          // size of internal fine grid in x (1) direction
  BIGINT nf2 = 1;          // " y (2)
  BIGINT nf3 = 1;          // " z (3)
  BIGINT nf  = 1;          // total # fine grid points (product of the above three)

  int fftSign;             // sign in exponential for NUFFT defn, guaranteed to be +-1

  std::vector<TF> phiHat1; // FT of kernel in t1,2, on x-axis mode grid
  std::vector<TF> phiHat2; // " y-axis.
  std::vector<TF> phiHat3; // " z-axis.

  // fwBatch: (batches of) fine working grid(s) for the FFT to plan & act on.
  // Usually the largest internal array. Its allocator is 64-byte (cache-line) aligned:
  std::vector<TC, xsimd::aligned_allocator<TC, 64>> fwBatch;

  std::vector<BIGINT> sortIndices; // precomputed NU pt permutation, speeds spread/interp
  bool didSort;                    // whether binsorting used (false: identity perm used)

  // for t1,2: ptr to user-supplied NU pts (no new allocs).
  // for t3: will become ptr to internally allocated "primed" (scaled) Xp, Yp, Zp vecs.
  TF *X = nullptr, *Y = nullptr, *Z = nullptr;

  // type 3 specific
  TF *S = nullptr, *T = nullptr, *U = nullptr; // ptrs to user's target NU-point arrays
                                               // (no new allocs)
  std::vector<TC> prephase;                    // pre-phase, for all input NU pts
  std::vector<TC> deconv;     // reciprocal of kernel FT, phase, all output NU pts
  std::vector<TC> CpBatch;    // working array of prephased strengths
  std::vector<TF> Xp, Yp, Zp; // internal primed NU points (x'_j, etc)
  std::vector<TF> Sp, Tp, Up; // internal primed targs (s'_k, etc)
  type3params<TF> t3P;        // groups together type 3 shift, scale, phase, parameters
  std::unique_ptr<FINUFFT_PLAN_T<TF>> innerT2plan; // ptr used for type 2 in step 2 of
                                                   // type 3

  // other internal structs
  std::unique_ptr<Finufft_FFT_plan<TF>> fftPlan;
  finufft_opts opts; // this and spopts could be made ptrs
  finufft_spread_opts spopts;

  // Remaining actions (not create/delete) in guru interface are now methods...
  int setpts(BIGINT nj, TF *xj, TF *yj, TF *zj, BIGINT nk, TF *s, TF *t, TF *u);
  int execute(std::complex<TF> *cj, std::complex<TF> *fk);
};

void finufft_default_opts_t(finufft_opts *o);
template<typename TF>
int finufft_makeplan_t(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans,
                       TF tol, FINUFFT_PLAN_T<TF> **pp, finufft_opts *opts);

#endif // FINUFFT_CORE_H
