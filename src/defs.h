// Library internal definitions. Eg, prec switch, complex type, various macros.
// Only need work in C++.
// Split out by Joakim Anden, Alex Barnett 9/20/18-9/24/18.

#ifndef DEFS_H
#define DEFS_H

// define types intrinsic to finufft interface (FLT, CPX, BIGINT, etc):
#include "finufft.h"


// ------------- Library-wide algorithm parameter settings ----------------

// Largest possible kernel spread width per dimension, in fine grid points
// (used only in spreadinterp.cpp)
#define MAX_NSPREAD 16

// Fraction growth cut-off in utils:arraywidcen, sets when translate in type-3
#define ARRAYWIDCEN_GROWFRAC 0.1

// Max number of positive quadr nodes for kernel FT (used only in common.cpp)
#define MAX_NQUAD 100

// Internal (nf1 etc) array allocation size that immediately raises error.
// (Note: next235 takes 1s for this size.)
// Increase this if you need >1TB RAM... (used only in common.cpp)
#define MAX_NF    (BIGINT)1e11



// ---------- Global error output codes for the library -----------------------
// (it could be argued these belong in finufft.h, but to avoid polluting
//  user's name space we keep them here)
#define ERR_EPS_TOO_SMALL        1
#define ERR_MAXNALLOC            2
#define ERR_SPREAD_BOX_SMALL     3
#define ERR_SPREAD_PTS_OUT_RANGE 4
#define ERR_SPREAD_ALLOC         5
#define ERR_SPREAD_DIR           6
#define ERR_UPSAMPFAC_TOO_SMALL  7
#define HORNER_WRONG_BETA        8
#define ERR_NDATA_NOTVALID       9



// -------------- Math consts (not in math.h) and useful math macros ----------

// prec-indep unit imaginary number
#define IMA std::complex<FLT>(0.0,1.0)
#define M_1_2PI 0.159154943091895336
#define M_2PI   6.28318530717958648
// to avoid mixed precision operators in eg i*pi...
#define PI (FLT)M_PI

using namespace std;        // means std:: not needed for cout, max, etc

#define MAX(a,b) (a>b) ? a : b  // but we use std::max instead
#define MIN(a,b) (a<b) ? a : b

// Random numbers: crappy unif random number generator in [0,1):
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



// ---------- Precision-indep complex types, macros to FFTW -------------------

#include <fftw3.h>          // (after complex.h) needed so can typedef FFTW_CPX

// prec-indep interfaces to FFTW and other math utilities...
#ifdef SINGLE
  // single-prec, machine epsilon for rounding
  #define EPSILON (float)6e-08
  #define FABS(x) fabsf(x)
  typedef fftwf_complex FFTW_CPX;           //  single-prec has fftwf_*
  typedef fftwf_plan FFTW_PLAN;
  #define FFTW_INIT fftwf_init_threads
  #define FFTW_PLAN_TH fftwf_plan_with_nthreads
  #define FFTW_ALLOC_RE fftwf_alloc_real
  #define FFTW_ALLOC_CPX fftwf_alloc_complex
  #define FFTW_PLAN_1D fftwf_plan_dft_1d
  #define FFTW_PLAN_2D fftwf_plan_dft_2d
  #define FFTW_PLAN_3D fftwf_plan_dft_3d
  #define FFTW_PLAN_MANY_DFT fftwf_plan_many_dft
  #define FFTW_EX fftwf_execute
  #define FFTW_DE fftwf_destroy_plan
  #define FFTW_FR fftwf_free
  #define FFTW_FORGET_WISDOM fftwf_forget_wisdom
#else
  // double-precision, machine epsilon for rounding
  #define EPSILON (double)1.1e-16
  #define FABS(x) fabs(x)
  typedef fftw_complex FFTW_CPX;           // double-prec has fftw_*
  typedef fftw_plan FFTW_PLAN;
  #define FFTW_INIT fftw_init_threads
  #define FFTW_PLAN_TH fftw_plan_with_nthreads
  #define FFTW_ALLOC_RE fftw_alloc_real
  #define FFTW_ALLOC_CPX fftw_alloc_complex
  #define FFTW_PLAN_1D fftw_plan_dft_1d
  #define FFTW_PLAN_2D fftw_plan_dft_2d
  #define FFTW_PLAN_3D fftw_plan_dft_3d
  #define FFTW_PLAN_MANY_DFT fftw_plan_many_dft
  #define FFTW_EX fftw_execute
  #define FFTW_DE fftw_destroy_plan
  #define FFTW_FR fftw_free
  #define FFTW_FORGET_WISDOM fftw_forget_wisdom
#endif



// ----------- OpenMP (and FFTW omp) macros which work even in single-core ----

// Allows compile-time switch off of openmp, so compilation without any openmp
// is done (Note: _OPENMP is automatically set by -fopenmp compile flag)
#ifdef _OPENMP
  #include <omp.h>
  // point to actual omp utils
  #define MY_OMP_GET_NUM_THREADS() omp_get_num_threads()
  #define MY_OMP_GET_MAX_THREADS() omp_get_max_threads()
  #define MY_OMP_GET_THREAD_NUM() omp_get_thread_num()
  #define MY_OMP_SET_NUM_THREADS(x) omp_set_num_threads(x)
  #define MY_OMP_SET_NESTED(x) omp_set_nested(x)
#else
  // non-omp safe dummy versions of omp utils, and dummy fftw threads calls...
  #define MY_OMP_GET_NUM_THREADS() 1
  #define MY_OMP_GET_MAX_THREADS() 1
  #define MY_OMP_GET_THREAD_NUM() 0
  #define MY_OMP_SET_NUM_THREADS(x)
  #define MY_OMP_SET_NESTED(x)
  #undef FFTW_INIT
  #define FFTW_INIT()
  #undef FFTW_PLAN_TH
  #define FFTW_PLAN_TH(x)
#endif

#endif  // DEFS_H
