#ifndef UTILS_H
#define UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

#include <complex>          // C++ type complex
#include <fftw3.h>          // needed so can typedef FFTW_CPX


// fraction growth cut-off in arraywidcen()
#define ARRAYWIDCEN_GROWFRAC 0.1

// math consts not in math.h ...
#define M_1_2PI 0.159154943091895336
#define M_2PI   6.28318530717958648
// to avoid mixed precision operators in eg i*pi...
#define PI (FLT)M_PI

using namespace std;        // means std:: not needed for cout, max, etc

typedef complex<double> dcomplex;  // slightly sneaky since duplicated by mwrap

// Compile-flag choice of single or double (default) precision:
// (Note in the other codes, FLT is "double" or "float", CPX same but complex)
#ifdef SINGLE
  // machine epsilon for rounding
  #define EPSILON (float)6e-08
  typedef float FLT;
  typedef complex<float> CPX;
#define ima complex<float>(0.0,1.0)
  #define FABS(x) fabs(x)
  typedef fftwf_complex FFTW_CPX;           //  single-prec has fftwf_*
  typedef fftwf_plan FFTW_PLAN;
  #define FFTW_INIT fftwf_init_threads
  #define FFTW_PLAN_TH fftwf_plan_with_nthreads
  #define FFTW_ALLOC_RE fftwf_alloc_real
  #define FFTW_ALLOC_CPX fftwf_alloc_complex
  #define FFTW_PLAN_1D fftwf_plan_dft_1d
  #define FFTW_PLAN_2D fftwf_plan_dft_2d
  #define FFTW_PLAN_3D fftwf_plan_dft_3d
  #define FFTW_EX fftwf_execute
  #define FFTW_DE fftwf_destroy_plan
  #define FFTW_FR fftwf_free
#else
  // machine epsilon for rounding
  #define EPSILON (double)1.1e-16
  typedef double FLT;
  typedef complex<double> CPX;
#define ima complex<double>(0.0,1.0)
  #define FABS(x) fabsf(x)
  typedef fftw_complex FFTW_CPX;           // double-prec has fftw_*
  typedef fftw_plan FFTW_PLAN;
  #define FFTW_INIT fftw_init_threads
  #define FFTW_PLAN_TH fftw_plan_with_nthreads
  #define FFTW_ALLOC_RE fftw_alloc_real
  #define FFTW_ALLOC_CPX fftw_alloc_complex
  #define FFTW_PLAN_1D fftw_plan_dft_1d
  #define FFTW_PLAN_2D fftw_plan_dft_2d
  #define FFTW_PLAN_3D fftw_plan_dft_3d
  #define FFTW_EX fftw_execute
  #define FFTW_DE fftw_destroy_plan
  #define FFTW_FR fftw_free
#endif

// Compile-flag choice of 64 (default) or 32 bit integers in interface:
#ifdef INTERFACE32
  typedef int INT;
#else
  typedef int64_t INT;
#endif

// Compile flag choice of 64 (default) or 32 bit internal integer indexing:
#ifdef SMALLINT
// It is possible that on some systems it will run faster, if you only care
// about small arrays, to index via 32-bit integers...
  typedef int BIGINT;
#else
// since "long" not guaranteed to exceed 32 bit signed int, ie 2^31,
// use long long (64-bit) since want handle huge array sizes (>2^31)...
typedef int64_t BIGINT;
#endif

// internal integers needed for figuring array sizes, regardless of INT, BIGINT
typedef int64_t INT64;


// global error codes for the library...
#define ERR_EPS_TOO_SMALL        1
#define ERR_MAXNALLOC            2
#define ERR_SPREAD_BOX_SMALL     3
#define ERR_SPREAD_PTS_OUT_RANGE 4
#define ERR_SPREAD_ALLOC         5
#define ERR_SPREAD_DIR           6


#define MAX(a,b) (a>b) ? a : b  // but we use std::max instead
#define MIN(a,b) (a<b) ? a : b

// ahb math helpers
FLT relerrtwonorm(BIGINT n, CPX* a, CPX* b);
FLT errtwonorm(BIGINT n, CPX* a, CPX* b);
FLT twonorm(BIGINT n, CPX* a);
FLT infnorm(BIGINT n, CPX* a);
void arrayrange(BIGINT n, FLT* a, FLT *lo, FLT *hi);
void indexedarrayrange(BIGINT n, BIGINT* i, FLT* a, FLT *lo, FLT *hi);
void arraywidcen(BIGINT n, FLT* a, FLT *w, FLT *c);
INT64 next235even(INT64 n);


// jfm timer stuff
#include <sys/time.h>
class CNTime {
 public:
  void start();
  int restart();
  int elapsed();
  double elapsedsec();
 private:
  struct timeval initial;
};


// Random numbers: crappy unif random number generator in [0,1):
//#define rand01() (((FLT)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((FLT)rand()/RAND_MAX)
// unif[-1,1]:
#define randm11() (2*rand01() - (FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11() (randm11() + ima*randm11())

// Thread-safe seed-carrying versions of above (x is ptr to seed)...
#define rand01r(x) ((FLT)rand_r(x)/RAND_MAX)
// unif[-1,1]:
#define randm11r(x) (2*rand01r(x) - (FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11r(x) (randm11r(x) + ima*randm11r(x))


// allow compile-time switch off of openmp, so compilation without any openmp
// is done (Note: _OPENMP is automatically set by -fopenmp compile flag)
#ifdef _OPENMP
  #include <omp.h>
  // point to actual omp utils
  #define MY_OMP_GET_NUM_THREADS() omp_get_num_threads()
  #define MY_OMP_GET_MAX_THREADS() omp_get_max_threads()
  #define MY_OMP_GET_THREAD_NUM() omp_get_thread_num()
  #define MY_OMP_SET_NUM_THREADS(x) omp_set_num_threads(x)
#else
  // non-omp safe dummy versions of omp utils
  #define MY_OMP_GET_NUM_THREADS() 1
  #define MY_OMP_GET_MAX_THREADS() 1
  #define MY_OMP_GET_THREAD_NUM() 0
  #define MY_OMP_SET_NUM_THREADS(x)
#endif

#endif  // UTILS_H
