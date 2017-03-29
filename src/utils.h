#ifndef UTILS_H
#define UTILS_H

using namespace std;        // means std:: not needed for cout, max, etc
#include <complex>          // C++ type complex, and useful abbrevs...
typedef complex<double> dcomplex;  // slightly sneaky since duplicated by mwrap
#define ima complex<double>{0.0,1.0}

// Compile-flag choice of 64 (default) or 32 bit integers in interface
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

// internal integers needed for figuring array sizes, regardless of BIGINT
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
double relerrtwonorm(BIGINT n, dcomplex* a, dcomplex* b);
double errtwonorm(BIGINT n, dcomplex* a, dcomplex* b);
double twonorm(BIGINT n, dcomplex* a);
double infnorm(BIGINT n, dcomplex* a);
void arrayrange(BIGINT n, double* a, double *lo, double *hi);
void arraywidcen(BIGINT n, double* a, double *w, double *c);
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
//#define rand01() (((double)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((double)rand()/RAND_MAX)
// unif[-1,1]:
#define randm11() (2*rand01() - 1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11() (randm11() + ima*randm11())

// Thread-safe seed-carrying versions of above (x is ptr to seed)...
#define rand01r(x) ((double)rand_r(x)/RAND_MAX)
// unif[-1,1]:
#define randm11r(x) (2*rand01r(x) - 1.0)
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
