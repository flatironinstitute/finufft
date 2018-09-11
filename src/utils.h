// This contains some library-wide definitions & precision/OMP switches,
// as well as the interfaces to utilities in utils.cpp. Barnett 6/18/18.

#ifndef UTILS_H
#define UTILS_H

#include "defs.h"

// fraction growth cut-off in arraywidcen(), to decide if translate in type-3
#define ARRAYWIDCEN_GROWFRAC 0.1

// math consts not in math.h ...
#define M_1_2PI 0.159154943091895336
#define M_2PI   6.28318530717958648
// to avoid mixed precision operators in eg i*pi...
#define PI (FLT)M_PI

using namespace std;        // means std:: not needed for cout, max, etc

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
BIGINT next235even(BIGINT n);


// jfm timer class
#include <sys/time.h>
class CNTime {
 public:
  void start();
  double restart();
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
#define crandm11() (randm11() + IMA*randm11())

// Thread-safe seed-carrying versions of above (x is ptr to seed)...
#define rand01r(x) ((FLT)rand_r(x)/RAND_MAX)
// unif[-1,1]:
#define randm11r(x) (2*rand01r(x) - (FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11r(x) (randm11r(x) + IMA*randm11r(x))


// allow compile-time switch off of openmp, so compilation without any openmp
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

#endif  // UTILS_H
