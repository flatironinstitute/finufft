#ifndef UTILS_H
#define UTILS_H

using namespace std;        // means std:: not needed for cout, max, etc
#include <complex>          // C++ type complex, and useful abbrevs...
#define dcomplex complex<double>
#define ima complex<double>{0.0,1.0}

// choose int64_t or long long if want handle huge I/O array sizes (>2^31)...
//#define BIGINT int64_t
#define BIGINT long

#define MAX(a,b) (a>b) ? a : b  // but we use std::max instead
#define MIN(a,b) (a<b) ? a : b

// ahb math helpers
double relerrtwonorm(BIGINT n, dcomplex* a, dcomplex* b);
double errtwonorm(BIGINT n, dcomplex* a, dcomplex* b);
double twonorm(BIGINT n, dcomplex* a);
double infnorm(BIGINT n, dcomplex* a);
void arrayrange(BIGINT n, double* a, double &lo, double &hi);
void arraywidcen(BIGINT n, double* a, double &w, double &c);
BIGINT next235even(BIGINT n);

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

// crappy unif random number generator in [0,1):
//#define rand01() (((double)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((double)rand()/RAND_MAX)
// unif[-1,1]:
#define randm11() (2*rand01() - 1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11() (randm11() + ima*randm11())

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
  // non-omp safe versions of utils
  #define MY_OMP_GET_NUM_THREADS() 1
  #define MY_OMP_GET_MAX_THREADS() 1
  #define MY_OMP_GET_THREAD_NUM() 0
  #define MY_OMP_SET_NUM_THREADS(x)
#endif

#endif  // UTILS_H
