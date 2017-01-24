#ifndef UTILS_H
#define UTILS_H

#include <sys/time.h>

using namespace std;

#include <complex>          // C++ type complex
#define dcomplex complex<double>
#define ima complex<double>{0.0,1.0}


class CNTime {
 public:
  void start();
  int restart();
  int elapsed();
  double elapsedsec();
 private:
  struct timeval initial;
};

// crappy random number generator in [0,1)
//#define rand01() (((double)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((double)rand()/RAND_MAX)

// allow compile-time switch off of openmp, so compilation without any openmp
// is done (Note: _OPENMP is automatically set by -fopenmp compile flag)
#ifdef _OPENMP
  #include <omp.h>
#else
//  hack to handle all the omp commands we use...
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
  #define omp_get_thread_num() 0
  #define omp_set_num_threads(x)
//#define fftw_init_threads()        // fails, clash with fftw3.h
//#define fftw_plan_with_nthreads(x) // "
#endif

#endif  // UTILS_H
