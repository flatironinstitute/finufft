// This contains some library-wide definitions & precision/OMP switches,
// as well as the interfaces to utilities in utils.cpp. Barnett 6/18/18.

#ifndef UTILS_H
#define UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

#include <complex>          // C++ type complex
#include <cuComplex.h>
#include "dataTypes.h"

// fraction growth cut-off in arraywidcen(), to decide if translate in type-3
#define ARRAYWIDCEN_GROWFRAC 0.1

// math consts not in math.h ...
#define M_1_2PI 0.159154943091895336
#define M_2PI   6.28318530717958648
// to avoid mixed precision operators in eg i*pi...
#define PI (FLT)M_PI

using namespace std;        // means std:: not needed for cout, max, etc

typedef complex<double> dcomplex;  // slightly sneaky since duplicated by mwrap

// Global error codes for the library...
#define WARN_EPS_TOO_SMALL       1
#define ERR_MAXNALLOC            2
#define ERR_SPREAD_BOX_SMALL     3
#define ERR_SPREAD_PTS_OUT_RANGE 4
#define ERR_SPREAD_ALLOC         5
#define ERR_SPREAD_DIR           6
#define ERR_UPSAMPFAC_TOO_SMALL  7
#define HORNER_WRONG_BETA        8
#define ERR_NDATA_NOTVALID       9


//#define MAX(a,b) (a>b) ? a : b  // but we use std::max instead
#define MIN(a,b) (a<b) ? a : b

// ahb math helpers
BIGINT next235beven(BIGINT n, BIGINT b);

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
  // non-omp safe dummy versions of omp utils
  #define MY_OMP_GET_NUM_THREADS() 1
  #define MY_OMP_GET_MAX_THREADS() 1
  #define MY_OMP_GET_THREAD_NUM() 0
  #define MY_OMP_SET_NUM_THREADS(x)
  #define MY_OMP_SET_NESTED(x)
#endif

#endif  // UTILS_H
