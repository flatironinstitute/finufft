#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

#include <complex>
// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <cstdint>
#include <cuComplex.h>

#include <cufinufft_types.h>


// fraction growth cut-off in arraywidcen(), to decide if translate in type-3
#define ARRAYWIDCEN_GROWFRAC 0.1

// math consts not in math.h ...
#define M_1_2PI 0.159154943091895336
#define M_2PI 6.28318530717958648
// to avoid mixed precision operators in eg i*pi...
#define PI (CUFINUFFT_FLT) M_PI

// Global error codes for the library...
#define WARN_EPS_TOO_SMALL 1
#define ERR_MAXNALLOC 2
#define ERR_SPREAD_BOX_SMALL 3
#define ERR_SPREAD_PTS_OUT_RANGE 4
#define ERR_SPREAD_ALLOC 5
#define ERR_SPREAD_DIR 6
#define ERR_UPSAMPFAC_TOO_SMALL 7
#define HORNER_WRONG_BETA 8
#define ERR_NDATA_NOTVALID 9

//#define MAX(a,b) (a>b) ? a : b  // but we use std::max instead
#define MIN(a, b) (a < b) ? a : b

// ahb math helpers
CUFINUFFT_BIGINT next235beven(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT b);

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


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN
        // (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

template <typename T>
T infnorm(int n, std::complex<T> *a) {
    T nrm = 0.0;
    for (int m = 0; m < n; ++m) {
        T aa = real(conj(a[m]) * a[m]);
        if (aa > nrm)
            nrm = aa;
    }
    return sqrt(nrm);
}


#endif
