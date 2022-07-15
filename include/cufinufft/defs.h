#ifndef CUFINUFFT_DEFS_H
#define CUFINUFFT_DEFS_H

#include <cufinufft_types.h>

// constants needed within common
// upper bound on w, ie nspread, even when padded (see evaluate_kernel_vector); also for common
#define MAX_NSPREAD 16

// max number of positive quadr nodes
#define MAX_NQUAD 100 

// In cufinufft we limit array sizes to 2^31 which is about 2 billion, since we set
// CUFINUFFT_BIGINT to int. (Differs from FINUFFT)
#define MAX_NF (CUFINUFFT_BIGINT) INT_MAX

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


#endif
