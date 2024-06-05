// -------- Public header for user-controllable FINUFFT options struct ---------
// Deliberately a plain C struct, without special types, or switchable prec's.
// See ../docs/devnotes.rst about what else to sync when you change this.

#ifndef FINUFFT_OPTS_H
#define FINUFFT_OPTS_H

typedef struct finufft_opts { // defaults see finufft.cpp:finufft_default_opts()
  // sphinx tag (don't remove): @opts_start
  // FINUFFT options:
  // data handling opts...
  int modeord; // (type 1,2 only): 0 CMCL-style increasing mode order
               //                  1 FFT-style mode order
  int chkbnds; // [DEPRECATED] 0 don't check NU pts in [-3pi,3pi), 1 do (<few % slower)

  // diagnostic opts...
  int debug;        // 0 silent, 1 some timing/debug, or 2 more
  int spread_debug; // spreader: 0 silent, 1 some timing/debug, or 2 tonnes
  int showwarn;     // 0 don't print warnings to stderr, 1 do

  // algorithm performance opts...
  int nthreads;           // number of threads to use, or 0 uses all available
  int fftw;               // plan flags to FFTW (FFTW_ESTIMATE=64, FFTW_MEASURE=0,...)
  int spread_sort;        // spreader: 0 don't sort, 1 do, or 2 heuristic choice
  int spread_kerevalmeth; // spreader: 0 exp(sqrt()), 1 Horner piecewise poly (faster)
  int spread_kerpad;      // (exp(sqrt()) only): 0 don't pad kernel to 4n, 1 do
  double upsampfac;       // upsampling ratio sigma: 2.0 std, 1.25 small FFT, 0.0 auto
  int spread_thread;      // (vectorized ntr>1 only): 0 auto, 1 seq multithreaded,
                          //                          2 parallel single-thread spread
  int maxbatchsize;       // (vectorized ntr>1 only): max transform batch, 0 auto
  int spread_nthr_atomic; // if >=0, threads above which spreader OMP critical goes
                          // atomic
  int spread_max_sp_size; // if >0, overrides spreader (dir=1) max subproblem size
                          // sphinx tag (don't remove): @opts_end
} finufft_opts;

// Those of the above of the form spread_* indicate pass through to finufft_spread_opts

// define deprecated opts macro
#if defined(__cplusplus) && (__cplusplus >= 201402L)
#define DEPRECATED_OPTS                                                          \
  [[deprecated("as of v2.1.0, nufft_opts is obsolete and renamed finufft_opts; " \
               "please use this instead.")]]
#elif defined(_MSC_VER)
#define DEPRECATED_OPTS                                                     \
  __declspec(deprecated("as of v2.1.0, nufft_opts is obsolete and renamed " \
                        "finufft_opts; please use this instead."))
#else
#define DEPRECATED_OPTS                                                         \
  __attribute__((deprecated("as of v2.1.0, nufft_opts is obsolete and renamed " \
                            "finufft_opts; please use this instead.")))
#endif

// Backwards-compatibility
DEPRECATED_OPTS typedef finufft_opts nufft_opts;

#endif // FINUFFT_OPTS_H
