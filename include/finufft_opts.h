// -------- Public header for user-controllable FINUFFT options struct ---------
// Deliberately a plain C struct, without special types, or switchable prec's.
// See ../docs/devnotes.rst about what else to sync when you change this.

#ifndef FINUFFT_OPTS_H
#define FINUFFT_OPTS_H

typedef struct finufft_opts { // defaults see finufft_core.cpp:finufft_default_opts_t()
  // sphinx tag (don't remove): @opts_start
  // FINUFFT options:
  // data handling opts...
  int modeord;          // (type 1,2 only): 0 CMCL-style increasing mode order
                        //                  1 FFT-style mode order
  int spreadinterponly; // (type 1,2 only): 0 do actual NUFFT
                        // 1 only spread (if type 1) or interpolate (type 2)

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

  // User can provide their own FFTW planner lock functions for thread safety
  // Null values ignored and use a default lock function (both or neither must be set)
  void (*fftw_lock_fun)(void *);   // Function ptr that locks the FFTW planner
  void (*fftw_unlock_fun)(void *); // Function ptr that unlocks the FFTW planner
  void *fftw_lock_data;            // Data to pass to the lock functions (e.g. a mutex)
} finufft_opts;

// Those of the above of the form spread_* indicate pass through to finufft_spread_opts

#endif // FINUFFT_OPTS_H
