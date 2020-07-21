#ifndef OPTS_H
#define OPTS_H

// ------------------- Struct for user-controllable options ------------------
// Deliberately a plain C struct, without special types

typedef struct nufft_opts{    // defaults see finufft.ccp:finufft_default_opts()
  int debug;          // 0: silent, 1: text basic timing output
  int spread_debug;   // passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)
  int spread_sort;    // passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)
  int spread_kerevalmeth; // "     spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)
  int spread_kerpad;  // passed to spread_opts, 0: don't pad to mult of 4, 1: do
  int chkbnds;        // 0: don't check if input NU pts in [-3pi,3pi], 1: do
  int fftw;           // 0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan but faster)
  int modeord;        // 0: CMCL-style increasing mode ordering (neg to pos), or
                      // 1: FFT-style mode ordering (affects type-1,2 only)
  double upsampfac;   // upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)
  int spread_thread;  // mode for ntrans>1 only. 0:auto, 1 sequential multithreaded, 2 parallel singlethreaded
  int maxbatchsize;   // for ntrans>1 only. max blocking size for vectorized, 0 for auto-set
  int showwarn;       // 0: don't print warnings to stderr; 1: do
  int nthreads;       // max number of threads to use, or 0: use all available
} nufft_opts;

#endif  // OPTS_H
