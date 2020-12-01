#ifndef SPREAD_OPTS_H
#define SPREAD_OPTS_H

#include <dataTypes.h>

// C-compatible options struct for spreader.
// (mostly internal to spreadinterp.cpp, with a little bleed to finufft.cpp)

typedef struct spread_opts {  // see spreadinterp:setup_spreader for defaults.
  // This is the main documentation for these options...
  int nspread;            // w, the kernel width in grid pts
  int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange;            // 0: NU periodic domain is [0,N), 1: domain [-pi,pi)
  int chkbnds;            // 0: don't check NU pts in 3-period range; 1: do
  int sort;               // 0: don't sort NU pts, 1: do, 2: heuristic choice
  int kerevalmeth;        // 0: direct exp(sqrt()), or 1: Horner ppval, fastest
  int kerpad;             // 0: no pad w to mult of 4, 1: do pad
                          // (this helps SIMD for kerevalmeth=0, eg on i7).
  int nthreads;           // # threads for spreadinterp (0: use max avail)
  int sort_threads;       // # threads for sort (0: auto-choice up to nthreads)
  int max_subproblem_size; // # pts per t1 subprob; sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans
                          // if changed from 0!). See spreadinterp.h
  int debug;              // 0: silent, 1: small text output, 2: verbose
  int atomic_threshold;   // num threads before switching spreadSorted to using atomic ops
  double upsampfac;       // sigma, upsampling factor
  // ES kernel specific consts used in fast eval, depend on precision FLT...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
} spread_opts;

#endif   // SPREAD_OPTS_H
