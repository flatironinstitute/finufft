#ifndef FINUFFT_SPREAD_OPTS_H
#define FINUFFT_SPREAD_OPTS_H

// C-compatible options struct for spread/interpolation within FINUFFT

// Notes: 1) Has to be part of public-facing
// headers since finufft_plan has an instance of this spread_opts struct.
// 2) Deliberately uses fixed types (no macro precision-switching).

typedef struct finufft_spread_opts {
  // See spreadinterp:setup_spreader for default values of the following fields.
  // This is the main documentation for these options...
  int nspread;             // w, the kernel width in grid pts
  int spread_direction;    // 1 means spread NU->U, 2 means interpolate U->NU
  int chkbnds;             // [DEPRECATED] 0: don't check NU pts in 3-period range; 1: do
  int sort;                // 0: don't sort NU pts, 1: do, 2: heuristic choice
  int kerevalmeth;         // 0: direct exp(sqrt()), or 1: Horner ppval, fastest
  int kerpad;              // 0: no pad w to mult of 4, 1: do pad
                           // (this helps SIMD for kerevalmeth=0, eg on i7).
  int nthreads;            // # threads for spreadinterp (0: use max avail)
  int sort_threads;        // # threads for sort (0: auto-choice up to nthreads)
  int max_subproblem_size; // # pts per t1 subprob; sets extra RAM per thread
  int flags;               // binary flags for timing only (may give wrong ans
                           // if changed from 0!). See spreadinterp.h
  int debug;               // 0: silent, 1: small text output, 2: verbose
  int atomic_threshold; // num threads before switching spreadSorted to using atomic ops
  double upsampfac;     // sigma, upsampling factor
  // ES kernel specific consts for eval. No longer FLT, to avoid name clash...
  double ES_beta;
  double ES_halfwidth;
  double ES_c;
} finufft_spread_opts;

#endif // FINUFFT_SPREAD_OPTS_H
