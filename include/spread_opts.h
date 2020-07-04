#ifndef SPREAD_OPTS_H
#define SPREAD_OPTS_H

#include <dataTypes.h>

// C-compatible options struct for spreader
// (mostly internal to spreadinterp.cpp, with a little bleed to finufft.cpp)

typedef struct spread_opts {    // see cnufftspread:setup_spreader for defaults.
  int nspread;            // w, the kernel width in grid pts
  int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange;            // 0: coords in [0,N), 1 coords in [-pi,pi)
  int chkbnds;            // 0: don't check NU pts are in range; 1: do
  int sort;               // 0: don't sort NU pts, 1: do, 2: heuristic choice
  int kerevalmeth;        // 0: exp(sqrt()), old, or 1: Horner ppval, fastest
  int kerpad;             // 0: no pad to mult of 4, 1: do (helps i7 kereval=0)
  int sort_threads;       // 0: auto-choice, >0: fix number of sort threads
  BIGINT max_subproblem_size; // sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans!)
  int debug;              // 0: silent, 1: small text output, 2: verbose
  FLT upsampfac;          // sigma, upsampling factor, default 2.0
  // ES kernel specific...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
} spread_opts;

#endif   // SPREAD_OPTS_H
