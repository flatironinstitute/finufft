#if (!defined(SPREADINTERP_H) && !defined(SINGLE)) || \
  (!defined(SPREADINTERPF_H) && defined(SINGLE))

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "utils_fp.h"

#define MAX_NSPREAD 16     // upper bound on w, ie nspread, even when padded
                           // (see evaluate_kernel_vector); also for common

#undef SPREAD_OPTS

#ifdef SINGLE
#define SPREAD_OPTS spread_optsf
#define SPREADINTERPF_H
#else
#define SPREAD_OPTS spread_opts
#define SPREADINTERP_H
#endif

struct SPREAD_OPTS {      // see cnufftspread:setup_spreader for defaults.
  int nspread;            // w, the kernel width in grid pts
  int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange;            // 0: coords in [0,N), 1 coords in [-pi,pi)
  FLT upsampfac;          // sigma, upsampling factor, default 2.0
  // ES kernel specific...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
};

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
#define RESCALE(x,N,p) (p ? \
		     ((x*M_1_2PI + (x<-PI ? 1.5 : (x>=PI ? -0.5 : 0.5)))*N) : \
		     (x<0 ? x+N : (x>=N ? x-N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.
FLT evaluate_kernel(FLT x, const SPREAD_OPTS &opts);
int setup_spreader(SPREAD_OPTS &opts, FLT eps, FLT upsampfac, int kerevalmeth);

#endif  // SPREADINTERP_H
