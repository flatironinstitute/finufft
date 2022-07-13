#if (!defined(SPREADINTERP_H) && !defined(CUFINUFFT_SINGLE)) || \
  (!defined(SPREADINTERPF_H) && defined(CUFINUFFT_SINGLE))

#include <cmath>
#include <cufinufft_types.h>

#define MAX_NSPREAD 16     // upper bound on w, ie nspread, even when padded
                           // (see evaluate_kernel_vector); also for common

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
#define RESCALE(x,N,p) (p ? \
		     ((x*M_1_2PI + (x<-PI ? 1.5 : (x>=PI ? -0.5 : 0.5)))*N) : \
		     (x<0 ? x+N : (x>=N ? x-N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.
CUFINUFFT_FLT evaluate_kernel(CUFINUFFT_FLT x, const SPREAD_OPTS &opts);
int setup_spreader(SPREAD_OPTS &opts, CUFINUFFT_FLT eps, CUFINUFFT_FLT upsampfac, int kerevalmeth);

#endif  // SPREADINTERP_H
