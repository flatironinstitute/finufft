#include "finufft_errors.h"
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif

#include <stdint.h>

#include <cufinufft.h>

// Two ways to exceed MAX_NF: a single dimension whose upsampled nf itself overflows
// CUFINUFFT_BIGINT (int32), and a 3D grid whose per-dim nf each fit int32 but whose
// product doesn't. Both must reach the makeplan/setpts guards (issue #890).
int main() {
  const int type      = 1;
  const int iflag     = 1;
  const int ntransf   = 1;
  const int64_t N1[3] = {1200000000, 1, 1}; // upsampfac*N1 alone overflows int32
  const int64_t N3[3] = {700, 700, 700};    // ms*mt*mu fits int32; nf1*nf2*nf3 does not

  {
    cufinufftf_plan plan;
    assert(cufinufftf_makeplan(type, 1, N1, iflag, ntransf, 1e-5f, &plan, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    cufinufft_plan plan;
    assert(cufinufft_makeplan(type, 1, N1, iflag, ntransf, 1e-5, &plan, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    cufinufftf_plan plan;
    assert(cufinufftf_makeplan(type, 3, N3, iflag, ntransf, 1e-5f, &plan, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  {
    cufinufft_plan plan;
    assert(cufinufft_makeplan(type, 3, N3, iflag, ntransf, 1e-5, &plan, NULL) ==
           FINUFFT_ERR_MAXNALLOC);
  }
  return 0;
}
