#include "common.h"

#define HALF_MAX_NS 16       // upper bnd on nspread/2

void onedim_dct_kernel(BIGINT nf, double *fwkerhalf,
		       double &prefac_unused_dim, spread_opts opts)
/*
  Computes DCT coeffs of cnufftspread's real symmetric kernel, directly,
  exploiting narrowness of kernel.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  fwkerhalf - should be allocated for at least nf/2+1 doubles.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier coeffs from indices 0 to nf/2 inclusive.
  prefac_unused_dim - the prefactor that cnufftspread multiplies for each
                       unused dimension (ie two such factors in 1d, one in 2d,
		       and none in 3d).
  Barnett 1/24/17
  todo: understand how to openmp it - subtle since private aj's. Want to break
        up fwkerhalf into contiguous pieces, one per thread.
 */
{
  int m=opts.nspread/2;        // how many modes in include
  double f[HALF_MAX_NS];
  for (int n=0;n<=m;++n)    // actual freq index will be nf/2-n, for cosines
    f[n] = evaluate_kernel((double)n, opts);  // center at nf/2
  prefac_unused_dim = f[0];   // ker @ 0, must match cnufftspread's behavior
  for (int n=1;n<=m;++n)    //  convert from exp to cosine ampls
    f[n] *= 2.0;
  dcomplex a[HALF_MAX_NS],aj[HALF_MAX_NS];
  for (int n=0;n<=m;++n) {    // set up our rotating phase array...
    a[n] = exp(2*M_PI*ima*(double)(nf/2-n)/(double)nf);   // phase differences
    aj[n] = dcomplex{1.0,0.0};       // init phase factors
  }
  for (BIGINT j=0;j<=nf/2;++j) {       // loop along output array
    double x = 0.0;                 // register
    for (int n=0;n<=m;++n) {
      x += f[n] * real(aj[n]);         // only want cosine part
      aj[n] *= a[n];       // wind the phases
    }
    fwkerhalf[j] = x;
  }
}

//void deconvolveshuffle
