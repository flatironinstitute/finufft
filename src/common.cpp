#include "common.h"

void onedim_dct_kernel(BIGINT nf, double *fwkerhalf,
		       double &prefac_unused_dim, spread_opts opts)
/*
  Use DCT in FFTW for Fourier coeffs of cnufftspread's real, symmetric kernel.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  fwkerhalf - should be allocated for at least nf/2+1 doubles.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier coeffs from indices 0 to nf/2 inclusive.
  prefac_unused_dim - the prefactor that cnufftspread multiplies for each
                       unused dimension (ie two such factors in 1d, one in 2d,
		       and none in 3d).
  Barnett 1/23/17
 */
{
  // note: in-place, and we make plan before filling input array...
  fftw_plan p = fftw_plan_r2r_1d(nf/2+1,fwkerhalf,fwkerhalf,FFTW_REDFT00,
				 FFTW_ESTIMATE);  // note no fftsign
  for (BIGINT i=0; i<=nf/2; ++i)
    fwkerhalf[i] = 0.0;   // zero it
  for (int i=0; i<=opts.nspread/2; ++i)    // i is dist from kernel origin
    fwkerhalf[nf/2-i] = evaluate_kernel((double)i, opts);  // center at nf/2
  prefac_unused_dim = fwkerhalf[nf/2];  // must match cnufftspread's behavior
  fftw_execute(p);
  fftw_destroy_plan(p);
}

