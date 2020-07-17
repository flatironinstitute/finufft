#include "common.h"
#include <cufinufft_eitherprec.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
  #include "legendre_rule_fast.h"
}
#else
  #include "legendre_rule_fast.h"
#endif

int setup_spreader_for_nufft(SPREAD_OPTS &spopts, FLT eps, cufinufft_opts opts)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Report status of setup_spreader.  Barnett 10/30/17
{
  int ier=setup_spreader(spopts, eps, opts.upsampfac, opts.gpu_kerevalmeth);
  spopts.pirange = 1;                 // could allow user control?
  return ier;
}

void SET_NF_TYPE12(BIGINT ms, cufinufft_opts opts, SPREAD_OPTS spopts,
				   BIGINT *nf, BIGINT bs)
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms.
{
  *nf = (BIGINT)(opts.upsampfac*ms);
  if (*nf<2*spopts.nspread) *nf=2*spopts.nspread; // otherwise spread fails
  if (*nf<MAX_NF){                                // otherwise will fail anyway
    if (opts.gpu_method == 4)                     // expensive at huge nf
      *nf = next235beven(*nf, bs);
    else
      *nf = next235beven(*nf, 1);
  }
}

void onedim_fseries_kernel(BIGINT nf, FLT *fwkerhalf, SPREAD_OPTS opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
              divided by h = 2pi/n.
              (should be allocated for at least nf/2+1 FLTs)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18
 */
{
  FLT J2 = opts.nspread/2.0;            // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q=(int)(2 + 3.0*J2);  // not sure why so large? cannot exceed MAX_NQUAD
  FLT f[MAX_NQUAD]; double z[2*MAX_NQUAD],w[2*MAX_NQUAD];
  legendre_compute_glr(2*q,z,w);        // only half the nodes used, eg on (0,1)
  dcomplex a[MAX_NQUAD];
  for (int n=0;n<q;++n) {               // set up nodes z_n and vals f_n
    z[n] *= J2;                         // rescale nodes
    f[n] = J2*(FLT)w[n] * evaluate_kernel((FLT)z[n], opts); // vals & quadr wei
    a[n] = exp(2*PI*IMA*(FLT)(nf/2-z[n])/(FLT)nf);  // phase winding rates
  }
  BIGINT nout=nf/2+1;                   // how many values we're writing to
  int nt = MIN(nout,MY_OMP_GET_MAX_THREADS());  // how many chunks
  std::vector<BIGINT> brk(nt+1);        // start indices for each thread
  for (int t=0; t<=nt; ++t)             // split nout mode indices btw threads
    brk[t] = (BIGINT)(0.5 + nout*t/(double)nt);
#pragma omp parallel
  {
    int t = MY_OMP_GET_THREAD_NUM();
    if (t<nt) {                         // could be nt < actual # threads
      dcomplex aj[MAX_NQUAD];           // phase rotator for this thread
      for (int n=0;n<q;++n)
	aj[n] = pow(a[n],(FLT)brk[t]);       // init phase factors for chunk
      for (BIGINT j=brk[t];j<brk[t+1];++j) {       // loop along output array
	FLT x = 0.0;                       // accumulator for answer at this j
	for (int n=0;n<q;++n) {
	  x += f[n] * 2*real(aj[n]);       // include the negative freq
	  aj[n] *= a[n];                   // wind the phases
	}
	fwkerhalf[j] = x;
      }
    }
  }
}
