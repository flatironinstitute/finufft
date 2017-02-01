#include "common.h"

BIGINT set_nf(BIGINT ms, nufft_opts opts, spread_opts spopts)
// type 1 & 2 recipe for how to set 1d size of upsampled array given opts
{
  BIGINT nf = 2*(BIGINT)(0.5*opts.R*ms);  // is even
  if (nf<2*spopts.nspread) nf=2*spopts.nspread;  // otherwise spread fails
  // now use next235?
  return nf;
}

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

void deconvolveshuffle1d(int dir,double prefac,double* ker, BIGINT ms,
			 double *fk, BIGINT nf1, fftw_complex* fw)
/*
  if dir==1: copies fw to fk with amplification by preface/ker
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  fk is complex array stored as 2*ms doubles alternating re,im parts.
  fw is a FFTW style complex array, ie double [nf1][2], effectively doubles
       alternating re,im parts.
  ker is real-valued double array of length nf1/2+1.

  todo: check RAM access in backwards order in 2nd loop is not a speed hit
  todo: check 2*(k0+k)+1 index calcs not slowing us down

  Barnett 
*/
{
  BIGINT k0 = ms/2;    // index shift in fk's = magnitude of most neg freq
  if (dir==1) {    // read fw, write out to fk
    for (BIGINT k=0;k<=(ms-1)/2;++k) {               // non-neg freqs k
      fk[2*(k0+k)] = prefac * fw[k][0] / ker[k];          // re
      fk[2*(k0+k)+1] = prefac * fw[k][1] / ker[k];        // im
    }
    for (BIGINT k=-1;k>=-k0;--k) {                 // neg freqs k
      fk[2*(k0+k)] = prefac * fw[nf1+k][0] / ker[-k];     // re
      fk[2*(k0+k)+1] = prefac * fw[nf1+k][1] / ker[-k];   // im
    }
  } else {    // read fk, write out to fw w/ zero padding
    for (BIGINT k=(ms-1)/2;k<nf1-k0;++k)             // zero pad
      fw[k][0] = fw[k][1] = 0.0;
    for (BIGINT k=0;k<=(ms-1)/2;++k) {               // non-neg freqs k
      fw[k][0] = prefac * fk[2*(k0+k)] / ker[k];          // re
      fw[k][1] = prefac * fk[2*(k0+k)+1] / ker[k];        // im
    }
    for (BIGINT k=-1;k>=-k0;--k) {                 // neg freqs k
      fw[nf1+k][0] = prefac * fk[2*(k0+k)] / ker[-k];          // re
      fw[nf1+k][1] = prefac * fk[2*(k0+k)+1] / ker[-k];        // im
    }
  }
}

void deconvolveshuffle2d(int dir,double prefac,double *ker1, double *ker2,
			 BIGINT ms, BIGINT mt,
			 double *fk, BIGINT nf1, BIGINT nf2, fftw_complex* fw)
/*
  2D version of deconvolveshuffle1d, calls it on each line using 1/ker2 fac.

  if dir==1: copies fw to fk with amplification by prefac/(ker1(k1)*ker2(k2)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  fk is complex array stored as 2*ms*mt doubles alternating re,im parts, with
    ms looped over fast and mt slow.
  fw is a FFTW style complex array, ie double [nf1*nf2][2], effectively doubles
       alternating re,im parts; again nf1 is fast and nf2 slow.
  ker1, ker2 are real-valued double arrays of lengths nf1/2+1, nf2/2+1
       respectively.
*/
{
  BIGINT k0 = mt/2;    // y-index shift in fk's = magnitude of most neg y-freq
  for (BIGINT k=0;k<=(mt-1)/2;++k)               // non-neg y-freqs k
    // point fk and fw to the start of this y value's row (2* is for complex):
    deconvolveshuffle1d(dir,prefac/ker2[k],ker1,ms,fk + 2*ms*(k0+k),nf1,&fw[nf1*k]);
  for (BIGINT k=-1;k>=-k0;--k)                 // neg y-freqs k
    deconvolveshuffle1d(dir,prefac/ker2[-k],ker1,ms,fk + 2*ms*(k0+k),nf1,&fw[nf1*(nf2+k)]);
}
