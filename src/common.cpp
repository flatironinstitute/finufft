#include "common.h"
#include "../contrib/legendre_rule_fast.h"
#include <fftw3.h>
#include <math.h>
#include <stdio.h>

// constants needed within common
#define MAX_NQUAD 100     // max number of positive quadr nodes

void set_nf_type12(BIGINT ms, nufft_opts opts, spread_opts spopts, BIGINT *nf)
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and number of Fourier modes ms.
{
  *nf = (BIGINT)(opts.R*ms);
  if (*nf<2*spopts.nspread) *nf=2*spopts.nspread;  // otherwise spread fails
  *nf = next235even(*nf);
}

void set_nhg_type3(double S, double X, nufft_opts opts, spread_opts spopts,
		     BIGINT *nf, double *h, double *gam)
/* sets nf, h (upsampled grid spacing), and gamma (x_j rescaling factor),
   for type 3 only.
   Inputs:
   X and S are the xj and sk interval half-widths respectively.
   opts and spopts are the NUFFT and spreader opts strucs, respectively.
   Outputs:
   nf is the size of upsampled grid for a given single dimension.
   h is the grid spacing = 2pi/nf
   gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).
   Barnett 2/13/17
*/
{
  int nss = spopts.nspread + 1;      // since ns may be odd
  *nf = (BIGINT)(2.0*opts.R*S*X/M_PI + nss);
  //printf("initial nf=%ld, ns=%d\n",nf,spopts.nspread);
  if (*nf<2*spopts.nspread) *nf=2*spopts.nspread;  // otherwise spread fails
  *nf = next235even(*nf);
  *h = 2*M_PI / *nf;                          // upsampled grid spacing
  *gam = (X/M_PI)/(1.0 - nss/(double)*nf);    // x scale fac
  *gam = max(*gam,1.0/S);                     // safely handle X=0 (zero width)
}

void onedim_dct_kernel(BIGINT nf, double *fwkerhalf, spread_opts opts)
/*
  Computes DCT coeffs of cnufftspread's real symmetric kernel, directly,
  exploiting narrowness of kernel. Uses phase winding for cheap eval on the
  regular freq grid.
  Note: obsolete, superceded by onedim_fseries_kernel.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier coeffs from indices 0 to nf/2 inclusive.
              (should be allocated for at least nf/2+1 doubles)

  Single thread only. Barnett 1/24/17
 */
{
  int m=ceil(opts.nspread/2.0);        // how many "modes" (ker pts) to include
  double f[MAX_NSPREAD/2];
  for (int n=0;n<=m;++n)    // actual freq index will be nf/2-n, for cosines
    f[n] = evaluate_kernel((double)n, opts);  // center at nf/2
  for (int n=1;n<=m;++n)               //  convert from exp to cosine ampls
    f[n] *= 2.0;
  dcomplex a[MAX_NSPREAD/2],aj[MAX_NSPREAD/2];
  for (int n=0;n<=m;++n) {             // set up our rotating phase array...
    a[n] = exp(2*M_PI*ima*(double)(nf/2-n)/(double)nf);   // phase differences
    aj[n] = dcomplex{1.0,0.0};         // init phase factors
  }
  for (BIGINT j=0;j<=nf/2;++j) {       // loop along output array
    double x = 0.0;                    // register
    for (int n=0;n<=m;++n) {
      x += f[n] * real(aj[n]);         // only want cosine part
      aj[n] *= a[n];                   // wind the phases
    }
    fwkerhalf[j] = x;
  }
}

void onedim_fseries_kernel(BIGINT nf, double *fwkerhalf, spread_opts opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive
              (should be allocated for at least nf/2+1 doubles)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  todo: understand how to openmp it? - subtle since private aj's. Want to break
        up fwkerhalf into contiguous pieces, one per thread. Low priority.
  Barnett 2/7/17
 */
{
  double J2 = opts.nspread/2.0;         // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q=(int)(2 + 3.0*J2);  // not sure why so large? cannot exceed MAX_NQUAD
  double f[MAX_NQUAD],z[2*MAX_NQUAD],w[2*MAX_NQUAD];
  legendre_compute_glr(2*q,z,w);        // only half the nodes used, eg on (0,1)
  dcomplex a[MAX_NQUAD],aj[MAX_NQUAD];  // phase rotators
  for (int n=0;n<q;++n) {
    z[n] *= J2;                 // rescale nodes
    f[n] = J2*w[n] * evaluate_kernel(z[n], opts);     // include quadr weights
    a[n] = exp(2*M_PI*ima*(double)(nf/2-z[n])/(double)nf);  // phase windings
    aj[n] = dcomplex{1.0,0.0};         // init phase factors
  }
  for (BIGINT j=0;j<=nf/2;++j) {       // loop along output array
    double x = 0.0;                    // register
    for (int n=0;n<q;++n) {
      x += f[n] * 2*real(aj[n]);       // include the negative freq
      aj[n] *= a[n];                   // wind the phases
    }
    fwkerhalf[j] = x;
  }
}

void onedim_nuft_kernel(BIGINT nk, double *k, double *phihat, spread_opts opts)
/*
  Approximates exact 1D Fourier transform of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Evaluates at set of arbitrary freqs k in [-pi,pi].

  Inputs:
  nk - number of freqs
  k - frequencies, dual to the kernel's natural argument, ie exp(i.k.z)
       Note, k values must be in [-pi,pi] for accuracy.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  phihat - real Fourier transform evaluated at freqs (alloc for nk doubles)

  Barnett 2/8/17. openmp since cos slow 2/9/17
 */
{
  double J2 = opts.nspread/2.0;        // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q=(int)(2 + 2.0*J2);     // > pi/2 ratio.  cannot exceed MAX_NQUAD
  if (opts.debug) printf("q (# ker FT quadr pts) = %d\n",q);
  double f[MAX_NQUAD],z[2*MAX_NQUAD],w[2*MAX_NQUAD];
  legendre_compute_glr(2*q,z,w);        // only half the nodes used, eg on (0,1)
  for (int n=0;n<q;++n) {
    z[n] *= J2;                                    // quadr nodes for [0,J/2]
    f[n] = J2*w[n] * evaluate_kernel(z[n], opts);  // include quadr weights
    //    printf("f[%d] = %.3g\n",n,f[n]);
  }
  #pragma omp parallel for schedule(dynamic)
  for (BIGINT j=0;j<nk;++j) {          // loop along output array
    double x = 0.0;                    // register
    for (int n=0;n<q;++n) x += f[n] * 2*cos(k[j]*z[n]);  // pos & neg freq pair
    phihat[j] = x;
  }
}  

void deconvolveshuffle1d(int dir,double prefac,double* ker, BIGINT ms,
			 double *fk, BIGINT nf1, fftw_complex* fw)
/*
  if dir==1: copies fw to fk with amplification by preface/ker
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  fk is size-ms double complex array (2*ms doubles alternating re,im parts)
  fw is a FFTW style complex array, ie double [nf1][2], essentially doubles
       alternating re,im parts.
  ker is real-valued double array of length nf1/2+1.

  Single thread only.

  It has been tested that the repeated floating division in this inner loop
  only contributes at the <3% level in 3D relative to the fftw cost (8 threads).
  This could be removed by passing in an inverse kernel and doing mults.

  todo: rewrite w/ native dcomplex I/O, check complex divide not slower than
        real divide, or is there a way to force a real divide?
  todo: check RAM access in backwards order in 2nd loop is not a speed hit
  todo: check 2*(k0+k)+1 index calcs not slowing us down

  Barnett 1/25/17
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
  2D version of deconvolveshuffle1d, calls it on each x-line using 1/ker2 fac.

  if dir==1: copies fw to fk with amplification by prefac/(ker1(k1)*ker2(k2)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  fk is complex array stored as 2*ms*mt doubles alternating re,im parts, with
    ms looped over fast and mt slow.
  fw is a FFTW style complex array, ie double [nf1*nf2][2], essentially doubles
       alternating re,im parts; again nf1 is fast and nf2 slow.
  ker1, ker2 are real-valued double arrays of lengths nf1/2+1, nf2/2+1
       respectively.

  Barnett 2/1/17
*/
{
  BIGINT k02 = mt/2;    // y-index shift in fk's = magnitude of most neg y-freq
  if (dir==2)               // zero pad needed x-lines (contiguous in memory)
    for (BIGINT k=nf1*(mt-1)/2;k<nf1*(nf2-k02);++k)  // k index sweeps all dims
	fw[k][0] = fw[k][1] = 0.0;
  for (BIGINT k2=0;k2<=(mt-1)/2;++k2)               // non-neg y-freqs
    // point fk and fw to the start of this y value's row (2* is for complex):
    deconvolveshuffle1d(dir,prefac/ker2[k2],ker1,ms,fk + 2*ms*(k02+k2),nf1,&fw[nf1*k2]);
  for (BIGINT k2=-1;k2>=-k02;--k2)                 // neg y-freqs
    deconvolveshuffle1d(dir,prefac/ker2[-k2],ker1,ms,fk + 2*ms*(k02+k2),nf1,&fw[nf1*(nf2+k2)]);
}

void deconvolveshuffle3d(int dir,double prefac,double *ker1, double *ker2,
			 double *ker3, BIGINT ms, BIGINT mt, BIGINT mu,
			 double *fk, BIGINT nf1, BIGINT nf2, BIGINT nf3,
			 fftw_complex* fw)
/*
  3D version of deconvolveshuffle2d, calls it on each xy-plane using 1/ker3 fac.

  if dir==1: copies fw to fk with ampl by prefac/(ker1(k1)*ker2(k2)*ker3(k3)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  fk is complex array stored as 2*ms*mt*mu doubles alternating re,im parts, with
    ms looped over fastest and mu slowest.
  fw is a FFTW style complex array, ie double [nf1*nf2*nf3][2], effectively
       doubles alternating re,im parts; again nf1 is fastest and nf3 slowest.
  ker1, ker2, ker3 are real-valued double arrays of lengths nf1/2+1, nf2/2+1,
       and nf3/2+1 respectively.

  Barnett 2/1/17
*/
{
  BIGINT k03 = mu/2;    // z-index shift in fk's = magnitude of most neg z-freq
  BIGINT np = nf1*nf2;  // # pts in an upsampled Fourier xy-plane
  if (dir==2)           // zero pad needed xy-planes (contiguous in memory)
    for (BIGINT k=np*(mu-1)/2;k<np*(nf3-k03);++k)  // sweeps all dims
      fw[k][0] = fw[k][1] = 0.0;
  for (BIGINT k3=0;k3<=(mu-1)/2;++k3)               // non-neg z-freqs
    // point fk and fw to the start of this z value's plane (2* is for complex):
    deconvolveshuffle2d(dir,prefac/ker3[k3],ker1,ker2,ms,mt,
			fk + 2*ms*mt*(k03+k3),nf1,nf2,&fw[np*k3]);
  for (BIGINT k3=-1;k3>=-k03;--k3)                 // neg z-freqs
    deconvolveshuffle2d(dir,prefac/ker3[-k3],ker1,ker2,ms,mt,
			fk + 2*ms*mt*(k03+k3),nf1,nf2,&fw[np*(nf3+k3)]);
}
