// public header
#include <finufft.h>

// private headers for lib build
// (must come after finufft.h which clobbers FINUFFT* macros)
#include <finufft/defs.h>
#include <finufft/fft.h>
#include <finufft/spreadinterp.h>
#include <finufft/utils.h>
#include <finufft/utils_precindep.h>

#include "../contrib/legendre_rule_fast.h"
#include <iomanip>
#include <iostream>
#include <math.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace finufft;
using namespace finufft::utils;
using namespace finufft::spreadinterp;
using namespace finufft::quadrature;

/* Computational core for FINUFFT.

   Based on Barnett 2017-2018 finufft?d.cpp containing nine drivers, plus
   2d1/2d2 many-vector drivers by Melody Shih, summer 2018.
   Original guru interface written by Andrea Malleo, summer 2019, mentored
   by Alex Barnett. Many rewrites in early 2020 by Alex Barnett & Libin Lu.

   As of v1.2 these replace the old hand-coded separate 9 finufft?d?() functions
   and the two finufft2d?many() functions. The (now 18) simple C++ interfaces
   are in simpleinterfaces.cpp.

Algorithm summaries taken from old finufft?d?() documentation, Feb-Jun 2017:

   TYPE 1:
   The type 1 NUFFT proceeds in three main steps:
   1) spread data to oversampled regular mesh using kernel.
   2) compute FFT on uniform mesh
   3) deconvolve by division of each Fourier mode independently by the kernel
    Fourier series coeffs (not merely FFT of kernel), shuffle to output.
   The kernel coeffs are precomputed in what is called step 0 in the code.

   TYPE 2:
   The type 2 algorithm proceeds in three main steps:
   1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
   2) compute inverse FFT on uniform fine grid
   3) spread (dir=2, ie interpolate) data to regular mesh
   The kernel coeffs are precomputed in what is called step 0 in the code.

   TYPE 3:
   The type 3 algorithm is basically a type 2 (which is implemented precisely
   as call to type 2) replacing the middle FFT (Step 2) of a type 1.
   Beyond this, the new twists are:
   i) nf1, number of upsampled points for the type-1, depends on the product
     of interval widths containing input and output points (X*S).
   ii) The deconvolve (post-amplify) step is division by the Fourier transform
     of the scaled kernel, evaluated on the *nonuniform* output frequency
     grid; this is done by direct approximation of the Fourier integral
     using quadrature of the kernel function times exponentials.
   iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
     half-widths X and S, hence nf1.

   MULTIPLE STRENGTH VECTORS FOR THE SAME NONUNIFORM POINTS (n_transf>1):
   maxBatchSize (set to max_num_omp_threads) times the RAM is needed, so
   this is good only for small problems.


Design notes for guru interface implementation:

* Since finufft_plan is C-compatible, we need to use malloc/free for its
  allocatable arrays, keeping it quite low-level. We can't use std::vector
  since that would only survive in the scope of each function.

* Thread-safety: FINUFFT plans are passed as pointers, so it has no global
  state apart from that associated with FFTW (and the did_fftw_init).
*/

// ---------- local math routines (were in common.cpp; no need now): --------

namespace finufft {
namespace common {

#ifndef FINUFFT_USE_DUCC0
// Technically global state...
// Needs to be static to avoid name collision with SINGLE/DOUBLE
static std::mutex fftw_lock;
#endif

// We macro because it has no FLT args but gets compiled for both prec's...
#ifdef SINGLE
#define SET_NF_TYPE12 set_nf_type12f
#else
#define SET_NF_TYPE12 set_nf_type12
#endif
int SET_NF_TYPE12(BIGINT ms, finufft_opts opts, finufft_spread_opts spopts, BIGINT *nf)
// Type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms. Returns 0 if success, else an
// error code if nf was unreasonably big (& tell the world).
{
  *nf = (BIGINT)(opts.upsampfac * ms); // manner of rounding not crucial
  if (*nf < 2 * spopts.nspread) *nf = 2 * spopts.nspread; // otherwise spread fails
  if (*nf < MAX_NF) {
    *nf = next235even(*nf);                               // expensive at huge nf
    return 0;
  } else {
    fprintf(stderr,
            "[%s] nf=%.3g exceeds MAX_NF of %.3g, so exit without attempting even a "
            "malloc\n",
            __func__, (double)*nf, (double)MAX_NF);
    return FINUFFT_ERR_MAXNALLOC;
  }
}

int setup_spreader_for_nufft(finufft_spread_opts &spopts, FLT eps, finufft_opts opts,
                             int dim)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Return status of setup_spreader. Uses pass-by-ref. Barnett 10/30/17
{
  // this calls spreadinterp.cpp...
  int ier = setup_spreader(spopts, eps, opts.upsampfac, opts.spread_kerevalmeth,
                           opts.spread_debug, opts.showwarn, dim);
  // override various spread opts from their defaults...
  spopts.debug    = opts.spread_debug;
  spopts.sort     = opts.spread_sort;   // could make dim or CPU choices here?
  spopts.kerpad   = opts.spread_kerpad; // (only applies to kerevalmeth=0)
  spopts.chkbnds  = opts.chkbnds;
  spopts.nthreads = opts.nthreads;      // 0 passed in becomes omp max by here
  if (opts.spread_nthr_atomic >= 0)     // overrides
    spopts.atomic_threshold = opts.spread_nthr_atomic;
  if (opts.spread_max_sp_size > 0)      // overrides
    spopts.max_subproblem_size = opts.spread_max_sp_size;
  if (opts.chkbnds != 1)                // deprecated default value hardcoded here
    fprintf(stderr,
            "[%s] opts.chkbnds is deprecated; ignoring change from default value.\n",
            __func__);
  return ier;
}

void set_nhg_type3(FLT S, FLT X, finufft_opts opts, finufft_spread_opts spopts,
                   BIGINT *nf, FLT *h, FLT *gam)
/* sets nf, h (upsampled grid spacing), and gamma (x_j rescaling factor),
   for type 3 only.
   Inputs:
   X and S are the xj and sk interval half-widths respectively.
   opts and spopts are the NUFFT and spreader opts strucs, respectively.
   Outputs:
   nf is the size of upsampled grid for a given single dimension.
   h is the grid spacing = 2pi/nf
   gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
*/
{
  int nss   = spopts.nspread + 1; // since ns may be odd
  FLT Xsafe = X, Ssafe = S;       // may be tweaked locally
  if (X == 0.0)                   // logic ensures XS>=1, handle X=0 a/o S=0
    if (S == 0.0) {
      Xsafe = 1.0;
      Ssafe = 1.0;
    } else
      Xsafe = max(Xsafe, 1 / S);
  else
    Ssafe = max(Ssafe, 1 / X);
  // use the safe X and S...
  FLT nfd = 2.0 * opts.upsampfac * Ssafe * Xsafe / PI + nss;
  if (!isfinite(nfd)) nfd = 0.0; // use FLT to catch inf
  *nf = (BIGINT)nfd;
  // printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  //  catch too small nf, and nan or +-inf, otherwise spread fails...
  if (*nf < 2 * spopts.nspread) *nf = 2 * spopts.nspread;
  if (*nf < MAX_NF)                                 // otherwise will fail anyway
    *nf = next235even(*nf);                         // expensive at huge nf
  *h   = 2 * PI / *nf;                              // upsampled grid spacing
  *gam = (FLT)*nf / (2.0 * opts.upsampfac * Ssafe); // x scale fac to x'
}

void onedim_fseries_kernel(BIGINT nf, FLT *fwkerhalf, finufft_spread_opts opts)
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

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18.
  Fixed num_threads 7/20/20
 */
{
  FLT J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 3.0 * J2); // not sure why so large? cannot exceed MAX_NQUAD
  FLT f[MAX_NQUAD];
  double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
  legendre_compute_glr(2 * q, z, w); // only half the nodes used, eg on (0,1)
  CPX a[MAX_NQUAD];
  for (int n = 0; n < q; ++n) {      // set up nodes z_n and vals f_n
    z[n] *= J2;                      // rescale nodes
    f[n] = J2 * (FLT)w[n] * evaluate_kernel((FLT)z[n], opts);  // vals & quadr wei
    a[n] = exp(2 * PI * IMA * (FLT)(nf / 2 - z[n]) / (FLT)nf); // phase winding rates
  }
  BIGINT nout = nf / 2 + 1;                       // how many values we're writing to
  int nt      = min(nout, (BIGINT)opts.nthreads); // how many chunks
  std::vector<BIGINT> brk(nt + 1);                // start indices for each thread
  for (int t = 0; t <= nt; ++t)                   // split nout mode indices btw threads
    brk[t] = (BIGINT)(0.5 + nout * t / (double)nt);
#pragma omp parallel num_threads(nt)
  {                                                // each thread gets own chunk to do
    int t = MY_OMP_GET_THREAD_NUM();
    CPX aj[MAX_NQUAD];                             // phase rotator for this thread
    for (int n = 0; n < q; ++n)
      aj[n] = pow(a[n], (FLT)brk[t]);              // init phase factors for chunk
    for (BIGINT j = brk[t]; j < brk[t + 1]; ++j) { // loop along output array
      FLT x = 0.0;                                 // accumulator for answer at this j
      for (int n = 0; n < q; ++n) {
        x += f[n] * 2 * real(aj[n]);               // include the negative freq
        aj[n] *= a[n];                             // wind the phases
      }
      fwkerhalf[j] = x;
    }
  }
}

void onedim_nuft_kernel(BIGINT nk, FLT *k, FLT *phihat, finufft_spread_opts opts)
/*
  Approximates exact 1D Fourier transform of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Evaluates at set of arbitrary freqs k in [-pi, pi),
  for a kernel with x measured in grid-spacings. (See previous routine for
  FT definition).

  Inputs:
  nk - number of freqs
  k - frequencies, dual to the kernel's natural argument, ie exp(i.k.z)
     Note, z is in grid-point units, and k values must be in [-pi, pi) for
     accuracy.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  phihat - real Fourier transform evaluated at freqs (alloc for nk FLTs)

  Barnett 2/8/17. openmp since cos slow 2/9/17
 */
{
  FLT J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 2.0 * J2); // > pi/2 ratio.  cannot exceed MAX_NQUAD
  if (opts.debug) printf("q (# ker FT quadr pts) = %d\n", q);
  FLT f[MAX_NQUAD];
  double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD]; // glr needs double
  legendre_compute_glr(2 * q, z, w);         // only half the nodes used, eg on (0,1)
  for (int n = 0; n < q; ++n) {
    z[n] *= (FLT)J2;                         // quadr nodes for [0,J/2]
    f[n] = J2 * (FLT)w[n] * evaluate_kernel((FLT)z[n], opts); // w/ quadr weights
    // printf("f[%d] = %.3g\n",n,f[n]);
  }
#pragma omp parallel for num_threads(opts.nthreads)
  for (BIGINT j = 0; j < nk; ++j) {          // loop along output array
    FLT x = 0.0;                             // register
    for (int n = 0; n < q; ++n)
      x += f[n] * 2 * cos(k[j] * (FLT)z[n]); // pos & neg freq pair.  use FLT cos!
    phihat[j] = x;
  }
}

void deconvolveshuffle1d(int dir, FLT prefac, FLT *ker, BIGINT ms, FLT *fk, BIGINT nf1,
                         CPX *fw, int modeord)
/*
  if dir==1: copies fw to fk with amplification by prefac/ker
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (from -N/2 up to N/2-1)
      1: use FFT-style (from 0 to N/2-1, then -N/2 up to -1).

  fk is a size-ms FLT complex array (2*ms FLTs alternating re,im parts)
  fw is a size-nf1 complex array (2*nf1 FLTs alternating re,im parts)
  ker is real-valued FLT array of length nf1/2+1.

  Single thread only, but shouldn't matter since mostly data movement.

  It has been tested that the repeated floating division in this inner loop
  only contributes at the <3% level in 3D relative to the FFT cost (8 threads).
  This could be removed by passing in an inverse kernel and doing mults.

  todo: rewrite w/ C++-complex I/O, check complex divide not slower than
    real divide, or is there a way to force a real divide?

  Barnett 1/25/17. Fixed ms=0 case 3/14/17. modeord flag & clean 10/25/17
*/
{
  BIGINT kmin = -ms / 2, kmax = (ms - 1) / 2; // inclusive range of k indices
  if (ms == 0) kmax = -1;                     // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2 * kmin, pn = 0; // CMCL mode-ordering case (2* since cmplx)
  if (modeord == 1) {
    pp = 0;
    pn = 2 * (kmax + 1);
  } // or, instead, FFT ordering
  if (dir == 1) {                                       // read fw, write out to fk...
    for (BIGINT k = 0; k <= kmax; ++k) {                // non-neg freqs k
      fk[pp++] = prefac * fw[k].real() / ker[k];        // re
      fk[pp++] = prefac * fw[k].imag() / ker[k];        // im
    }
    for (BIGINT k = kmin; k < 0; ++k) {                 // neg freqs k
      fk[pn++] = prefac * fw[nf1 + k].real() / ker[-k]; // re
      fk[pn++] = prefac * fw[nf1 + k].imag() / ker[-k]; // im
    }
  } else { // read fk, write out to fw w/ zero padding...
    for (BIGINT k = kmax + 1; k < nf1 + kmin; ++k) { // zero pad precisely where
                                                     // needed
      fw[k] = 0.0;
    }
    for (BIGINT k = 0; k <= kmax; ++k) {             // non-neg freqs k
      fw[k].real(prefac * fk[pp++] / ker[k]);        // re
      fw[k].imag(prefac * fk[pp++] / ker[k]);        // im
    }
    for (BIGINT k = kmin; k < 0; ++k) {              // neg freqs k
      fw[nf1 + k].real(prefac * fk[pn++] / ker[-k]); // re
      fw[nf1 + k].imag(prefac * fk[pn++] / ker[-k]); // im
    }
  }
}

void deconvolveshuffle2d(int dir, FLT prefac, FLT *ker1, FLT *ker2, BIGINT ms, BIGINT mt,
                         FLT *fk, BIGINT nf1, BIGINT nf2, CPX *fw, int modeord)
/*
  2D version of deconvolveshuffle1d, calls it on each x-line using 1/ker2 fac.

  if dir==1: copies fw to fk with amplification by prefac/(ker1(k1)*ker2(k2)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (each dim increasing)
      1: use FFT-style (pos then negative, on each dim)

  fk is a complex array stored as 2*ms*mt FLTs alternating re,im parts, with
  ms looped over fast and mt slow.
  fw is a complex array stored as 2*nf1*nf2] FLTs alternating re,im parts, with
  nf1 looped over fast and nf2 slow.
  ker1, ker2 are real-valued FLT arrays of lengths nf1/2+1, nf2/2+1
     respectively.

  Barnett 2/1/17, Fixed mt=0 case 3/14/17. modeord 10/25/17
*/
{
  BIGINT k2min = -mt / 2, k2max = (mt - 1) / 2; // inclusive range of k2 indices
  if (mt == 0) k2max = -1;                      // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2 * k2min * ms, pn = 0; // CMCL mode-ordering case (2* since cmplx)
  if (modeord == 1) {
    pp = 0;
    pn = 2 * (k2max + 1) * ms;
  } // or, instead, FFT ordering
  if (dir == 2) // zero pad needed x-lines (contiguous in memory)
    for (BIGINT j = nf1 * (k2max + 1); j < nf1 * (nf2 + k2min); ++j) // sweeps all
                                                                     // dims
      fw[j] = 0.0;
  for (BIGINT k2 = 0; k2 <= k2max; ++k2, pp += 2 * ms)               // non-neg y-freqs
    // point fk and fw to the start of this y value's row (2* is for complex):
    common::deconvolveshuffle1d(dir, prefac / ker2[k2], ker1, ms, fk + pp, nf1,
                                &fw[nf1 * k2], modeord);
  for (BIGINT k2 = k2min; k2 < 0; ++k2, pn += 2 * ms) // neg y-freqs
    common::deconvolveshuffle1d(dir, prefac / ker2[-k2], ker1, ms, fk + pn, nf1,
                                &fw[nf1 * (nf2 + k2)], modeord);
}

void deconvolveshuffle3d(int dir, FLT prefac, FLT *ker1, FLT *ker2, FLT *ker3, BIGINT ms,
                         BIGINT mt, BIGINT mu, FLT *fk, BIGINT nf1, BIGINT nf2,
                         BIGINT nf3, CPX *fw, int modeord)
/*
  3D version of deconvolveshuffle2d, calls it on each xy-plane using 1/ker3 fac.

  if dir==1: copies fw to fk with ampl by prefac/(ker1(k1)*ker2(k2)*ker3(k3)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (each dim increasing)
      1: use FFT-style (pos then negative, on each dim)

  fk is a complex array stored as 2*ms*mt*mu FLTs alternating re,im parts, with
  ms looped over fastest and mu slowest.
  fw is a complex array stored as 2*nf1*nf2*nf3 FLTs alternating re,im parts, with
  nf1 looped over fastest and nf3 slowest.
  ker1, ker2, ker3 are real-valued FLT arrays of lengths nf1/2+1, nf2/2+1,
     and nf3/2+1 respectively.

  Barnett 2/1/17, Fixed mu=0 case 3/14/17. modeord 10/25/17
*/
{
  BIGINT k3min = -mu / 2, k3max = (mu - 1) / 2; // inclusive range of k3 indices
  if (mu == 0) k3max = -1;                      // fixes zero-pad for trivial no-mode case
  // set up pp & pn as ptrs to start of pos(ie nonneg) & neg chunks of fk array
  BIGINT pp = -2 * k3min * ms * mt, pn = 0; // CMCL mode-ordering (2* since cmplx)
  if (modeord == 1) {
    pp = 0;
    pn = 2 * (k3max + 1) * ms * mt;
  } // or FFT ordering
  BIGINT np = nf1 * nf2; // # pts in an upsampled Fourier xy-plane
  if (dir == 2)          // zero pad needed xy-planes (contiguous in memory)
    for (BIGINT j = np * (k3max + 1); j < np * (nf3 + k3min); ++j) // sweeps all dims
      fw[j] = 0.0;
  for (BIGINT k3 = 0; k3 <= k3max; ++k3, pp += 2 * ms * mt)        // non-neg z-freqs
    // point fk and fw to the start of this z value's plane (2* is for complex):
    common::deconvolveshuffle2d(dir, prefac / ker3[k3], ker1, ker2, ms, mt, fk + pp, nf1,
                                nf2, &fw[np * k3], modeord);
  for (BIGINT k3 = k3min; k3 < 0; ++k3, pn += 2 * ms * mt) // neg z-freqs
    common::deconvolveshuffle2d(dir, prefac / ker3[-k3], ker1, ker2, ms, mt, fk + pn, nf1,
                                nf2, &fw[np * (nf3 + k3)], modeord);
}

// --------- batch helper functions for t1,2 exec: ---------------------------

int spreadinterpSortedBatch(int batchSize, FINUFFT_PLAN p, CPX *cBatch)
/*
  Spreads (or interpolates) a batch of batchSize strength vectors in cBatch
  to (or from) the batch of fine working grids p->fwBatch, using the same set of
  (index-sorted) NU points p->X,Y,Z for each vector in the batch.
  The direction (spread vs interpolate) is set by p->spopts.spread_direction.
  Returns 0 (no error reporting for now).
  Notes:
  1) cBatch is already assumed to have the correct offset, ie here we
   read from the start of cBatch (unlike Malleo). fwBatch also has zero offset
  2) this routine is a batched version of spreadinterpSorted in spreadinterp.cpp
  Barnett 5/19/20, based on Malleo 2019.
*/
{
  // opts.spread_thread: 1 sequential multithread, 2 parallel single-thread.
  // omp_sets_nested deprecated, so don't use; assume not nested for 2 to work.
  // But when nthr_outer=1 here, omp par inside the loop sees all threads...
#ifdef _OPENMP
  int nthr_outer = p->opts.spread_thread == 1 ? 1 : batchSize;
#endif
#pragma omp parallel for num_threads(nthr_outer)
  for (int i = 0; i < batchSize; i++) {
    CPX *fwi = p->fwBatch + i * p->nf; // start of i'th fw array in wkspace
    CPX *ci  = cBatch + i * p->nj;     // start of i'th c array in cBatch
    spreadinterpSorted(p->sortIndices, p->nf1, p->nf2, p->nf3, (FLT *)fwi, p->nj, p->X,
                       p->Y, p->Z, (FLT *)ci, p->spopts, p->didSort);
  }
  return 0;
}

int deconvolveBatch(int batchSize, FINUFFT_PLAN p, CPX *fkBatch)
/*
  Type 1: deconvolves (amplifies) from each interior fw array in p->fwBatch
  into each output array fk in fkBatch.
  Type 2: deconvolves from user-supplied input fk to 0-padded interior fw,
  again looping over fk in fkBatch and fw in p->fwBatch.
  The direction (spread vs interpolate) is set by p->spopts.spread_direction.
  This is mostly a loop calling deconvolveshuffle?d for the needed dim batchSize
  times.
  Barnett 5/21/20, simplified from Malleo 2019 (eg t3 logic won't be in here)
*/
{
  // since deconvolveshuffle?d are single-thread, omp par seems to help here...
#pragma omp parallel for num_threads(batchSize)
  for (int i = 0; i < batchSize; i++) {
    CPX *fwi = p->fwBatch + i * p->nf; // start of i'th fw array in wkspace
    CPX *fki = fkBatch + i * p->N;     // start of i'th fk array in fkBatch

    // Call routine from common.cpp for the dim; prefactors hardcoded to 1.0...
    if (p->dim == 1)
      deconvolveshuffle1d(p->spopts.spread_direction, 1.0, p->phiHat1, p->ms, (FLT *)fki,
                          p->nf1, fwi, p->opts.modeord);
    else if (p->dim == 2)
      deconvolveshuffle2d(p->spopts.spread_direction, 1.0, p->phiHat1, p->phiHat2, p->ms,
                          p->mt, (FLT *)fki, p->nf1, p->nf2, fwi, p->opts.modeord);
    else
      deconvolveshuffle3d(p->spopts.spread_direction, 1.0, p->phiHat1, p->phiHat2,
                          p->phiHat3, p->ms, p->mt, p->mu, (FLT *)fki, p->nf1, p->nf2,
                          p->nf3, fwi, p->opts.modeord);
  }
  return 0;
}

} // namespace common
} // namespace finufft

// --------------- rest is the 5 user guru (plan) interface drivers: ---------
// (not namespaced since have safe names finufft{f}_* )
using namespace finufft::common; // accesses routines defined above

// Marco Barbone: 5.8.2024
// These are user-facing.
// The various options could be macros to follow c standard library conventions.
// Question: would these be enums?

// OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
void FINUFFT_DEFAULT_OPTS(finufft_opts *o)
// Sets default nufft opts (referenced by all language interfaces too).
// See finufft_opts.h for meanings.
// This was created to avoid uncertainty about C++11 style static initialization
// when called from MEX, but now is generally used. Barnett 10/30/17 onwards.
// Sphinx sucks the below code block into the web docs, hence keep it clean...
{
  // sphinx tag (don't remove): @defopts_start
  o->modeord = 0;
  o->chkbnds = 1;

  o->debug        = 0;
  o->spread_debug = 0;
  o->showwarn     = 1;

  o->nthreads = 0;
#ifdef FINUFFT_USE_DUCC0
  o->fftw = 0;
#else
  o->fftw = FFTW_ESTIMATE;
#endif
  o->spread_sort        = 2;
  o->spread_kerevalmeth = 1;
  o->spread_kerpad      = 1;
  o->upsampfac          = 0.0;
  o->spread_thread      = 0;
  o->maxbatchsize       = 0;
  o->spread_nthr_atomic = -1;
  o->spread_max_sp_size = 0;
  // sphinx tag (don't remove): @defopts_end
}

// PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
int FINUFFT_MAKEPLAN(int type, int dim, BIGINT *n_modes, int iflag, int ntrans, FLT tol,
                     FINUFFT_PLAN *pp, finufft_opts *opts)
// Populates the fields of finufft_plan which is pointed to by "pp".
// opts is ptr to a finufft_opts to set options, or NULL to use defaults.
// For some of the fields (if "auto" selected) here choose the actual setting.
// For types 1,2 allocates memory for internal working arrays,
// evaluates spreading kernel coefficients, and instantiates the fftw_plan
{
  FINUFFT_PLAN p;
  p   = new FINUFFT_PLAN_S; // allocate fresh plan struct
  *pp = p;                  // pass out plan as ptr to plan struct

  if (opts == NULL)         // use default opts
    FINUFFT_DEFAULT_OPTS(&(p->opts));
  else                      // or read from what's passed in
    p->opts = *opts;        // keep a deep copy; changing *opts now has no effect

  if (p->opts.debug)        // do a hello world
    printf("[%s] new plan: FINUFFT version " FINUFFT_VER " .................\n",
           __func__);

  if ((type != 1) && (type != 2) && (type != 3)) {
    fprintf(stderr, "[%s] Invalid type (%d), should be 1, 2 or 3.\n", __func__, type);
    return FINUFFT_ERR_TYPE_NOTVALID;
  }
  if ((dim != 1) && (dim != 2) && (dim != 3)) {
    fprintf(stderr, "[%s] Invalid dim (%d), should be 1, 2 or 3.\n", __func__, dim);
    return FINUFFT_ERR_DIM_NOTVALID;
  }
  if (ntrans < 1) {
    fprintf(stderr, "[%s] ntrans (%d) should be at least 1.\n", __func__, ntrans);
    return FINUFFT_ERR_NTRANS_NOTVALID;
  }

  // get stuff from args...
  p->type    = type;
  p->dim     = dim;
  p->ntrans  = ntrans;
  p->tol     = tol;
  p->fftSign = (iflag >= 0) ? 1 : -1; // clean up flag input

                                      // choose overall # threads...
#ifdef _OPENMP
  int ompmaxnthr = MY_OMP_GET_MAX_THREADS();
  int nthr       = ompmaxnthr; // default: use as many as OMP gives us
  // (the above could be set, or suggested set, to 1 for small enough problems...)
  if (p->opts.nthreads > 0) {
    nthr = p->opts.nthreads; // user override, now without limit
    if (p->opts.showwarn && (nthr > ompmaxnthr))
      fprintf(stderr,
              "%s warning: using opts.nthreads=%d, more than the %d OpenMP claims "
              "available; note large nthreads can be slower.\n",
              __func__, nthr, ompmaxnthr);
  }
#else
  int nthr = 1; // always 1 thread (avoid segfault)
  if (p->opts.nthreads > 1)
    fprintf(stderr,
            "%s warning: opts.nthreads=%d but library is single-threaded; ignoring!\n",
            __func__, p->opts.nthreads);
#endif
  p->opts.nthreads = nthr; // store actual # thr planned for
  // (this sets/limits all downstream spread/interp, 1dkernel, and FFT thread counts...)

  // choose batchSize for types 1,2 or 3... (uses int ceil(b/a)=1+(b-1)/a trick)
  if (p->opts.maxbatchsize == 0) {                  // logic to auto-set best batchsize
    p->nbatch    = 1 + (ntrans - 1) / nthr;         // min # batches poss
    p->batchSize = 1 + (ntrans - 1) / p->nbatch;    // then cut # thr in each b
  } else {                                          // batchSize override by user
    p->batchSize = min(p->opts.maxbatchsize, ntrans);
    p->nbatch    = 1 + (ntrans - 1) / p->batchSize; // resulting # batches
  }
  if (p->opts.spread_thread == 0) p->opts.spread_thread = 2; // our auto choice
  if (p->opts.spread_thread != 1 && p->opts.spread_thread != 2) {
    fprintf(stderr, "[%s] illegal opts.spread_thread!\n", __func__);
    return FINUFFT_ERR_SPREAD_THREAD_NOTVALID;
  }

  if (type != 3) {                      // read in user Fourier mode array sizes...
    p->ms = n_modes[0];
    p->mt = (dim > 1) ? n_modes[1] : 1; // leave as 1 for unused dims
    p->mu = (dim > 2) ? n_modes[2] : 1;
    p->N  = p->ms * p->mt * p->mu;      // N = total # modes
  }

  // heuristic to choose default upsampfac... (currently two poss)
  if (p->opts.upsampfac == 0.0) {            // indicates auto-choose
    p->opts.upsampfac = 2.0;                 // default, and need for tol small
    if (tol >= (FLT)1E-9) {                  // the tol sigma=5/4 can reach
      if (type == 3)                         // could move to setpts, more known?
        p->opts.upsampfac = 1.25;            // faster b/c smaller RAM & FFT
      else if ((dim == 1 && p->N > 10000000) || (dim == 2 && p->N > 300000) ||
               (dim == 3 && p->N > 3000000)) // type 1,2 heuristic cutoffs, double,
                                             // typ tol, 12-core xeon
        p->opts.upsampfac = 1.25;
    }
    if (p->opts.debug > 1)
      printf("[%s] set auto upsampfac=%.2f\n", __func__, p->opts.upsampfac);
  }
  // use opts to choose and write into plan's spread options...
  int ier = setup_spreader_for_nufft(p->spopts, tol, p->opts, dim);
  if (ier > 1) // proceed if success or warning
    return ier;

  // set others as defaults (or unallocated for arrays)...
  p->X           = NULL;
  p->Y           = NULL;
  p->Z           = NULL;
  p->phiHat1     = NULL;
  p->phiHat2     = NULL;
  p->phiHat3     = NULL;
  p->nf1         = 1;
  p->nf2         = 1;
  p->nf3         = 1;    // crucial to leave as 1 for unused dims
  p->sortIndices = NULL; // used in all three types

  //  ------------------------ types 1,2: planning needed ---------------------
  if (type == 1 || type == 2) {

#ifndef FINUFFT_USE_DUCC0
    int nthr_fft = nthr; // give FFTW all threads (or use o.spread_thread?)
                         // Note: batchSize not used since might be only 1.
    // Now place FFTW initialization in a lock, courtesy of OMP. Makes FINUFFT
    // thread-safe (can be called inside OMP)
    {
      static bool did_fftw_init = false; // the only global state of FINUFFT
      std::lock_guard<std::mutex> lock(fftw_lock);
      if (!did_fftw_init) {
        FFTW_INIT();          // setup FFTW global state; should only do once
        did_fftw_init = true; // ensure other FINUFFT threads don't clash
      }
    }
#endif

    p->spopts.spread_direction = type;

    if (p->opts.showwarn) { // user warn round-off error...
      if (EPSILON * p->ms > 1.0)
        fprintf(stderr, "%s warning: rounding err predicted eps_mach*N1 = %.3g > 1 !\n",
                __func__, (double)(EPSILON * p->ms));
      if (EPSILON * p->mt > 1.0)
        fprintf(stderr, "%s warning: rounding err predicted eps_mach*N2 = %.3g > 1 !\n",
                __func__, (double)(EPSILON * p->mt));
      if (EPSILON * p->mu > 1.0)
        fprintf(stderr, "%s warning: rounding err predicted eps_mach*N3 = %.3g > 1 !\n",
                __func__, (double)(EPSILON * p->mu));
    }

    // determine fine grid sizes, sanity check..
    int nfier = SET_NF_TYPE12(p->ms, p->opts, p->spopts, &(p->nf1));
    if (nfier) return nfier; // nf too big; we're done
    p->phiHat1 = (FLT *)malloc(sizeof(FLT) * (p->nf1 / 2 + 1));
    if (dim > 1) {
      nfier = SET_NF_TYPE12(p->mt, p->opts, p->spopts, &(p->nf2));
      if (nfier) return nfier;
      p->phiHat2 = (FLT *)malloc(sizeof(FLT) * (p->nf2 / 2 + 1));
    }
    if (dim > 2) {
      nfier = SET_NF_TYPE12(p->mu, p->opts, p->spopts, &(p->nf3));
      if (nfier) return nfier;
      p->phiHat3 = (FLT *)malloc(sizeof(FLT) * (p->nf3 / 2 + 1));
    }

    if (p->opts.debug) { // "long long" here is to avoid warnings with printf...
      printf("[%s] %dd%d: (ms,mt,mu)=(%lld,%lld,%lld) "
             "(nf1,nf2,nf3)=(%lld,%lld,%lld)\n               ntrans=%d nthr=%d "
             "batchSize=%d ",
             __func__, dim, type, (long long)p->ms, (long long)p->mt, (long long)p->mu,
             (long long)p->nf1, (long long)p->nf2, (long long)p->nf3, ntrans, nthr,
             p->batchSize);
      if (p->batchSize == 1) // spread_thread has no effect in this case
        printf("\n");
      else
        printf(" spread_thread=%d\n", p->opts.spread_thread);
    }

    // STEP 0: get Fourier coeffs of spreading kernel along each fine grid dim
    CNTime timer;
    timer.start();
    onedim_fseries_kernel(p->nf1, p->phiHat1, p->spopts);
    if (dim > 1) onedim_fseries_kernel(p->nf2, p->phiHat2, p->spopts);
    if (dim > 2) onedim_fseries_kernel(p->nf3, p->phiHat3, p->spopts);
    if (p->opts.debug)
      printf("[%s] kernel fser (ns=%d):\t\t%.3g s\n", __func__, p->spopts.nspread,
             timer.elapsedsec());

    p->nf = p->nf1 * p->nf2 * p->nf3; // fine grid total number of points
    if (p->nf * p->batchSize > MAX_NF) {
      fprintf(stderr,
              "[%s] fwBatch would be bigger than MAX_NF, not attempting malloc!\n",
              __func__);
      return FINUFFT_ERR_MAXNALLOC;
    }

    timer.restart();
#ifdef FINUFFT_USE_DUCC0
    p->fwBatch = (CPX *)malloc(p->nf * p->batchSize * sizeof(CPX)); // the big workspace
#else
    p->fwBatch = (CPX *)FFTW_ALLOC_CPX(p->nf * p->batchSize); // the big workspace
#endif
    if (p->opts.debug)
      printf("[%s] fwBatch %.2fGB alloc:   \t%.3g s\n", __func__,
             (double)1E-09 * sizeof(CPX) * p->nf * p->batchSize, timer.elapsedsec());
    if (!p->fwBatch) { // we don't catch all such mallocs, just this big one
      fprintf(stderr, "[%s] FFTW malloc failed for fwBatch (working fine grids)!\n",
              __func__);
      free(p->phiHat1);
      free(p->phiHat2);
      free(p->phiHat3);
      return FINUFFT_ERR_ALLOC;
    }

#ifndef FINUFFT_USE_DUCC0
    timer.restart(); // plan the FFTW
    int *ns = gridsize_for_fft(p);
    // fftw_plan_many_dft args: rank, gridsize/dim, howmany, in, inembed, istride,
    // idist, ot, onembed, ostride, odist, sign, flags
    {
      std::lock_guard<std::mutex> lock(fftw_lock);

      // FFTW_PLAN_TH sets all future fftw_plan calls to use nthr_fft threads.
      // FIXME: Since this might override what the user wants for fftw, we'd like to
      // set it just for our one plan and then revert to the user value.
      // Unfortunately fftw_planner_nthreads wasn't introduced until fftw 3.3.9, and
      // there isn't a convenient mechanism to probe the version
      FFTW_PLAN_TH(nthr_fft);
      p->fftwPlan = FFTW_PLAN_MANY_DFT(dim, ns, p->batchSize, (FFTW_CPX *)p->fwBatch,
                                       NULL, 1, p->nf, (FFTW_CPX *)p->fwBatch, NULL, 1,
                                       p->nf, p->fftSign, p->opts.fftw);
    }
    if (p->opts.debug)
      printf("[%s] FFTW plan (mode %d, nthr=%d):\t%.3g s\n", __func__, p->opts.fftw,
             nthr_fft, timer.elapsedsec());
    delete[] ns;
#endif

  } else { // -------------------------- type 3 (no planning) ------------

    if (p->opts.debug) printf("[%s] %dd%d: ntrans=%d\n", __func__, dim, type, ntrans);
    // in case destroy occurs before setpts, need safe dummy ptrs/plans...
    p->CpBatch     = NULL;
    p->fwBatch     = NULL;
    p->Sp          = NULL;
    p->Tp          = NULL;
    p->Up          = NULL;
    p->prephase    = NULL;
    p->deconv      = NULL;
    p->innerT2plan = NULL;
    // Type 3 will call finufft_makeplan for type 2; no need to init FFTW
    // Note we don't even know nj or nk yet, so can't do anything else!
  }
  return ier; // report setup_spreader status (could be warning)
}

// SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
int FINUFFT_SETPTS(FINUFFT_PLAN p, BIGINT nj, FLT *xj, FLT *yj, FLT *zj, BIGINT nk,
                   FLT *s, FLT *t, FLT *u)
/* For type 1,2: just checks and (possibly) sorts the NU xyz points, in prep for
   spreading. (The last 4 arguments are ignored.)
   For type 3: allocates internal working arrays, scales/centers the NU points
   and NU target freqs (stu), evaluates spreading kernel FT at all target freqs.
*/
{
  int d = p->dim; // abbrev for spatial dim
  CNTime timer;
  timer.start();
  p->nj = nj; // the user only now chooses how many NU (x,y,z) pts
  if (nj < 0) {
    fprintf(stderr, "[%s] nj (%lld) cannot be negative!\n", __func__, (long long)nj);
    return FINUFFT_ERR_NUM_NU_PTS_INVALID;
  } else if (nj > MAX_NU_PTS) {
    fprintf(stderr, "[%s] nj (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nj);
    return FINUFFT_ERR_NUM_NU_PTS_INVALID;
  }

  if (p->type != 3) { // ------------------ TYPE 1,2 SETPTS -------------------
                      // (all we can do is check and maybe bin-sort the NU pts)
    p->X    = xj;     // plan must keep pointers to user's fixed NU pts
    p->Y    = yj;
    p->Z    = zj;
    int ier = spreadcheck(p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (p->opts.debug > 1)
      printf("[%s] spreadcheck (%d):\t%.3g s\n", __func__, p->spopts.chkbnds,
             timer.elapsedsec());
    if (ier) // no warnings allowed here
      return ier;
    timer.restart();
    // Free sortIndices if it has been allocated before in case of repeated setpts
    // calls causing memory leak. We don't know it is the same size as before, so we
    // have to malloc each time.
    if (p->sortIndices) free(p->sortIndices);
    p->sortIndices = (BIGINT *)malloc(sizeof(BIGINT) * p->nj);
    if (!p->sortIndices) {
      fprintf(stderr, "[%s] failed to allocate sortIndices!\n", __func__);
      return FINUFFT_ERR_SPREAD_ALLOC;
    }
    p->didSort =
        indexSort(p->sortIndices, p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (p->opts.debug)
      printf("[%s] sort (didSort=%d):\t\t%.3g s\n", __func__, p->didSort,
             timer.elapsedsec());

  } else { // ------------------------- TYPE 3 SETPTS -----------------------
           // (here we can precompute pre/post-phase factors and plan the t2)

    if (nk < 0) {
      fprintf(stderr, "[%s] nk (%lld) cannot be negative!\n", __func__, (long long)nk);
      return FINUFFT_ERR_NUM_NU_PTS_INVALID;
    } else if (nk > MAX_NU_PTS) {
      fprintf(stderr, "[%s] nk (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nk);
      return FINUFFT_ERR_NUM_NU_PTS_INVALID;
    }
    p->nk = nk; // user set # targ freq pts
    p->S  = s;  // keep pointers to user's input target pts
    p->T  = t;
    p->U  = u;

    // pick x, s intervals & shifts & # fine grid pts (nf) in each dim...
    FLT S1, S2, S3; // get half-width X, center C, which contains {x_j}...
    arraywidcen(nj, xj, &(p->t3P.X1), &(p->t3P.C1));
    arraywidcen(nk, s, &S1, &(p->t3P.D1)); // same D, S, but for {s_k}
    set_nhg_type3(S1, p->t3P.X1, p->opts, p->spopts, &(p->nf1), &(p->t3P.h1),
                  &(p->t3P.gam1));         // applies twist i)
    p->t3P.C2 = 0.0;                       // their defaults if dim 2 unused, etc
    p->t3P.D2 = 0.0;
    if (d > 1) {
      arraywidcen(nj, yj, &(p->t3P.X2), &(p->t3P.C2)); // {y_j}
      arraywidcen(nk, t, &S2, &(p->t3P.D2));           // {t_k}
      set_nhg_type3(S2, p->t3P.X2, p->opts, p->spopts, &(p->nf2), &(p->t3P.h2),
                    &(p->t3P.gam2));
    }
    p->t3P.C3 = 0.0;
    p->t3P.D3 = 0.0;
    if (d > 2) {
      arraywidcen(nj, zj, &(p->t3P.X3), &(p->t3P.C3)); // {z_j}
      arraywidcen(nk, u, &S3, &(p->t3P.D3));           // {u_k}
      set_nhg_type3(S3, p->t3P.X3, p->opts, p->spopts, &(p->nf3), &(p->t3P.h3),
                    &(p->t3P.gam3));
    }

    if (p->opts.debug) { // report on choices of shifts, centers, etc...
      printf("\tM=%lld N=%lld\n", (long long)nj, (long long)nk);
      printf("\tX1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%lld\t\n", p->t3P.X1,
             p->t3P.C1, S1, p->t3P.D1, p->t3P.gam1, (long long)p->nf1);
      if (d > 1)
        printf("\tX2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%lld\n", p->t3P.X2,
               p->t3P.C2, S2, p->t3P.D2, p->t3P.gam2, (long long)p->nf2);
      if (d > 2)
        printf("\tX3=%.3g C3=%.3g S3=%.3g D3=%.3g gam3=%g nf3=%lld\n", p->t3P.X3,
               p->t3P.C3, S3, p->t3P.D3, p->t3P.gam3, (long long)p->nf3);
    }
    p->nf = p->nf1 * p->nf2 * p->nf3; // fine grid total number of points
    if (p->nf * p->batchSize > MAX_NF) {
      fprintf(stderr,
              "[%s t3] fwBatch would be bigger than MAX_NF, not attempting malloc!\n",
              __func__);
      return FINUFFT_ERR_MAXNALLOC;
    }
#ifdef FINUFFT_USE_DUCC0
    free(p->fwBatch);
    p->fwBatch = (CPX *)malloc(p->nf * p->batchSize * sizeof(CPX)); // maybe big workspace
#else
    if (p->fwBatch) FFTW_FR(p->fwBatch);
    p->fwBatch = (CPX *)FFTW_ALLOC_CPX(p->nf * p->batchSize); // maybe big workspace
#endif

    // (note FFTW_ALLOC is not needed over malloc, but matches its type)
    if (p->CpBatch) free(p->CpBatch);
    p->CpBatch = (CPX *)malloc(sizeof(CPX) * nj * p->batchSize); // batch c' work

    if (p->opts.debug)
      printf("[%s t3] widcen, batch %.2fGB alloc:\t%.3g s\n", __func__,
             (double)1E-09 * sizeof(CPX) * (p->nf + nj) * p->batchSize,
             timer.elapsedsec());
    if (!p->fwBatch || !p->CpBatch) {
      fprintf(stderr, "[%s t3] malloc fail for fwBatch or CpBatch!\n", __func__);
      return FINUFFT_ERR_ALLOC;
    }
    // printf("fwbatch, cpbatch ptrs: %llx %llx\n",p->fwBatch,p->CpBatch);

    // alloc rescaled NU src pts x'_j (in X etc), rescaled NU targ pts s'_k ...
    if (p->X) free(p->X);
    if (p->Sp) free(p->Sp);
    p->X  = (FLT *)malloc(sizeof(FLT) * nj);
    p->Sp = (FLT *)malloc(sizeof(FLT) * nk);
    if (d > 1) {
      if (p->Y) free(p->Y);
      if (p->Tp) free(p->Tp);
      p->Y  = (FLT *)malloc(sizeof(FLT) * nj);
      p->Tp = (FLT *)malloc(sizeof(FLT) * nk);
    }
    if (d > 2) {
      if (p->Z) free(p->Z);
      if (p->Up) free(p->Up);
      p->Z  = (FLT *)malloc(sizeof(FLT) * nj);
      p->Up = (FLT *)malloc(sizeof(FLT) * nk);
    }

    // always shift as use gam to rescale x_j to x'_j, etc (twist iii)...
    FLT ig1 = 1.0 / p->t3P.gam1, ig2 = 0.0, ig3 = 0.0; // "reciprocal-math" optim
    if (d > 1) ig2 = 1.0 / p->t3P.gam2;
    if (d > 2) ig3 = 1.0 / p->t3P.gam3;
#pragma omp parallel for num_threads(p->opts.nthreads) schedule(static)
    for (BIGINT j = 0; j < nj; ++j) {
      p->X[j] = (xj[j] - p->t3P.C1) * ig1; // rescale x_j
      if (d > 1) // (ok to do inside loop because of branch predict)
        p->Y[j] = (yj[j] - p->t3P.C2) * ig2;          // rescale y_j
      if (d > 2) p->Z[j] = (zj[j] - p->t3P.C3) * ig3; // rescale z_j
    }

    // set up prephase array...
    CPX imasign = (p->fftSign >= 0) ? IMA : -IMA; // +-i
    if (p->prephase) free(p->prephase);
    p->prephase = (CPX *)malloc(sizeof(CPX) * nj);
    if (p->t3P.D1 != 0.0 || p->t3P.D2 != 0.0 || p->t3P.D3 != 0.0) {
#pragma omp parallel for num_threads(p->opts.nthreads) schedule(static)
      for (BIGINT j = 0; j < nj; ++j) { // ... loop over src NU locs
        FLT phase = p->t3P.D1 * xj[j];
        if (d > 1) phase += p->t3P.D2 * yj[j];
        if (d > 2) phase += p->t3P.D3 * zj[j];
        p->prephase[j] = cos(phase) + imasign * sin(phase); // Euler
                                                            // e^{+-i.phase}
      }
    } else
      for (BIGINT j = 0; j < nj; ++j)
        p->prephase[j] = (CPX)1.0; // *** or keep flag so no mult in exec??

                                   // rescale the target s_k etc to s'_k etc...
#pragma omp parallel for num_threads(p->opts.nthreads) schedule(static)
    for (BIGINT k = 0; k < nk; ++k) {
      p->Sp[k] = p->t3P.h1 * p->t3P.gam1 * (s[k] - p->t3P.D1);   // so |s'_k| < pi/R
      if (d > 1)
        p->Tp[k] = p->t3P.h2 * p->t3P.gam2 * (t[k] - p->t3P.D2); // so |t'_k| <
                                                                 // pi/R
      if (d > 2)
        p->Up[k] = p->t3P.h3 * p->t3P.gam3 * (u[k] - p->t3P.D3); // so |u'_k| <
                                                                 // pi/R
    }

    // (old STEP 3a) Compute deconvolution post-factors array (per targ pt)...
    // (exploits that FT separates because kernel is prod of 1D funcs)
    if (p->deconv) free(p->deconv);
    p->deconv     = (CPX *)malloc(sizeof(CPX) * nk);
    FLT *phiHatk1 = (FLT *)malloc(sizeof(FLT) * nk);    // don't confuse w/ p->phiHat
    onedim_nuft_kernel(nk, p->Sp, phiHatk1, p->spopts); // fill phiHat1
    FLT *phiHatk2 = NULL, *phiHatk3 = NULL;
    if (d > 1) {
      phiHatk2 = (FLT *)malloc(sizeof(FLT) * nk);
      onedim_nuft_kernel(nk, p->Tp, phiHatk2, p->spopts); // fill phiHat2
    }
    if (d > 2) {
      phiHatk3 = (FLT *)malloc(sizeof(FLT) * nk);
      onedim_nuft_kernel(nk, p->Up, phiHatk3, p->spopts); // fill phiHat3
    }
    int Cfinite =
        isfinite(p->t3P.C1) && isfinite(p->t3P.C2) && isfinite(p->t3P.C3); // C can be nan
                                                                           // or inf if
                                                                           // M=0, no
                                                                           // input NU pts
    int Cnonzero = p->t3P.C1 != 0.0 || p->t3P.C2 != 0.0 || p->t3P.C3 != 0.0; // cen
#pragma omp parallel for num_threads(p->opts.nthreads) schedule(static)
    for (BIGINT k = 0; k < nk; ++k) { // .... loop over NU targ freqs
      FLT phiHat = phiHatk1[k];
      if (d > 1) phiHat *= phiHatk2[k];
      if (d > 2) phiHat *= phiHatk3[k];
      p->deconv[k] = (CPX)(1.0 / phiHat);
      if (Cfinite && Cnonzero) {
        FLT phase = (s[k] - p->t3P.D1) * p->t3P.C1;
        if (d > 1) phase += (t[k] - p->t3P.D2) * p->t3P.C2;
        if (d > 2) phase += (u[k] - p->t3P.D3) * p->t3P.C3;
        p->deconv[k] *= cos(phase) + imasign * sin(phase); // Euler e^{+-i.phase}
      }
    }
    free(phiHatk1);
    free(phiHatk2);
    free(phiHatk3); // done w/ deconv fill
    if (p->opts.debug)
      printf("[%s t3] phase & deconv factors:\t%.3g s\n", __func__, timer.elapsedsec());

    // Set up sort for spreading Cp (from primed NU src pts X, Y, Z) to fw...
    timer.restart();
    // Free sortIndices if it has been allocated before in case of repeated setpts
    // calls causing memory leak. We don't know it is the same size as before, so we
    // have to malloc each time.
    if (p->sortIndices) free(p->sortIndices);
    p->sortIndices = (BIGINT *)malloc(sizeof(BIGINT) * p->nj);
    if (!p->sortIndices) {
      fprintf(stderr, "[%s t3] failed to allocate sortIndices!\n", __func__);
      return FINUFFT_ERR_SPREAD_ALLOC;
    }
    p->didSort = indexSort(p->sortIndices, p->nf1, p->nf2, p->nf3, p->nj, p->X, p->Y,
                           p->Z, p->spopts);
    if (p->opts.debug)
      printf("[%s t3] sort (didSort=%d):\t\t%.3g s\n", __func__, p->didSort,
             timer.elapsedsec());

    // Plan and setpts once, for the (repeated) inner type 2 finufft call...
    timer.restart();
    BIGINT t2nmodes[]   = {p->nf1, p->nf2, p->nf3};  // t2 input is actually fw
    finufft_opts t2opts = p->opts;                   // deep copy, since not ptrs
    t2opts.modeord      = 0;                         // needed for correct t3!
    t2opts.debug        = max(0, p->opts.debug - 1); // don't print as much detail
    t2opts.spread_debug = max(0, p->opts.spread_debug - 1);
    t2opts.showwarn     = 0;                         // so don't see warnings 2x
    // (...could vary other t2opts here?)
    if (p->innerT2plan) FINUFFT_DESTROY(p->innerT2plan);
    int ier = FINUFFT_MAKEPLAN(2, d, t2nmodes, p->fftSign, p->batchSize, p->tol,
                               &p->innerT2plan, &t2opts);
    if (ier > 1) { // if merely warning, still proceed
      fprintf(stderr, "[%s t3]: inner type 2 plan creation failed with ier=%d!\n",
              __func__, ier);
      return ier;
    }
    ier = FINUFFT_SETPTS(p->innerT2plan, nk, p->Sp, p->Tp, p->Up, 0, NULL, NULL,
                         NULL); // note nk = # output points (not nj)
    if (ier > 1) {
      fprintf(stderr, "[%s t3]: inner type 2 setpts failed, ier=%d!\n", __func__, ier);
      return ier;
    }
    if (p->opts.debug)
      printf("[%s t3] inner t2 plan & setpts: \t%.3g s\n", __func__, timer.elapsedsec());
  }
  return 0;
}
// ............ end setpts ..................................................

// EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
int FINUFFT_EXECUTE(FINUFFT_PLAN p, CPX *cj, CPX *fk) {
  /* See ../docs/cguru.doc for current documentation.

   For given (stack of) weights cj or coefficients fk, performs NUFFTs with
   existing (sorted) NU pts and existing plan.
   For type 1 and 3: cj is input, fk is output.
   For type 2: fk is input, cj is output.
   Performs spread/interp, pre/post deconvolve, and FFT as appropriate
   for each of the 3 types.
   For cases of ntrans>1, performs work in blocks of size up to batchSize.
   Return value 0 (no error diagnosis yet).
   Barnett 5/20/20, based on Malleo 2019.
*/
  CNTime timer;
  timer.start();

  if (p->type != 3) { // --------------------- TYPE 1,2 EXEC ------------------

    double t_sprint = 0.0, t_fft = 0.0, t_deconv = 0.0; // accumulated timing
    if (p->opts.debug)
      printf("[%s] start ntrans=%d (%d batches, bsize=%d)...\n", __func__, p->ntrans,
             p->nbatch, p->batchSize);

    for (int b = 0; b * p->batchSize < p->ntrans; b++) { // .....loop b over batches

      // current batch is either batchSize, or possibly truncated if last one
      int thisBatchSize = min(p->ntrans - b * p->batchSize, p->batchSize);
      int bB            = b * p->batchSize; // index of vector, since batchsizes same
      CPX *cjb          = cj + bB * p->nj;  // point to batch of weights
      CPX *fkb          = fk + bB * p->N;   // point to batch of mode coeffs
      if (p->opts.debug > 1)
        printf("[%s] start batch %d (size %d):\n", __func__, b, thisBatchSize);

      // STEP 1: (varies by type)
      timer.restart();
      if (p->type == 1) { // type 1: spread NU pts p->X, weights cj, to fw grid
        spreadinterpSortedBatch(thisBatchSize, p, cjb);
        t_sprint += timer.elapsedsec();
      } else { //  type 2: amplify Fourier coeffs fk into 0-padded fw
        deconvolveBatch(thisBatchSize, p, fkb);
        t_deconv += timer.elapsedsec();
      }

      // STEP 2: call the FFT on this batch
      timer.restart();
      do_fft(p);
      t_fft += timer.elapsedsec();
      if (p->opts.debug > 1) printf("\tFFT exec:\t\t%.3g s\n", timer.elapsedsec());

      // STEP 3: (varies by type)
      timer.restart();
      if (p->type == 1) { // type 1: deconvolve (amplify) fw and shuffle to fk
        deconvolveBatch(thisBatchSize, p, fkb);
        t_deconv += timer.elapsedsec();
      } else { // type 2: interpolate unif fw grid to NU target pts
        spreadinterpSortedBatch(thisBatchSize, p, cjb);
        t_sprint += timer.elapsedsec();
      }
    } // ........end b loop

    if (p->opts.debug) { // report total times in their natural order...
      if (p->type == 1) {
        printf("[%s] done. tot spread:\t\t%.3g s\n", __func__, t_sprint);
        printf("               tot FFT:\t\t\t\t%.3g s\n", t_fft);
        printf("               tot deconvolve:\t\t\t%.3g s\n", t_deconv);
      } else {
        printf("[%s] done. tot deconvolve:\t\t%.3g s\n", __func__, t_deconv);
        printf("               tot FFT:\t\t\t\t%.3g s\n", t_fft);
        printf("               tot interp:\t\t\t%.3g s\n", t_sprint);
      }
    }
  }

  else { // ----------------------------- TYPE 3 EXEC ---------------------

    // for (BIGINT j=0;j<10;++j) printf("\tcj[%ld]=%.15g+%.15gi\n",(long
    // int)j,(double)real(cj[j]),(double)imag(cj[j]));  // debug

    double t_pre = 0.0, t_spr = 0.0, t_t2 = 0.0,
           t_deconv = 0.0; // accumulated timings
    if (p->opts.debug)
      printf("[%s t3] start ntrans=%d (%d batches, bsize=%d)...\n", __func__, p->ntrans,
             p->nbatch, p->batchSize);

    for (int b = 0; b * p->batchSize < p->ntrans; b++) { // .....loop b over batches

      // batching and pointers to this batch, identical to t1,2 above...
      int thisBatchSize = min(p->ntrans - b * p->batchSize, p->batchSize);
      int bB            = b * p->batchSize;
      CPX *cjb          = cj + bB * p->nj; // batch of input strengths
      CPX *fkb          = fk + bB * p->nk; // batch of output strengths
      if (p->opts.debug > 1)
        printf("[%s t3] start batch %d (size %d):\n", __func__, b, thisBatchSize);

      // STEP 0: pre-phase (possibly) the c_j input strengths into c'_j batch...
      timer.restart();
#pragma omp parallel for num_threads(p->opts.nthreads) // or p->batchSize?
      for (int i = 0; i < thisBatchSize; i++) {
        BIGINT ioff = i * p->nj;
        for (BIGINT j = 0; j < p->nj; ++j)
          p->CpBatch[ioff + j] = p->prephase[j] * cjb[ioff + j];
      }
      t_pre += timer.elapsedsec();

      // STEP 1: spread c'_j batch (x'_j NU pts) into fw batch grid...
      timer.restart();
      p->spopts.spread_direction = 1;                        // spread
      spreadinterpSortedBatch(thisBatchSize, p, p->CpBatch); // p->X are primed
      t_spr += timer.elapsedsec();

      // for (int j=0;j<p->nf1;++j)
      // printf("fw[%d]=%.3g+%.3gi\n",j,p->fwBatch[j][0],p->fwBatch[j][1]);  //
      // debug

      // STEP 2: type 2 NUFFT from fw batch to user output fk array batch...
      timer.restart();
      // illegal possible shrink of ntrans *after* plan for smaller last batch:
      p->innerT2plan->ntrans = thisBatchSize; // do not try this at home!
      /* (alarming that FFT not shrunk, but safe, because t2's fwBatch array
     still the same size, as Andrea explained; just wastes a few flops) */
      FINUFFT_EXECUTE(p->innerT2plan, fkb, p->fwBatch);
      t_t2 += timer.elapsedsec();

      // STEP 3: apply deconvolve (precomputed 1/phiHat(targ_k), phasing too)...
      timer.restart();
#pragma omp parallel for num_threads(p->opts.nthreads)
      for (int i = 0; i < thisBatchSize; i++) {
        BIGINT ioff = i * p->nk;
        for (BIGINT k = 0; k < p->nk; ++k) fkb[ioff + k] *= p->deconv[k];
      }
      t_deconv += timer.elapsedsec();
    } // ........end b loop

    if (p->opts.debug) { // report total times in their natural order...
      printf("[%s t3] done. tot prephase:\t\t%.3g s\n", __func__, t_pre);
      printf("                  tot spread:\t\t\t%.3g s\n", t_spr);
      printf("                  tot type 2:\t\t\t%.3g s\n", t_t2);
      printf("                  tot deconvolve:\t\t%.3g s\n", t_deconv);
    }
  }
  // for (BIGINT k=0;k<10;++k) printf("\tfk[%ld]=%.15g+%.15gi\n",(long
  // int)k,(double)real(fk[k]),(double)imag(fk[k]));  // debug

  return 0;
}

// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
int FINUFFT_DESTROY(FINUFFT_PLAN p)
// Free everything we allocated inside of finufft_plan pointed to by p.
// Also must not crash if called immediately after finufft_makeplan.
// Thus either each thing free'd here is guaranteed to be NULL or correctly
// allocated.
{
  if (!p) // NULL ptr, so not a ptr to a plan, report error
    return 1;

#ifdef FINUFFT_USE_DUCC0
  free(p->fwBatch); // free the big FFTW (or t3 spread) working array
#else
  FFTW_FR(p->fwBatch); // free the big FFTW (or t3 spread) working array
#endif
  free(p->sortIndices);
  if (p->type == 1 || p->type == 2) {
#ifndef FINUFFT_USE_DUCC0
    {
      std::lock_guard<std::mutex> lock(fftw_lock);
      FFTW_DE(p->fftwPlan);
    }
#endif
    free(p->phiHat1);
    free(p->phiHat2);
    free(p->phiHat3);
  } else {                           // free the stuff alloc for type 3 only
    FINUFFT_DESTROY(p->innerT2plan); // if NULL, ignore its error code
    free(p->CpBatch);
    free(p->Sp);
    free(p->Tp);
    free(p->Up);
    free(p->X);
    free(p->Y);
    free(p->Z);
    free(p->prephase);
    free(p->deconv);
  }
  delete p;
  return 0; // success
}
