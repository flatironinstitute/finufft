#include <finufft/fft.h>
#include <finufft/finufft_core.h>
#include <finufft/spreadinterp.h>
#include <finufft/utils.h>

#include "../contrib/legendre_rule_fast.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

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
   are in c_interface.cpp.

   As of v2.3.1 the plan object is a class with constructor and methods.
   (mostly done by Martin Reinecke, 2024).

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

* Thread-safety: FINUFFT plans are passed as pointers, so it has no global
  state apart from (if FFTW used) that associated with FFTW (and did_fftw_init).
*/

// ---------- local math routines (were in common.cpp; no need now): --------

namespace finufft {
namespace common {

static int set_nf_type12(BIGINT ms, const finufft_opts &opts,
                         const finufft_spread_opts &spopts, BIGINT *nf)
// Type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms. Returns 0 if success, else an
// error code if nf was unreasonably big (& tell the world).
{
  *nf = BIGINT(opts.upsampfac * double(ms)); // manner of rounding not crucial
  if (*nf < 2 * spopts.nspread) *nf = 2 * spopts.nspread; // otherwise spread fails
  if (*nf < MAX_NF) {
    *nf = next235even(*nf);                               // expensive at huge nf
    return 0;
  } else {
    fprintf(stderr,
            "[%s] nf=%.3g exceeds MAX_NF of %.3g, so exit without attempting "
            "memory allocation\n",
            __func__, (double)*nf, (double)MAX_NF);
    return FINUFFT_ERR_MAXNALLOC;
  }
}

template<typename T>
static int setup_spreader_for_nufft(finufft_spread_opts &spopts, T eps,
                                    const finufft_opts &opts, int dim)
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

template<typename T>
static void set_nhg_type3(T S, T X, const finufft_opts &opts,
                          const finufft_spread_opts &spopts, BIGINT *nf, T *h, T *gam)
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
  int nss = spopts.nspread + 1; // since ns may be odd
  T Xsafe = X, Ssafe = S;       // may be tweaked locally
  if (X == 0.0)                 // logic ensures XS>=1, handle X=0 a/o S=0
    if (S == 0.0) {
      Xsafe = 1.0;
      Ssafe = 1.0;
    } else
      Xsafe = std::max(Xsafe, 1 / S);
  else
    Ssafe = std::max(Ssafe, 1 / X);
  // use the safe X and S...
  auto nfd = T(2.0 * opts.upsampfac * Ssafe * Xsafe / PI + nss);
  if (!std::isfinite(nfd)) nfd = 0.0; // use T to catch inf
  *nf = (BIGINT)nfd;
  // printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  //  catch too small nf, and nan or +-inf, otherwise spread fails...
  if (*nf < 2 * spopts.nspread) *nf = 2 * spopts.nspread;
  if (*nf < MAX_NF)                               // otherwise will fail anyway
    *nf = next235even(*nf);                       // expensive at huge nf
  *h   = T(2.0 * PI / *nf);                       // upsampled grid spacing
  *gam = T(*nf / (2.0 * opts.upsampfac * Ssafe)); // x scale fac to x'
}

template<typename T>
static void onedim_fseries_kernel(BIGINT nf, std::vector<T> &fwkerhalf,
                                  const finufft_spread_opts &opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.
  The kernel is actually centered at nf/2, related to the centering of the grid;
  this is now achieved by the sign flip in a[n] below.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
        divided by h = 2pi/n.
        (should be allocated for at least nf/2+1 Ts)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18.
  Fixed num_threads 7/20/20. Reduced rounding error in a[n] calc 8/20/24.
  To do (Nov 2024): replace evaluate_kernel by evaluate_kernel_horner.
 */
{
  T J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 3.0 * J2); // not sure why so large? cannot exceed MAX_NQUAD
  T f[MAX_NQUAD];
  double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
  legendre_compute_glr(2 * q, z, w); // only half the nodes used, eg on (0,1)
  std::complex<T> a[MAX_NQUAD];
  for (int n = 0; n < q; ++n) {      // set up nodes z_n and vals f_n
    z[n] *= J2;                      // rescale nodes
    f[n] = J2 * (T)w[n] * evaluate_kernel((T)z[n], opts); // vals & quadr wei
    a[n] = -std::exp(2 * PI * std::complex<double>(0, 1) * z[n] / double(nf)); // phase
                                                                               // winding
                                                                               // rates
  }
  BIGINT nout = nf / 2 + 1;                            // how many values we're writing to
  int nt      = std::min(nout, (BIGINT)opts.nthreads); // how many chunks
  std::vector<BIGINT> brk(nt + 1);                     // start indices for each thread
  for (int t = 0; t <= nt; ++t) // split nout mode indices btw threads
    brk[t] = (BIGINT)(0.5 + nout * t / (double)nt);
#pragma omp parallel num_threads(nt)
  {                                                // each thread gets own chunk to do
    int t = MY_OMP_GET_THREAD_NUM();
    std::complex<T> aj[MAX_NQUAD];                 // phase rotator for this thread
    for (int n = 0; n < q; ++n)
      aj[n] = std::pow(a[n], (T)brk[t]);           // init phase factors for chunk
    for (BIGINT j = brk[t]; j < brk[t + 1]; ++j) { // loop along output array
      T x = 0.0;                                   // accumulator for answer at this j
      for (int n = 0; n < q; ++n) {
        x += f[n] * 2 * real(aj[n]);               // include the negative freq
        aj[n] *= a[n];                             // wind the phases
      }
      fwkerhalf[j] = x;
    }
  }
}

template<typename T>
static void onedim_nuft_kernel(BIGINT nk, const std::vector<T> &k, std::vector<T> &phihat,
                               const finufft_spread_opts &opts)
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
  phihat - real Fourier transform evaluated at freqs (alloc for nk Ts)

  Barnett 2/8/17. openmp since cos slow 2/9/17.
  To do (Nov 2024): replace evaluate_kernel by evaluate_kernel_horner.
 */
{
  T J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 2.0 * J2); // > pi/2 ratio.  cannot exceed MAX_NQUAD
  if (opts.debug) printf("q (# ker FT quadr pts) = %d\n", q);
  T f[MAX_NQUAD];
  double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD]; // glr needs double
  legendre_compute_glr(2 * q, z, w);         // only half the nodes used, eg on (0,1)
  for (int n = 0; n < q; ++n) {
    z[n] *= (T)J2;                           // quadr nodes for [0,J/2]
    f[n] = J2 * (T)w[n] * evaluate_kernel((T)z[n], opts); // w/ quadr weights
  }
#pragma omp parallel for num_threads(opts.nthreads)
  for (BIGINT j = 0; j < nk; ++j) {        // loop along output array
    T x = 0.0;                             // register
    for (int n = 0; n < q; ++n)
      x += f[n] * 2 * cos(k[j] * (T)z[n]); // pos & neg freq pair.  use T cos!
    phihat[j] = x;
  }
}

template<typename T>
static void deconvolveshuffle1d(int dir, T prefac, const std::vector<T> &ker, BIGINT ms,
                                T *fk, BIGINT nf1, std::complex<T> *fw, int modeord)
/*
  if dir==1: copies fw to fk with amplification by prefac/ker
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (from -N/2 up to N/2-1)
      1: use FFT-style (from 0 to N/2-1, then -N/2 up to -1).

  fk is a size-ms T complex array (2*ms Ts alternating re,im parts)
  fw is a size-nf1 complex array (2*nf1 Ts alternating re,im parts)
  ker is real-valued T array of length nf1/2+1.

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

template<typename T>
static void deconvolveshuffle2d(int dir, T prefac, const std::vector<T> &ker1,
                                const std::vector<T> &ker2, BIGINT ms, BIGINT mt, T *fk,
                                BIGINT nf1, BIGINT nf2, std::complex<T> *fw, int modeord)
/*
  2D version of deconvolveshuffle1d, calls it on each x-line using 1/ker2 fac.

  if dir==1: copies fw to fk with amplification by prefac/(ker1(k1)*ker2(k2)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (each dim increasing)
      1: use FFT-style (pos then negative, on each dim)

  fk is a complex array stored as 2*ms*mt Ts alternating re,im parts, with
  ms looped over fast and mt slow.
  fw is a complex array stored as 2*nf1*nf2] Ts alternating re,im parts, with
  nf1 looped over fast and nf2 slow.
  ker1, ker2 are real-valued T arrays of lengths nf1/2+1, nf2/2+1
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

template<typename T>
static void deconvolveshuffle3d(int dir, T prefac, std::vector<T> &ker1,
                                std::vector<T> &ker2, std::vector<T> &ker3, BIGINT ms,
                                BIGINT mt, BIGINT mu, T *fk, BIGINT nf1, BIGINT nf2,
                                BIGINT nf3, std::complex<T> *fw, int modeord)
/*
  3D version of deconvolveshuffle2d, calls it on each xy-plane using 1/ker3 fac.

  if dir==1: copies fw to fk with ampl by prefac/(ker1(k1)*ker2(k2)*ker3(k3)).
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (each dim increasing)
      1: use FFT-style (pos then negative, on each dim)

  fk is a complex array stored as 2*ms*mt*mu Ts alternating re,im parts, with
  ms looped over fastest and mu slowest.
  fw is a complex array stored as 2*nf1*nf2*nf3 Ts alternating re,im parts, with
  nf1 looped over fastest and nf3 slowest.
  ker1, ker2, ker3 are real-valued T arrays of lengths nf1/2+1, nf2/2+1,
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

template<typename T>
static int spreadinterpSortedBatch(int batchSize, FINUFFT_PLAN_T<T> *p,
                                   std::complex<T> *cBatch)
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
    std::complex<T> *fwi = p->fwBatch.data() + i * p->nf; // start of i'th fw array in
                                                          // wkspace
    std::complex<T> *ci = cBatch + i * p->nj; // start of i'th c array in cBatch
    spreadinterpSorted(p->sortIndices, p->nf1, p->nf2, p->nf3, (T *)fwi, p->nj, p->X,
                       p->Y, p->Z, (T *)ci, p->spopts, p->didSort);
  }
  return 0;
}

template<typename T>
static int deconvolveBatch(int batchSize, FINUFFT_PLAN_T<T> *p, std::complex<T> *fkBatch)
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
    std::complex<T> *fwi = p->fwBatch.data() + i * p->nf; // start of i'th fw array in
                                                          // wkspace
    std::complex<T> *fki = fkBatch + i * p->N; // start of i'th fk array in fkBatch

    // Call routine from common.cpp for the dim; prefactors hardcoded to 1.0...
    if (p->dim == 1)
      deconvolveshuffle1d(p->spopts.spread_direction, T(1), p->phiHat1, p->ms, (T *)fki,
                          p->nf1, fwi, p->opts.modeord);
    else if (p->dim == 2)
      deconvolveshuffle2d(p->spopts.spread_direction, T(1), p->phiHat1, p->phiHat2, p->ms,
                          p->mt, (T *)fki, p->nf1, p->nf2, fwi, p->opts.modeord);
    else
      deconvolveshuffle3d(p->spopts.spread_direction, T(1), p->phiHat1, p->phiHat2,
                          p->phiHat3, p->ms, p->mt, p->mu, (T *)fki, p->nf1, p->nf2,
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
void finufft_default_opts_t(finufft_opts *o)
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
  o->spreadinterponly = 0;
  o->maxbatchsize       = 0;
  o->spread_nthr_atomic = -1;
  o->spread_max_sp_size = 0;
  o->fftw_lock_fun      = nullptr;
  o->fftw_unlock_fun    = nullptr;
  o->fftw_lock_data     = nullptr;
  // sphinx tag (don't remove): @defopts_end
}

// PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
template<typename TF>
FINUFFT_PLAN_T<TF>::FINUFFT_PLAN_T(int type_, int dim_, const BIGINT *n_modes, int iflag,
                                   int ntrans_, TF tol_, finufft_opts *opts_, int &ier)
    : type(type_), dim(dim_), ntrans(ntrans_), tol(tol_)
// Constructor for finufft_plan object.
// opts is ptr to a finufft_opts to set options, or nullptr to use defaults.
// For some of the fields (if "auto" selected) here choose the actual setting.
// For types 1,2 allocates memory for internal working arrays,
// evaluates spreading kernel coefficients, and does FFT plan if needed.
// ier is an output written to pass out warning codes (errors now thrown in C++ style).
{
  if (!opts_)      // use default opts
    finufft_default_opts_t(&opts);
  else             // or read from what's passed in
    opts = *opts_; // keep a deep copy; changing *opts_ now has no effect

  if (opts.debug)  // do a hello world
    printf("[%s] new plan: FINUFFT version " FINUFFT_VER " .................\n",
           __func__);

  fftPlan = std::make_unique<Finufft_FFT_plan<TF>>(
      opts.fftw_lock_fun, opts.fftw_unlock_fun, opts.fftw_lock_data);

  if ((type != 1) && (type != 2) && (type != 3)) {
    fprintf(stderr, "[%s] Invalid type (%d), should be 1, 2 or 3.\n", __func__, type);
    throw int(FINUFFT_ERR_TYPE_NOTVALID);
  }
  if ((dim != 1) && (dim != 2) && (dim != 3)) {
    fprintf(stderr, "[%s] Invalid dim (%d), should be 1, 2 or 3.\n", __func__, dim);
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
  if (ntrans < 1) {
    fprintf(stderr, "[%s] ntrans (%d) should be at least 1.\n", __func__, ntrans);
    throw int(FINUFFT_ERR_NTRANS_NOTVALID);
  }
  if (!opts.fftw_lock_fun != !opts.fftw_unlock_fun) {
    fprintf(stderr, "[%s] fftw_(un)lock functions should be both null or both set\n",
            __func__);
    throw int(FINUFFT_ERR_LOCK_FUNS_INVALID);
  }

  // get stuff from args...
  fftSign = (iflag >= 0) ? 1 : -1; // clean up flag input

                                   // choose overall # threads...
#ifdef _OPENMP
  int ompmaxnthr = MY_OMP_GET_MAX_THREADS();
  int nthr       = ompmaxnthr; // default: use as many as OMP gives us
  // (the above could be set, or suggested set, to 1 for small enough problems...)
  if (opts.nthreads > 0) {
    nthr = opts.nthreads; // user override, now without limit
    if (opts.showwarn && (nthr > ompmaxnthr))
      fprintf(stderr,
              "%s warning: using opts.nthreads=%d, more than the %d OpenMP claims "
              "available; note large nthreads can be slower.\n",
              __func__, nthr, ompmaxnthr);
  }
#else
  int nthr = 1; // always 1 thread (avoid segfault)
  if (opts.nthreads > 1)
    fprintf(stderr,
            "%s warning: opts.nthreads=%d but library is single-threaded; ignoring!\n",
            __func__, opts.nthreads);
#endif
  opts.nthreads = nthr; // store actual # thr planned for
  // (this sets/limits all downstream spread/interp, 1dkernel, and FFT thread counts...)

  // choose batchSize for types 1,2 or 3... (uses int ceil(b/a)=1+(b-1)/a trick)
  if (opts.maxbatchsize == 0) {                        // logic to auto-set best batchsize
    nbatch    = 1 + (ntrans - 1) / nthr;               // min # batches poss
    batchSize = 1 + (ntrans - 1) / nbatch;             // then cut # thr in each b
  } else {                                             // batchSize override by user
    batchSize = std::min(opts.maxbatchsize, ntrans);
    nbatch    = 1 + (ntrans - 1) / batchSize;          // resulting # batches
  }
  if (opts.spread_thread == 0) opts.spread_thread = 2; // our auto choice
  if (opts.spread_thread != 1 && opts.spread_thread != 2) {
    fprintf(stderr, "[%s] illegal opts.spread_thread!\n", __func__);
    throw int(FINUFFT_ERR_SPREAD_THREAD_NOTVALID);
  }

  if (type != 3) {                   // read in user Fourier mode array sizes...
    ms = n_modes[0];
    mt = (dim > 1) ? n_modes[1] : 1; // leave as 1 for unused dims
    mu = (dim > 2) ? n_modes[2] : 1;
    N  = ms * mt * mu;               // N = total # modes
  }

  // heuristic to choose default upsampfac... (currently two poss)
  if (opts.upsampfac == 0.0) {            // indicates auto-choose
    opts.upsampfac = 2.0;                 // default, and need for tol small
    if (tol >= (TF)1E-9) {                // the tol sigma=5/4 can reach
      if (type == 3)                      // could move to setpts, more known?
        opts.upsampfac = 1.25;            // faster b/c smaller RAM & FFT
      else if ((dim == 1 && N > 10000000) || (dim == 2 && N > 300000) ||
               (dim == 3 && N > 3000000)) // type 1,2 heuristic cutoffs, double,
                                          // typ tol, 12-core xeon
        opts.upsampfac = 1.25;
    }
    if (opts.debug > 1)
      printf("[%s] set auto upsampfac=%.2f\n", __func__, opts.upsampfac);
  }
  // use opts to choose and write into plan's spread options...
  ier = setup_spreader_for_nufft(spopts, tol, opts, dim);
  if (ier > 1) // proceed if success or warning
    throw int(ier);

  //  ------------------------ types 1,2: planning needed ---------------------
  if (type == 1 || type == 2) {

    int nthr_fft = nthr; // give FFT all threads (or use o.spread_thread?)
                         // Note: batchSize not used since might be only 1.

    spopts.spread_direction = type;

    constexpr TF EPSILON = std::numeric_limits<TF>::epsilon();
    if (opts.showwarn) { // user warn round-off error...
      if (EPSILON * ms > 1.0)
        fprintf(stderr, "%s warning: rounding err predicted eps_mach*N1 = %.3g > 1 !\n",
                __func__, (double)(EPSILON * ms));
      if (EPSILON * mt > 1.0)
        fprintf(stderr, "%s warning: rounding err predicted eps_mach*N2 = %.3g > 1 !\n",
                __func__, (double)(EPSILON * mt));
      if (EPSILON * mu > 1.0)
        fprintf(stderr, "%s warning: rounding err predicted eps_mach*N3 = %.3g > 1 !\n",
                __func__, (double)(EPSILON * mu));
    }

    // determine fine grid sizes, sanity check..
    int nfier = set_nf_type12(ms, opts, spopts, &nf1);
    if (nfier) throw nfier; // nf too big; we're done
    phiHat1.resize(nf1 / 2 + 1);
    if (dim > 1) {
      nfier = set_nf_type12(mt, opts, spopts, &nf2);
      if (nfier) throw nfier;
      phiHat2.resize(nf2 / 2 + 1);
    }
    if (dim > 2) {
      nfier = set_nf_type12(mu, opts, spopts, &nf3);
      if (nfier) throw nfier;
      phiHat3.resize(nf3 / 2 + 1);
    }

    if (opts.debug) { // "long long" here is to avoid warnings with printf...
      printf("[%s] %dd%d: (ms,mt,mu)=(%lld,%lld,%lld) "
             "(nf1,nf2,nf3)=(%lld,%lld,%lld)\n               ntrans=%d nthr=%d "
             "batchSize=%d ",
             __func__, dim, type, (long long)ms, (long long)mt, (long long)mu,
             (long long)nf1, (long long)nf2, (long long)nf3, ntrans, nthr, batchSize);
      if (batchSize == 1) // spread_thread has no effect in this case
        printf("\n");
      else
        printf(" spread_thread=%d\n", opts.spread_thread);
    }

    // STEP 0: get Fourier coeffs of spreading kernel along each fine grid dim
    CNTime timer;
    timer.start();
    onedim_fseries_kernel(nf1, phiHat1, spopts);
    if (dim > 1) onedim_fseries_kernel(nf2, phiHat2, spopts);
    if (dim > 2) onedim_fseries_kernel(nf3, phiHat3, spopts);
    if (opts.debug)
      printf("[%s] kernel fser (ns=%d):\t\t%.3g s\n", __func__, spopts.nspread,
             timer.elapsedsec());

    nf = nf1 * nf2 * nf3; // fine grid total number of points
    if (nf * batchSize > MAX_NF) {
      fprintf(
          stderr,
          "[%s] fwBatch would be bigger than MAX_NF, not attempting memory allocation!\n",
          __func__);
      throw int(FINUFFT_ERR_MAXNALLOC);
    }

    timer.restart();
    fwBatch.resize(nf * batchSize); // the big workspace
    if (opts.debug)
      printf("[%s] fwBatch %.2fGB alloc:   \t%.3g s\n", __func__,
             (double)1E-09 * sizeof(std::complex<TF>) * nf * batchSize,
             timer.elapsedsec());

    timer.restart(); // plan the FFTW
    const auto ns = gridsize_for_fft(this);
    fftPlan->plan(ns, batchSize, fwBatch.data(), fftSign, opts.fftw, nthr_fft);
    if (opts.debug)
      printf("[%s] FFT plan (mode %d, nthr=%d):\t%.3g s\n", __func__, opts.fftw, nthr_fft,
             timer.elapsedsec());

  } else { // -------------------------- type 3 (no planning) ------------

    if (opts.debug) printf("[%s] %dd%d: ntrans=%d\n", __func__, dim, type, ntrans);
    // Type 3 will call finufft_makeplan for type 2; no need to init FFTW
    // Note we don't even know nj or nk yet, so can't do anything else!
  }
  if (ier > 1) throw int(ier); // report setup_spreader status (could be warning)
}

template<typename TF>
int finufft_makeplan_t(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans,
                       TF tol, FINUFFT_PLAN_T<TF> **pp, finufft_opts *opts)
// C-API wrapper around the C++ constructor. Writes a pointer to the plan in *pp.
// Returns ier (warning or error codes as per C interface).
{
  *pp     = nullptr;
  int ier = 0;
  try {
    *pp = new FINUFFT_PLAN_T<TF>(type, dim, n_modes, iflag, ntrans, tol, opts, ier);
  } catch (int errcode) {
    return errcode;
  }
  return ier;
}

// For this function and the following ones (i.e. everything that is accessible
// from outside), we need to state for which data types we want the template
// to be instantiated. At the current location in the code, the compiler knows
// how exactly it can construct the function "finufft_makeplan_t" for any given
// type TF, but it doesn't know for which types it actually should do so.
// The following two statements instruct it to do that for TF=float and
// TF=double :  (Reinecke, Sept 2024)
template int finufft_makeplan_t<float>(int type, int dim, const BIGINT *n_modes,
                                       int iflag, int ntrans, float tol,
                                       FINUFFT_PLAN_T<float> **pp, finufft_opts *opts);
template int finufft_makeplan_t<double>(int type, int dim, const BIGINT *n_modes,
                                        int iflag, int ntrans, double tol,
                                        FINUFFT_PLAN_T<double> **pp, finufft_opts *opts);

// SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
template<typename TF>
int FINUFFT_PLAN_T<TF>::setpts(BIGINT nj, TF *xj, TF *yj, TF *zj, BIGINT nk, TF *s, TF *t,
                               TF *u) {
  // Method function to set NU points and do precomputations. Barnett 2020.
  // See ../docs/cguru.doc for current documentation.
  int d = dim; // abbrev for spatial dim
  CNTime timer;
  timer.start();
  this->nj = nj; // the user only now chooses how many NU (x,y,z) pts
  if (nj < 0) {
    fprintf(stderr, "[%s] nj (%lld) cannot be negative!\n", __func__, (long long)nj);
    return FINUFFT_ERR_NUM_NU_PTS_INVALID;
  } else if (nj > MAX_NU_PTS) {
    fprintf(stderr, "[%s] nj (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nj);
    return FINUFFT_ERR_NUM_NU_PTS_INVALID;
  }

  if (type != 3) { // ------------------ TYPE 1,2 SETPTS -------------------
                   // (all we can do is check and maybe bin-sort the NU pts)
    X       = xj;  // plan must keep pointers to user's fixed NU pts
    Y       = yj;
    Z       = zj;
    int ier = spreadcheck(nf1, nf2, nf3, nj, xj, yj, zj, spopts);
    if (opts.debug > 1)
      printf("[%s] spreadcheck (%d):\t%.3g s\n", __func__, spopts.chkbnds,
             timer.elapsedsec());
    if (ier) // no warnings allowed here
      return ier;
    timer.restart();
    sortIndices.resize(nj);
    didSort = indexSort(sortIndices, nf1, nf2, nf3, nj, xj, yj, zj, spopts);
    if (opts.debug)
      printf("[%s] sort (didSort=%d):\t\t%.3g s\n", __func__, didSort,
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
    this->nk = nk; // user set # targ freq pts
    S        = s;  // keep pointers to user's input target pts
    T        = t;
    U        = u;

    // pick x, s intervals & shifts & # fine grid pts (nf) in each dim...
    TF S1 = 0, S2 = 0, S3 = 0; // get half-width X, center C, which contains {x_j}...
    arraywidcen(nj, xj, &(t3P.X1), &(t3P.C1));
    arraywidcen(nk, s, &S1, &(t3P.D1)); // same D, S, but for {s_k}
    set_nhg_type3(S1, t3P.X1, opts, spopts, &(nf1), &(t3P.h1),
                  &(t3P.gam1));         // applies twist i)
    t3P.C2 = 0.0;                       // their defaults if dim 2 unused, etc
    t3P.D2 = 0.0;
    if (d > 1) {
      arraywidcen(nj, yj, &(t3P.X2), &(t3P.C2)); // {y_j}
      arraywidcen(nk, t, &S2, &(t3P.D2));        // {t_k}
      set_nhg_type3(S2, t3P.X2, opts, spopts, &(nf2), &(t3P.h2), &(t3P.gam2));
    }
    t3P.C3 = 0.0;
    t3P.D3 = 0.0;
    if (d > 2) {
      arraywidcen(nj, zj, &(t3P.X3), &(t3P.C3)); // {z_j}
      arraywidcen(nk, u, &S3, &(t3P.D3));        // {u_k}
      set_nhg_type3(S3, t3P.X3, opts, spopts, &(nf3), &(t3P.h3), &(t3P.gam3));
    }

    if (opts.debug) { // report on choices of shifts, centers, etc...
      printf("\tM=%lld N=%lld\n", (long long)nj, (long long)nk);
      printf("\tX1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%lld h1=%.3g\t\n", t3P.X1,
             t3P.C1, S1, t3P.D1, t3P.gam1, (long long)nf1, t3P.h1);
      if (d > 1)
        printf("\tX2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%lld h2=%.3g\n", t3P.X2,
               t3P.C2, S2, t3P.D2, t3P.gam2, (long long)nf2, t3P.h2);
      if (d > 2)
        printf("\tX3=%.3g C3=%.3g S3=%.3g D3=%.3g gam3=%g nf3=%lld h3=%.3g\n", t3P.X3,
               t3P.C3, S3, t3P.D3, t3P.gam3, (long long)nf3, t3P.h3);
    }
    nf = nf1 * nf2 * nf3; // fine grid total number of points
    if (nf * batchSize > MAX_NF) {
      fprintf(stderr,
              "[%s t3] fwBatch would be bigger than MAX_NF, not attempting memory "
              "allocation!\n",
              __func__);
      return FINUFFT_ERR_MAXNALLOC;
    }
    fwBatch.resize(nf * batchSize); // maybe big workspace

    CpBatch.resize(nj * batchSize); // batch c' work

    if (opts.debug)
      printf("[%s t3] widcen, batch %.2fGB alloc:\t%.3g s\n", __func__,
             (double)1E-09 * sizeof(std::complex<TF>) * (nf + nj) * batchSize,
             timer.elapsedsec());
    // printf("fwbatch, cpbatch ptrs: %llx %llx\n",fwBatch,CpBatch);

    // alloc rescaled NU src pts x'_j (in X etc), rescaled NU targ pts s'_k ...
    // We do this by resizing Xp, Yp, and Zp, and pointing X, Y, Z to their data;
    // this avoids any need for explicit cleanup.
    Xp.resize(nj);
    X = Xp.data();
    Sp.resize(nk);
    if (d > 1) {
      Yp.resize(nj);
      Y = Yp.data();
      Tp.resize(nk);
    }
    if (d > 2) {
      Zp.resize(nj);
      Z = Zp.data();
      Up.resize(nk);
    }

    // always shift as use gam to rescale x_j to x'_j, etc (twist iii)...
    TF ig1 = 1.0 / t3P.gam1, ig2 = 0.0, ig3 = 0.0; // "reciprocal-math" optim
    if (d > 1) ig2 = 1.0 / t3P.gam2;
    if (d > 2) ig3 = 1.0 / t3P.gam3;
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT j = 0; j < nj; ++j) {
      X[j] = (xj[j] - t3P.C1) * ig1;   // rescale x_j
      if (d > 1)                       // (ok to do inside loop because of branch predict)
        Y[j] = (yj[j] - t3P.C2) * ig2; // rescale y_j
      if (d > 2) Z[j] = (zj[j] - t3P.C3) * ig3; // rescale z_j
    }

    // set up prephase array...
    std::complex<TF> imasign =
        (fftSign >= 0) ? std::complex<TF>(0, 1) : std::complex<TF>(0, -1); // +-i
    prephase.resize(nj);
    if (t3P.D1 != 0.0 || t3P.D2 != 0.0 || t3P.D3 != 0.0) {
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
      for (BIGINT j = 0; j < nj; ++j) { // ... loop over src NU locs
        TF phase = t3P.D1 * xj[j];
        if (d > 1) phase += t3P.D2 * yj[j];
        if (d > 2) phase += t3P.D3 * zj[j];
        prephase[j] = std::cos(phase) + imasign * std::sin(phase); // Euler
                                                                   // e^{+-i.phase}
      }
    } else
      for (BIGINT j = 0; j < nj; ++j)
        prephase[j] = {1.0, 0.0}; // *** or keep flag so no mult in exec??

                                  // rescale the target s_k etc to s'_k etc...
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT k = 0; k < nk; ++k) {
      Sp[k] = t3P.h1 * t3P.gam1 * (s[k] - t3P.D1);   // so |s'_k| < pi/R
      if (d > 1)
        Tp[k] = t3P.h2 * t3P.gam2 * (t[k] - t3P.D2); // so |t'_k| <
                                                     // pi/R
      if (d > 2)
        Up[k] = t3P.h3 * t3P.gam3 * (u[k] - t3P.D3); // so |u'_k| <
                                                     // pi/R
    }
    // (old STEP 3a) Compute deconvolution post-factors array (per targ pt)...
    // (exploits that FT separates because kernel is prod of 1D funcs)
    deconv.resize(nk);
    std::vector<TF> phiHatk1(nk);                 // don't confuse w/ phiHat
    onedim_nuft_kernel(nk, Sp, phiHatk1, spopts); // fill phiHat1
    std::vector<TF> phiHatk2, phiHatk3;
    if (d > 1) {
      phiHatk2.resize(nk);
      onedim_nuft_kernel(nk, Tp, phiHatk2, spopts); // fill phiHat2
    }
    if (d > 2) {
      phiHatk3.resize(nk);
      onedim_nuft_kernel(nk, Up, phiHatk3, spopts); // fill phiHat3
    }
    // C can be nan or inf if M=0, no input NU pts
    int Cfinite = std::isfinite(t3P.C1) && std::isfinite(t3P.C2) && std::isfinite(t3P.C3);
    int Cnonzero = t3P.C1 != 0.0 || t3P.C2 != 0.0 || t3P.C3 != 0.0; // cen
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT k = 0; k < nk; ++k) { // .... loop over NU targ freqs
      TF phiHat = phiHatk1[k];
      if (d > 1) phiHat *= phiHatk2[k];
      if (d > 2) phiHat *= phiHatk3[k];
      deconv[k] = (std::complex<TF>)(1.0 / phiHat);
      if (Cfinite && Cnonzero) {
        TF phase = (s[k] - t3P.D1) * t3P.C1;
        if (d > 1) phase += (t[k] - t3P.D2) * t3P.C2;
        if (d > 2) phase += (u[k] - t3P.D3) * t3P.C3;
        deconv[k] *= std::cos(phase) + imasign * std::sin(phase); // Euler e^{+-i.phase}
      }
    }
    if (opts.debug)
      printf("[%s t3] phase & deconv factors:\t%.3g s\n", __func__, timer.elapsedsec());

    // Set up sort for spreading Cp (from primed NU src pts X, Y, Z) to fw...
    timer.restart();
    sortIndices.resize(nj);
    didSort = indexSort(sortIndices, nf1, nf2, nf3, nj, X, Y, Z, spopts);
    if (opts.debug)
      printf("[%s t3] sort (didSort=%d):\t\t%.3g s\n", __func__, didSort,
             timer.elapsedsec());

    // Plan and setpts once, for the (repeated) inner type 2 finufft call...
    timer.restart();
    BIGINT t2nmodes[]   = {nf1, nf2, nf3};             // t2 input is actually fw
    finufft_opts t2opts = opts;                        // deep copy, since not ptrs
    t2opts.modeord      = 0;                           // needed for correct t3!
    t2opts.debug        = std::max(0, opts.debug - 1); // don't print as much detail
    t2opts.spread_debug = std::max(0, opts.spread_debug - 1);
    t2opts.showwarn     = 0;                           // so don't see warnings 2x
    // (...could vary other t2opts here?)
    // MR: temporary hack, until we have figured out the C++ interface.
    FINUFFT_PLAN_T<TF> *tmpplan;
    int ier = finufft_makeplan_t<TF>(2, d, t2nmodes, fftSign, batchSize, tol, &tmpplan,
                                     &t2opts);
    innerT2plan.reset(tmpplan);
    if (ier > 1) { // if merely warning, still proceed
      fprintf(stderr, "[%s t3]: inner type 2 plan creation failed with ier=%d!\n",
              __func__, ier);
      return ier;
    }
    ier = innerT2plan->setpts(nk, Sp.data(), Tp.data(), Up.data(), 0, nullptr, nullptr,
                              nullptr); // note nk = # output points (not nj)
    if (ier > 1) {
      fprintf(stderr, "[%s t3]: inner type 2 setpts failed, ier=%d!\n", __func__, ier);
      return ier;
    }
    if (opts.debug)
      printf("[%s t3] inner t2 plan & setpts: \t%.3g s\n", __func__, timer.elapsedsec());
  }
  return 0;
}
template int FINUFFT_PLAN_T<float>::setpts(BIGINT nj, float *xj, float *yj, float *zj,
                                           BIGINT nk, float *s, float *t, float *u);
template int FINUFFT_PLAN_T<double>::setpts(BIGINT nj, double *xj, double *yj, double *zj,
                                            BIGINT nk, double *s, double *t, double *u);

// ............ end setpts ..................................................

// EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
template<typename TF>
int FINUFFT_PLAN_T<TF>::execute(std::complex<TF> *cj, std::complex<TF> *fk) {
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

  if (type != 3) { // --------------------- TYPE 1,2 EXEC ------------------

    double t_sprint = 0.0, t_fft = 0.0, t_deconv = 0.0; // accumulated timing
    if (opts.debug)
      printf("[%s] start ntrans=%d (%d batches, bsize=%d)...\n", __func__, ntrans, nbatch,
             batchSize);

    for (int b = 0; b * batchSize < ntrans; b++) { // .....loop b over batches

      // current batch is either batchSize, or possibly truncated if last one
      int thisBatchSize     = std::min(ntrans - b * batchSize, batchSize);
      int bB                = b * batchSize; // index of vector, since batchsizes same
      std::complex<TF> *cjb = cj + bB * nj;  // point to batch of weights
      std::complex<TF> *fkb = fk + bB * N;   // point to batch of mode coeffs
      if (opts.debug > 1)
        printf("[%s] start batch %d (size %d):\n", __func__, b, thisBatchSize);

      // STEP 1: (varies by type)
      timer.restart();
      if (type == 1) { // type 1: spread NU pts X, weights cj, to fw grid
        if (opts.spreadinterponly)
        {
          wrapArrayInVector(fkb, thisBatchSize*N, this->fwBatch);
        }
        spreadinterpSortedBatch<TF>(thisBatchSize, this, cjb);
        t_sprint += timer.elapsedsec();
        // Stop here if it is spread interp only.
        if (opts.spreadinterponly)
          continue;
      } else if(!opts.spreadinterponly) { //  type 2: amplify Fourier coeffs fk into 0-padded fw, but dont do it if it is spread interp only.
        deconvolveBatch<TF>(thisBatchSize, this, fkb);
        t_deconv += timer.elapsedsec();
      }
      if (!opts.spreadinterponly) // Do FFT only if its not spread interp only.
      {
        // STEP 2: call the FFT on this batch
        timer.restart();
        do_fft(this);
        t_fft += timer.elapsedsec();
        if (opts.debug > 1) printf("\tFFT exec:\t\t%.3g s\n", timer.elapsedsec());
      }
      else
        wrapArrayInVector(cjb, thisBatchSize*nj, this->fwBatch);
      // STEP 3: (varies by type)
      timer.restart();
      if (type == 1) { // type 1: deconvolve (amplify) fw and shuffle to fk
        deconvolveBatch<TF>(thisBatchSize, this, fkb);
        t_deconv += timer.elapsedsec();
      } else { // type 2: interpolate unif fw grid to NU target pts
        spreadinterpSortedBatch<TF>(thisBatchSize, this, cjb);
        t_sprint += timer.elapsedsec();
      }
      // Release the fwBatch vector to prevent double freeing of memory.
      if(opts.spreadinterponly)
        releaseVectorWrapper(this->fwBatch);
  
    } // ........end b loop

    if (opts.debug) { // report total times in their natural order...
      if (type == 1) {
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
    if (opts.debug)
      printf("[%s t3] start ntrans=%d (%d batches, bsize=%d)...\n", __func__, ntrans,
             nbatch, batchSize);

    for (int b = 0; b * batchSize < ntrans; b++) { // .....loop b over batches

      // batching and pointers to this batch, identical to t1,2 above...
      int thisBatchSize     = std::min(ntrans - b * batchSize, batchSize);
      int bB                = b * batchSize;
      std::complex<TF> *cjb = cj + bB * nj; // batch of input strengths
      std::complex<TF> *fkb = fk + bB * nk; // batch of output strengths
      if (opts.debug > 1)
        printf("[%s t3] start batch %d (size %d):\n", __func__, b, thisBatchSize);

      // STEP 0: pre-phase (possibly) the c_j input strengths into c'_j batch...
      timer.restart();
#pragma omp parallel for num_threads(opts.nthreads) // or batchSize?
      for (int i = 0; i < thisBatchSize; i++) {
        BIGINT ioff = i * nj;
        for (BIGINT j = 0; j < nj; ++j) {
          CpBatch[ioff + j] = prephase[j] * cjb[ioff + j];
        }
      }
      t_pre += timer.elapsedsec();

      // STEP 1: spread c'_j batch (x'_j NU pts) into fw batch grid...
      timer.restart();
      spopts.spread_direction = 1;                                      // spread
      spreadinterpSortedBatch<TF>(thisBatchSize, this, CpBatch.data()); // X are primed
      t_spr += timer.elapsedsec();

      // STEP 2: type 2 NUFFT from fw batch to user output fk array batch...
      timer.restart();
      // illegal possible shrink of ntrans *after* plan for smaller last batch:
      innerT2plan->ntrans = thisBatchSize; // do not try this at home!
      /* (alarming that FFT not shrunk, but safe, because t2's fwBatch array
     still the same size, as Andrea explained; just wastes a few flops) */
      innerT2plan->execute(fkb, fwBatch.data());
      t_t2 += timer.elapsedsec();
      // STEP 3: apply deconvolve (precomputed 1/phiHat(targ_k), phasing too)...
      timer.restart();
#pragma omp parallel for num_threads(opts.nthreads)
      for (int i = 0; i < thisBatchSize; i++) {
        BIGINT ioff = i * nk;
        for (BIGINT k = 0; k < nk; ++k) fkb[ioff + k] *= deconv[k];
      }
      t_deconv += timer.elapsedsec();
    } // ........end b loop

    if (opts.debug) { // report total times in their natural order...
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
template int FINUFFT_PLAN_T<float>::execute(std::complex<float> *cj,
                                            std::complex<float> *fk);
template int FINUFFT_PLAN_T<double>::execute(std::complex<double> *cj,
                                             std::complex<double> *fk);

// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
template<typename TF> FINUFFT_PLAN_T<TF>::~FINUFFT_PLAN_T() {
  // Destructor for plan object. All deallocations are simply automatic now.
}
template FINUFFT_PLAN_T<float>::~FINUFFT_PLAN_T();
template FINUFFT_PLAN_T<double>::~FINUFFT_PLAN_T();
