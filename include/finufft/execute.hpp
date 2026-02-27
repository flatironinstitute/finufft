#pragma once

#include <algorithm>
#include <complex>
#include <cstdio>
#include <vector>

#include <finufft/plan.hpp>
#include <finufft/utils.hpp>
#include <finufft/spreadinterp.hpp>
#include <finufft/simd.hpp>

/* Computational core for FINUFFT.

   Based on Barnett 2017-2018 finufft?d.cpp containing nine drivers, plus
   2d1/2d2 many-vector drivers by Melody Shih, summer 2018.
   Original guru interface written by Andrea Malleo, summer 2019, mentored
   by Alex Barnett. Many rewrites in early 2020 by Alex Barnett & Libin Lu.

   As of v1.2 these replace the old hand-coded separate 9 finufft?d() functions
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

// ---------- deconvolveshuffle: private methods on FINUFFT_PLAN_T ----------

template<typename T>
void FINUFFT_PLAN_T<T>::deconvolveshuffle1d(int dir, T prefac, T *fk,
                                            std::complex<T> *fw) const
/*
  if dir==1: copies fw to fk with amplification by prefac/ker
  if dir==2: copies fk to fw (and zero pads rest of it), same amplification.

  modeord=0: use CMCL-compatible mode ordering in fk (from -N/2 up to N/2-1)
      1: use FFT-style (from 0 to N/2-1, then -N/2 up to -1).

  fk is a size-ms T complex array (2*ms Ts alternating re,im parts)
  fw is a size-nf1 complex array (2*nf1 Ts alternating re,im parts)
  ker is real-valued T array of length nf1/2+1 (accessed via phiHat[0]).

  Single thread only, but shouldn't matter since mostly data movement.

  It has been tested that the repeated floating division in this inner loop
  only contributes at the <3% level in 3D relative to the FFT cost (8 threads).
  This could be removed by passing in an inverse kernel and doing mults.

  todo: rewrite w/ C++-complex I/O, check complex divide not slower than
    real divide, or is there a way to force a real divide?

  Barnett 1/25/17. Fixed ms=0 case 3/14/17. modeord flag & clean 10/25/17
  Previous args (ker, nf1, ms, modeord) are now read from plan members
  (phiHat[0], nfdim[0], mstu[0], opts.modeord).
  Converted to class member, ms param removed. Barbone 2/26/26.
*/
{
  const BIGINT ms   = mstu[0];
  const auto &ker   = phiHat[0];
  const BIGINT nf1  = nfdim[0];
  const int modeord = opts.modeord;
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
void FINUFFT_PLAN_T<T>::deconvolveshuffle2d(int dir, T prefac, T *fk,
                                            std::complex<T> *fw) const
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
     respectively (accessed via phiHat[0], phiHat[1]).

  Barnett 2/1/17, Fixed mt=0 case 3/14/17. modeord 10/25/17
  Previous args (ker2, nf1, nf2, ms, mt, modeord) are now read from plan
  members (phiHat[1], nfdim[0..1], mstu[0..1], opts.modeord). ker1 is read
  from phiHat[0] inside deconvolveshuffle1d.
  Converted to class member, ms/mt params removed. Barbone 2/26/26.
*/
{
  const BIGINT ms   = mstu[0];
  const BIGINT mt   = mstu[1];
  const auto &ker2  = phiHat[1];
  const BIGINT nf1  = nfdim[0];
  const BIGINT nf2  = nfdim[1];
  const int modeord = opts.modeord;
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
    deconvolveshuffle1d(dir, prefac / ker2[k2], fk + pp, &fw[nf1 * k2]);
  for (BIGINT k2 = k2min; k2 < 0; ++k2, pn += 2 * ms) // neg y-freqs
    deconvolveshuffle1d(dir, prefac / ker2[-k2], fk + pn, &fw[nf1 * (nf2 + k2)]);
}

template<typename T>
void FINUFFT_PLAN_T<T>::deconvolveshuffle3d(int dir, T prefac, T *fk,
                                            std::complex<T> *fw) const
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
     and nf3/2+1 respectively (accessed via phiHat[0], phiHat[1], phiHat[2]).

  Barnett 2/1/17, Fixed mu=0 case 3/14/17. modeord 10/25/17
  Previous args (ker3, nf1, nf2, nf3, ms, mt, mu, modeord) are now read from
  plan members (phiHat[2], nfdim[0..2], mstu[0..2], opts.modeord). ker1/ker2
  are read from phiHat[0..1] inside deconvolveshuffle1d/2d.
  Converted to class member, ms/mt/mu params removed. Barbone 2/26/26.
*/
{
  const BIGINT ms   = mstu[0];
  const BIGINT mt   = mstu[1];
  const BIGINT mu   = mstu[2];
  const auto &ker3  = phiHat[2];
  const BIGINT nf1  = nfdim[0];
  const BIGINT nf2  = nfdim[1];
  const BIGINT nf3  = nfdim[2];
  const int modeord = opts.modeord;
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
    deconvolveshuffle2d(dir, prefac / ker3[k3], fk + pp, &fw[np * k3]);
  for (BIGINT k3 = k3min; k3 < 0; ++k3, pn += 2 * ms * mt) // neg z-freqs
    deconvolveshuffle2d(dir, prefac / ker3[-k3], fk + pn, &fw[np * (nf3 + k3)]);
}

// --------- batch helper functions for t1,2 exec: ---------------------------

template<typename T>
int FINUFFT_PLAN_T<T>::spreadinterpSortedBatch(
    int batchSize, std::complex<T> *fwBatch, std::complex<T> *cBatch, bool adjoint) const
/*
  Spreads (or interpolates) a batch of batchSize strength vectors in cBatch
  to (or from) the batch of fine working grids fwBatch, using the same set of
  (index-sorted) NU points p.X,Y,Z for each vector in the batch.
  The direction (spread vs interpolate) is set by p.spopts.spread_direction.
  Returns 0 (no error reporting for now).
  Notes:
  1) cBatch (c_j I/O) is already assumed to have the correct offset, ie here we
   read from the start of cBatch (unlike Malleo). fwBatch also has zero offset.
  2) this routine is a batched version of spreadinterpSorted in spreadinterp.cpp
  Barnett 5/19/20, based on Malleo 2019.
  ChaithyaGR 1/7/25: new arg fwBatch (won't be p.fwBatch if spreadinterponly=1)
*/
{
  // opts.spread_thread: 1 sequential multithread, 2 parallel single-thread.
  // omp_sets_nested deprecated, so don't use; assume not nested for 2 to work.
  // But when nthr_outer=1 here, omp par inside the loop sees all threads...
#ifdef _OPENMP
  int nthr_outer = opts.spread_thread == 1 ? 1 : batchSize;
#endif
#pragma omp parallel for num_threads(nthr_outer)
  for (int i = 0; i < batchSize; i++) {
    std::complex<T> *fwi = fwBatch + i * nf(); // start of i'th fw array in
                                               // fwBatch workspace or user array
    std::complex<T> *ci = cBatch + i * nj;     // start of i'th c array in cBatch
    spreadinterpSorted((T *)fwi, (T *)ci, adjoint);
  }
  return 0;
}

template<typename T>
int FINUFFT_PLAN_T<T>::deconvolveBatch(int batchSize, std::complex<T> *fkBatch,
                                       std::complex<T> *fwBatch, bool adjoint) const
/*
  Type 1: deconvolves (amplifies) from each interior fw array in fwBatch
  into each output array fk in fkBatch.
  Type 2: deconvolves from user-supplied input fk to 0-padded interior fw,
  again looping over fk in fkBatch and fw in fwBatch.
  The direction (spread vs interpolate) is set by spopts.spread_direction
  and the adjoint parameter.
  This is mostly a loop calling deconvolveshuffle?d for the needed dim, batchSize
  times.
  Barnett 5/21/20, simplified from Malleo 2019 (eg t3 logic won't be in here)
*/
{
  // since deconvolveshuffle?d are single-thread, omp par seems to help here...
  int dir = spopts.spread_direction;
  // Quick and dirty way to change direction 1 into direction 2 and vice versa
  // if adjoint operation is requested.
  if (adjoint) dir = 3 - dir;
#pragma omp parallel for num_threads(batchSize)
  for (int i = 0; i < batchSize; i++) {
    std::complex<T> *fwi = fwBatch + i * nf(); // start of i'th fw array in
                                               // wkspace
    std::complex<T> *fki = fkBatch + i * N();  // start of i'th fk array in fkBatch

    // pick dim-specific routine; note prefactors hardcoded to 1.0...
    if (dim == 1)
      deconvolveshuffle1d(dir, T(1), (T *)fki, fwi);
    else if (dim == 2)
      deconvolveshuffle2d(dir, T(1), (T *)fki, fwi);
    else
      deconvolveshuffle3d(dir, T(1), (T *)fki, fwi);
  }
  return 0;
}

// --------------- execute user guru interface driver ----------

template<typename TF>
int FINUFFT_PLAN_T<TF>::execute_internal(TC *cj, TC *fk, bool adjoint, int ntrans_actual,
                                         TC *aligned_scratch, size_t scratch_size) const {
  /* See ../docs/cguru.doc for current documentation.

   For given (stack of) weights cj or coefficients fk, performs NUFFTs with
   existing (sorted) NU pts and existing plan.
   For adjoint == false:
     For type 1 and 3: cj is input, fk is output.
     For type 2: fk is input, cj is output.
   For adjoint == true:
     For type 1 and 3: fk is input, cj is output.
     For type 2: cj is input, fk is output.
   Performs spread/interp, pre/post deconvolve, and FFT as appropriate
   for each of the 3 types.
   For cases of ntrans>1, performs work in blocks of size up to batchSize.
   Return value 0 (no error diagnosis yet).
   Barnett 5/20/20, based on Malleo 2019.

   Additional parameters introducd by MR, 02/2025

   adjoint: if false, behave as before; if true, compute the adjoint of the
     planned transform, i.e.:
     - if type is 1, perform an analogous type 2 with opposite isign
     - if type is 2, perform an analogous type 1 with opposite isign
     - if type is 3, perform the planned transform "backwards" and with opposite isign

   ntrans_actual:
     Helper variable for specifying the number of transforms to execute in one
     go when calling the "inner" NUFFT during type 3 plan execution
     - <= 0: behave as before
     - 0 < ntrans_actual <= batchSize: instead of doing ntrans transforms,
         perform only ntrans_actual

   scratch_size, aligned_scratch:
     Helpers to avoid repeated allocation/deallocation of scratch space in the
     "inner" NUFFT during type 3 plan execution
     If scratch_size>0. then aligned_scratch points to FFTW-aligned storage with
     at least scratch_size entries. This can be used as scratch space.
*/
  using finufft::utils::CNTime;
  CNTime timer;
  timer.start();

  // if no number of actual transforms has been specified, use the default
  if (ntrans_actual <= 0) ntrans_actual = ntrans;

  if (type != 3) { // --------------------- TYPE 1,2 EXEC ------------------

    double t_sprint = 0.0, t_fft = 0.0, t_deconv = 0.0; // accumulated timing
    if (opts.debug)
      printf("[%s] start%s ntrans=%d (%d batches, bsize=%d)...\n", "execute",
             adjoint ? " adjoint" : "", ntrans_actual, nbatch, batchSize);
    // allocate temporary buffers
    bool scratch_provided = scratch_size >= size_t(nf() * batchSize);
    std::vector<TC, xsimd::aligned_allocator<TC, 64>> fwBatch_(
        scratch_provided ? 0 : nf() * batchSize);
    TC *fwBatch = scratch_provided ? aligned_scratch : fwBatch_.data();
    for (int b = 0; b * batchSize < ntrans_actual; b++) { // .....loop b over batches

      // current batch is either batchSize, or possibly truncated if last one
      int thisBatchSize = std::min(ntrans_actual - b * batchSize, batchSize);
      int bB            = b * batchSize; // index of vector, since batchsizes same
      TC *cjb           = cj + bB * nj;  // point to batch of user weights
      TC *fkb           = fk + bB * N(); // point to batch of user mode coeffs
      if (opts.debug > 1)
        printf("[%s] start batch %d (size %d):\n", "execute", b, thisBatchSize);

      // STEP 1: (varies by type)
      timer.restart();
      // usually spread/interp to/from fwBatch (vs spreadinterponly: to/from user grid)
      TC *fwBatch_or_fkb = opts.spreadinterponly ? fkb : fwBatch;
      if ((type == 1) != adjoint) { // spread NU pts X, weights cj, to fw grid
        spreadinterpSortedBatch(thisBatchSize, fwBatch_or_fkb, cjb, adjoint);
        t_sprint += timer.elapsedsec();
        if (opts.spreadinterponly) // we're done (skip to next iteration of loop)
          continue;
      } else if (!opts.spreadinterponly) {
        // amplify Fourier coeffs fk into 0-padded fw
        deconvolveBatch(thisBatchSize, fkb, fwBatch, adjoint);
        t_deconv += timer.elapsedsec();
      }
      if (!opts.spreadinterponly) { // Do FFT unless spread/interp only...
        // STEP 2: call the FFT on this batch
        timer.restart();

        do_fft(fwBatch, thisBatchSize, adjoint);
        t_fft += timer.elapsedsec();
        if (opts.debug > 1) printf("\tFFT exec:\t\t%.3g s\n", timer.elapsedsec());
      }
      // STEP 3: (varies by type)
      timer.restart();
      if ((type == 1) != adjoint) { // deconvolve (amplify) fw and shuffle to fk
        deconvolveBatch(thisBatchSize, fkb, fwBatch, adjoint);
        t_deconv += timer.elapsedsec();
      } else { // interpolate unif fw grid to NU target pts
        spreadinterpSortedBatch(thisBatchSize, fwBatch_or_fkb, cjb, adjoint);
        t_sprint += timer.elapsedsec();
      }
    } // ........end b loop

    if (opts.debug) { // report total times in their natural order...
      if ((type == 1) != adjoint) {
        printf("[%s] done. tot spread:\t\t%.3g s\n", "execute", t_sprint);
        printf("                tot FFT:\t\t%.3g s\n", t_fft);
        printf("                tot deconvolve:\t\t%.3g s\n", t_deconv);
      } else {
        printf("[%s] done. tot deconvolve:\t\t%.3g s\n", "execute", t_deconv);
        printf("                tot FFT:\t\t%.3g s\n", t_fft);
        printf("                tot interp:\t\t%.3g s\n", t_sprint);
      }
    }
  }

  else { // ----------------------------- TYPE 3 EXEC ---------------------

    // for (BIGINT j=0;j<10;++j) printf("\tcj[%ld]=%.15g+%.15gi\n",(long
    // int)j,(double)real(cj[j]),(double)imag(cj[j]));  // debug

    double t_phase = 0.0, t_sprint = 0.0, t_inner = 0.0,
           t_deconv = 0.0; // accumulated timings
    if (opts.debug)
      printf("[%s t3] start%s ntrans=%d (%d batches, bsize=%d)...\n", "execute",
             adjoint ? " adjoint" : "", ntrans_actual, nbatch, batchSize);

    // allocate temporary buffers
    // We are trying to be clever here and re-use memory whenever possible.
    // Also, we allocate the memory for the "inner" NUFFT here as well,
    // so that it doesn't need to be reallocated for every batch.
    std::vector<TC, xsimd::aligned_allocator<TC, 64>> buf1, buf2, buf3;
    TC *CpBatch, *fwBatch, *fwBatch_inner;
    if (!adjoint) { // we can combine CpBatch and fwBatch_inner!
      buf1.resize(std::max(nj * batchSize, innerT2plan->nf() * innerT2plan->batchSize));
      CpBatch = fwBatch_inner = buf1.data();
      buf2.resize(nf() * batchSize);
      fwBatch = buf2.data();
    } else { // we may be able to combine CpBatch and fwBatch!
      // This only works if the inner plan performs our calls (that we do once
      // per batch) without doing any of its own batching ...
      if (innerT2plan->batchSize >= batchSize) {
        buf1.resize(std::max(nk * batchSize, nf() * batchSize));
        CpBatch = fwBatch = buf1.data();
        buf2.resize(innerT2plan->nf() * innerT2plan->batchSize);
        fwBatch_inner = buf2.data();
      } else {
        buf1.resize(nk * batchSize);
        CpBatch = buf1.data();
        buf2.resize(nf() * batchSize);
        fwBatch = buf2.data();
        buf3.resize(innerT2plan->nf() * innerT2plan->batchSize);
        fwBatch_inner = buf3.data();
      }
    }

    for (int b = 0; b * batchSize < ntrans_actual; b++) { // .....loop b over batches

      // batching and pointers to this batch, identical to t1,2 above...
      int thisBatchSize = std::min(ntrans_actual - b * batchSize, batchSize);
      int bB            = b * batchSize;
      TC *cjb           = cj + bB * nj; // batch of input strengths
      TC *fkb           = fk + bB * nk; // batch of output strengths
      if (opts.debug > 1)
        printf("[%s t3] start batch %d (size %d):\n", "execute", b, thisBatchSize);

      if (!adjoint) {
        // STEP 0: pre-phase (possibly) the c_j input strengths into c'_j batch...
        timer.restart();
#pragma omp parallel for num_threads(opts.nthreads) // or batchSize?
        for (BIGINT j = 0; j < nj; ++j) {
          auto phase = prephase[j];
          for (int i = 0; i < thisBatchSize; i++)
            CpBatch[i * nj + j] = phase * cjb[i * nj + j];
        }
        t_phase += timer.elapsedsec();

        // STEP 1: spread c'_j batch (x'_j NU pts) into internal fw batch grid...
        timer.restart();
        spreadinterpSortedBatch(thisBatchSize, fwBatch, CpBatch,
                                adjoint); // X are primed
        t_sprint += timer.elapsedsec();

        // STEP 2: type 2 NUFFT from fw batch to user output fk array batch...
        timer.restart();
        /* (alarming that FFT not shrunk, but safe, because t2's fwBatch array
       still the same size, as Andrea explained; just wastes a few flops) */
        innerT2plan->execute_internal(fkb, fwBatch, adjoint, thisBatchSize, fwBatch_inner,
                                      innerT2plan->nf() * innerT2plan->batchSize);
        t_inner += timer.elapsedsec();
        // STEP 3: apply deconvolve (precomputed 1/phiHat(targ_k), phasing too)...
        timer.restart();
#pragma omp parallel for num_threads(opts.nthreads)
        for (BIGINT k = 0; k < nk; ++k)
          for (int i = 0; i < thisBatchSize; i++) fkb[i * nk + k] *= deconv[k];
        t_deconv += timer.elapsedsec();
      } else { // adjoint mode
        // STEP 0: apply deconvolve (precomputed 1/phiHat(targ_k), conjugate phasing
        // too)... write output into CpBatch
        timer.restart();
#pragma omp parallel for num_threads(opts.nthreads)
        for (BIGINT k = 0; k < nk; ++k)
          for (int i = 0; i < thisBatchSize; i++)
            CpBatch[i * nk + k] = fkb[i * nk + k] * conj(deconv[k]);
        t_deconv += timer.elapsedsec();
        // STEP 1: adjoint type 2 (i.e. type 1) NUFFT from CpBatch to fwBatch...
        timer.restart();
        innerT2plan->execute_internal(CpBatch, fwBatch, adjoint, thisBatchSize,
                                      fwBatch_inner,
                                      innerT2plan->nf() * innerT2plan->batchSize);
        t_inner += timer.elapsedsec();
        // STEP 2: interpolate fwBatch into user output array ...
        timer.restart();
        spreadinterpSortedBatch(thisBatchSize, fwBatch, cjb,
                                adjoint); // X are primed
        t_sprint += timer.elapsedsec();
        // STEP 3: post-phase (possibly) the c_j output strengths (in place) ...
        timer.restart();
#pragma omp parallel for num_threads(opts.nthreads) // or batchSize?
        for (BIGINT j = 0; j < nj; ++j) {
          auto phase = conj(prephase[j]);
          for (int i = 0; i < thisBatchSize; i++) cjb[i * nj + j] *= phase;
        }
        t_phase += timer.elapsedsec();
      }
    } // ........end b loop

    if (opts.debug) { // report total times in their natural order...
      if (!adjoint) {
        printf("[%s t3] done. tot prephase:\t%.3g s\n", "execute", t_phase);
        printf("                   tot spread:\t\t%.3g s\n", t_sprint);
        printf("                   tot inner NUFFT:\t%.3g s\n", t_inner);
        printf("                   tot deconvolve:\t%.3g s\n", t_deconv);
      } else {
        printf("[%s t3] done. tot deconvolve:\t%.3g s\n", "execute", t_deconv);
        printf("                   tot inner NUFFT:\t%.3g s\n", t_inner);
        printf("                   tot interp:\t\t%.3g s\n", t_sprint);
        printf("                   tot postphase:\t%.3g s\n", t_phase);
      }
    }
  }
  // for (BIGINT k=0;k<10;++k) printf("\tfk[%ld]=%.15g+%.15gi\n",(long
  // int)k,(double)real(fk[k]),(double)imag(fk[k]));  // debug

  return 0;
}
