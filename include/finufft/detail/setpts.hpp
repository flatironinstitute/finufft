#pragma once

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <finufft/detail/spreadinterp.hpp>
#include <finufft/finufft_core.hpp>
#include <finufft/finufft_utils.hpp>
#include <finufft/heuristics.hpp>
#include <finufft_common/kernel.h>

// ---------- local math routines for type-3 setpts: --------

template<typename TF>
void FINUFFT_PLAN_T<TF>::set_nhg_type3(int idim, TF S, TF X)
/* sets nfdim[idim], t3P.h[idim], and t3P.gam[idim], for type 3 only.
   Inputs:
   idim - which dimension (0,1,2)
   X and S are the xj and sk interval half-widths respectively.
   Reads opts and spopts from the plan.
   Outputs written to plan members:
   nfdim[idim] - size of upsampled grid for this dimension.
   t3P.h[idim] - grid spacing = 2pi/nf
   t3P.gam[idim] - x rescale factor, ie x'_j = x_j/gam (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
   2/24/26 Barbone: converted from free function to method on FINUFFT_PLAN_T.
   Previous args (opts, spopts) are now plan members; previous output pointers
   (nf, h, gam) are now written directly to plan members nfdim, t3P.h, t3P.gam.
*/
{
  using namespace finufft::common;
  using namespace finufft::utils;
  int nss = spopts.nspread + 1; // since ns may be odd
  TF Xsafe = X, Ssafe = S;       // may be tweaked locally
  if (X == 0.0)                 // logic ensures XS>=1, handle X=0 a/o S=0
    if (S == 0.0) {
      Xsafe = 1.0;
      Ssafe = 1.0;
    } else
      Xsafe = std::max(Xsafe, 1 / S);
  else
    Ssafe = std::max(Ssafe, 1 / X);
  // use the safe X and S...
  auto nfd = TF(2.0 * opts.upsampfac * Ssafe * Xsafe / PI + nss);
  if (!std::isfinite(nfd)) nfd = 0.0;
  nfdim[idim] = (BIGINT)nfd;
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (nfdim[idim] < 2 * spopts.nspread) nfdim[idim] = 2 * spopts.nspread;
  if (nfdim[idim] < MAX_NF)                   // otherwise will fail
    nfdim[idim] = next235even(nfdim[idim]);   // expensive at huge nf
  t3P.h[idim]   = TF(2.0 * PI / nfdim[idim]); // upsampled grid spacing
  t3P.gam[idim] = TF(nfdim[idim] / (2.0 * opts.upsampfac * Ssafe)); // x scale fac to x'
}

// --------- setpts user guru interface driver ----------

template<typename TF>
int FINUFFT_PLAN_T<TF>::setpts(BIGINT nj, const TF *xj, const TF *yj, const TF *zj,
                               BIGINT nk, const TF *s, const TF *t, const TF *u) {
  using namespace finufft::utils;
  using namespace finufft::heuristics;
  using namespace finufft::spreadinterp;
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
    // If upsampfac is not locked by user (auto mode), choose or update it now
    // based on the actual density nj/N(). Re-plan if density changed significantly.
    if (!upsamp_locked) {
      double density   = double(nj) / double(N());
      double upsampfac = bestUpsamplingFactor<TF>(opts.nthreads, density, dim, type, tol);
      // Re-plan if this is the first call (upsampfac==0) or if upsampfac changed
      if (upsampfac != opts.upsampfac) {
        opts.upsampfac = upsampfac;
        if (opts.debug)
          printf("[setpts] selected best upsampfac=%.3g (density=%.3g, nj=%lld)\n",
                 opts.upsampfac, density, (long long)nj);
        int code = setup_spreadinterp();
        if (code > 1) return code;
        precompute_horner_coeffs();
        // Perform the planning steps (first call or re-plan due to density change).
        code = init_grid_kerFT_FFT();
        if (code) return code;
      }
    }

    XYZ     = {xj, yj, zj}; // plan must keep pointers to user's fixed NU pts
    int ier = spreadcheck(nfdim[0], nfdim[1], nfdim[2], spopts);
    if (ier)                // no warnings allowed here
      return ier;
    timer.restart();
    sortIndices.resize(nj);
    indexSort();
    if (opts.debug)
      printf("[%s] sort (didSort=%d):\t\t%.3g s\n", __func__, (int)didSort,
             timer.elapsedsec());

  } else { // ------------------------- TYPE 3 SETPTS -----------------------
           // (here we can precompute pre/post-phase factors and plan the t2)

    std::array<const TF *, 3> XYZ_in{xj, yj, zj};
    std::array<const TF *, 3> STU_in{s, t, u};
    if (nk < 0) {
      fprintf(stderr, "[%s] nk (%lld) cannot be negative!\n", __func__, (long long)nk);
      return FINUFFT_ERR_NUM_NU_PTS_INVALID;
    } else if (nk > MAX_NU_PTS) {
      fprintf(stderr, "[%s] nk (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nk);
      return FINUFFT_ERR_NUM_NU_PTS_INVALID;
    }
    this->nk = nk; // user set # targ freq pts
    STU      = {s, t, u};

    // For type 3 with deferred upsampfac (not locked by user), pick and persist
    // an upsamp now using density=1.0 so that subsequent steps (set_nhg_type3 etc.)
    // have a concrete upsampfac to work with. This choice is persisted so inner
    // type-2 plans will inherit it.
    double upsampfac = bestUpsamplingFactor<TF>(opts.nthreads, 1.0, dim, type, tol);
    if (!upsamp_locked && (upsampfac != opts.upsampfac)) {
      opts.upsampfac = upsampfac;
      if (opts.debug > 1)
        printf("[setpts t3] selected upsampfac=%.2f (density=1 used; persisted)\n",
               opts.upsampfac);
      int sier = setup_spreadinterp();
      if (sier > 1) return sier;
      precompute_horner_coeffs();
    }

    // pick x, s intervals & shifts & # fine grid pts (nf) in each dim...
    std::array<TF, 3> S = {0, 0, 0};
    if (opts.debug) printf("\tM=%lld N=%lld\n", (long long)nj, (long long)nk);
    for (int idim = 0; idim < dim; ++idim) {
      arraywidcen(nj, XYZ_in[idim], &(t3P.X[idim]), &(t3P.C[idim]));
      arraywidcen(nk, STU_in[idim], &S[idim], &(t3P.D[idim])); // same D, S, but for {s_k}
      set_nhg_type3(idim, S[idim], t3P.X[idim]);               // applies twist i)
      if (opts.debug) // report on choices of shifts, centers, etc...
        printf("\tX%d=%.3g C%d=%.3g S%d=%.3g D%d=%.3g gam%d=%g nf%d=%lld h%d=%.3g\t\n",
               idim, t3P.X[idim], idim, t3P.C[idim], idim, S[idim], idim, t3P.D[idim],
               idim, t3P.gam[idim], idim, (long long)nfdim[idim], idim, t3P.h[idim]);
    }
    for (int idim = dim; idim < 3; ++idim)
      t3P.C[idim] = t3P.D[idim] = 0.0; // their defaults if dim 2 unused, etc

    if (nf() * batchSize > MAX_NF) {
      fprintf(stderr,
              "[%s t3] fwBatch would be bigger than MAX_NF, not attempting memory "
              "allocation!\n",
              __func__);
      return FINUFFT_ERR_MAXNALLOC;
    }

    // alloc rescaled NU src pts x'_j (in X etc), rescaled NU targ pts s'_k ...
    // We do this by resizing Xp, Yp, and Zp, and pointing X, Y, Z to their data;
    // this avoids any need for explicit cleanup.
    for (int idim = 0; idim < dim; ++idim) {
      XYZp[idim].resize(nj);
      XYZ[idim] = XYZp[idim].data();
      STUp[idim].resize(nk);
    }

    // always shift as use gam to rescale x_j to x'_j, etc (twist iii)...
    std::array<TF, 3> ig = {0, 0, 0};
    for (int idim = 0; idim < dim; ++idim) ig[idim] = 1.0 / t3P.gam[idim];
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT j = 0; j < nj; ++j) {
      for (int idim = 0; idim < dim; ++idim)
        XYZp[idim][j] = (XYZ_in[idim][j] - t3P.C[idim]) * ig[idim]; // rescale x_j
    }

    // set up prephase array...
    TF isign = (fftSign >= 0) ? 1 : -1;
    prephase.resize(nj);
    if (t3P.D[0] != 0.0 || t3P.D[1] != 0.0 || t3P.D[2] != 0.0) {
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
      for (BIGINT j = 0; j < nj; ++j) { // ... loop over src NU locs
        TF phase = 0;
        for (int idim = 0; idim < dim; ++idim) phase += t3P.D[idim] * XYZ_in[idim][j];
        prephase[j] = std::polar(TF(1), isign * phase); // Euler
      }
    } else
      for (BIGINT j = 0; j < nj; ++j)
        prephase[j] = {1.0, 0.0}; // *** or keep flag so no mult in exec??

    // create a 1D phihat evaluator
    Kernel_onedim_FT onedim_phihat(*this);

    // (old STEP 3a) Compute deconvolution post-factors array (per targ pt)...
    // (exploits that FT separates because kernel is prod of 1D funcs)
    deconv.resize(nk);
    // C can be nan or inf if M=0, no input NU pts
    bool Cfinite =
        std::isfinite(t3P.C[0]) && std::isfinite(t3P.C[1]) && std::isfinite(t3P.C[2]);
    bool Cnonzero = t3P.C[0] != 0.0 || t3P.C[1] != 0.0 || t3P.C[2] != 0.0; // cen
    bool do_phase = Cfinite && Cnonzero;
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT k = 0; k < nk; ++k) { // .... loop over NU targ freqs
      TF phiHat = 1;
      TF phase  = 0;
      for (int idim = 0; idim < dim; ++idim) {
        auto tSTUin = STU_in[idim][k];
        // rescale the target s_k etc to s'_k etc...
        auto tSTUp = t3P.h[idim] * t3P.gam[idim] * (tSTUin - t3P.D[idim]); // so |s'_k| <
                                                                           // pi/R
        phiHat *= onedim_phihat(tSTUp);
        if (do_phase) phase += (tSTUin - t3P.D[idim]) * t3P.C[idim];
        STUp[idim][k] = tSTUp;
      }
      deconv[k] = do_phase ? std::polar(TF(1) / phiHat, isign * phase) : TF(1) / phiHat;
    }
    if (opts.debug)
      printf("[%s t3] phase & deconv factors:\t%.3g s\n", __func__, timer.elapsedsec());

    // Set up sort for spreading Cp (from primed NU src pts X, Y, Z) to fw...
    timer.restart();
    sortIndices.resize(nj);
    spopts.spread_direction = 1;
    indexSort();
    if (opts.debug)
      printf("[%s t3] sort (didSort=%d):\t\t%.3g s\n", __func__, (int)didSort,
             timer.elapsedsec());

    // Plan and setpts once, for the (repeated) inner type 2 finufft call...
    timer.restart();
    BIGINT t2nmodes[]   = {nfdim[0], nfdim[1], nfdim[2]}; // t2's input actually fw
    finufft_opts t2opts = opts;                           // deep copy, since not ptrs
    t2opts.modeord      = 0;                              // needed for correct t3!
    t2opts.debug        = std::max(0, opts.debug - 1);    // don't print as much detail
    t2opts.spread_debug = std::max(0, opts.spread_debug - 1);
    t2opts.showwarn     = 0;                              // so don't see warnings 2x
    if (!upsamp_locked)
      t2opts.upsampfac = 0.0; // if the upsampfac was auto, let inner
                              // t2 pick it again (from density=nj/Nf)
    // (...could vary other t2opts here?)
    // MR: temporary hack, until we have figured out the C++ interface.
    FINUFFT_PLAN_T<TF> *tmpplan;
    int ier = finufft_makeplan_t<TF>(2, d, t2nmodes, fftSign, batchSize, tol, &tmpplan,
                                     &t2opts);
    if (ier > 1) { // if merely warning, still proceed
      fprintf(stderr, "[%s t3]: inner type 2 plan creation failed with ier=%d!\n",
              __func__, ier);
      return ier;
    }
    ier = tmpplan->setpts(nk, STUp[0].data(), STUp[1].data(), STUp[2].data(), 0, nullptr,
                          nullptr,
                          nullptr); // note nk = # output points (not nj)
    innerT2plan.reset(tmpplan);
    if (ier > 1) {
      fprintf(stderr, "[%s t3]: inner type 2 setpts failed, ier=%d!\n", __func__, ier);
      return ier;
    }
    if (opts.debug)
      printf("[%s t3] inner t2 plan & setpts: \t%.3g s\n", __func__, timer.elapsedsec());
  }
  return 0;
}
