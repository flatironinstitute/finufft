#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include <cassert>

#include <finufft/spreadinterp.hpp>
#include <finufft/plan.hpp>
#include <finufft/utils.hpp>
#include <finufft/heuristics.hpp>
#include <finufft_common/kernel.h>

// ---------- local helpers for setpts: --------

template<typename TF> void FINUFFT_PLAN_T<TF>::check_sigma() {
  constexpr double eps_mach = std::numeric_limits<TF>::epsilon();
  const double gridlen      = *std::max_element(m.nfdim.begin(), m.nfdim.begin() + dim);
  const double sigma_min =
      finufft::common::lowest_sigma((double)m.tol, dim, m.spopts.nspread, eps_mach,
                                    gridlen);
  const double eps_round  = 0.48 * eps_mach * gridlen;
  const bool unachievable = (double)m.tol <= eps_round; // MAXSIGMA still not enough
  if (!unachievable && sigma_min <= m.spopts.upsampfac) return; // fine
  const double suggest = std::min(sigma_min, finufft::common::MAXSIGMA);
  const bool do_throw  = !opts.allow_eps_too_small;             // opt-in wins
  fprintf(stderr, "%s %s: upsampfac=%.3g too low for tol=%.3g; %s\n", __func__,
          do_throw ? "error" : "warning", m.spopts.upsampfac, (double)m.tol,
          unachievable
              ? "rounding floor dominates (eps_round ~= tol); no upsampfac helps"
              : (opts.allow_eps_too_small ? "suggest upsampfac>=" : "need upsampfac>="));
  if (!unachievable) fprintf(stderr, "  (%.3g)\n", suggest);
  if (do_throw) throw finufft::exception(FINUFFT_ERR_EPS_TOO_SMALL);
}

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
   Previous args (opts, spopts) are now plan members; outputs (nf, h, gam) are
   written directly to plan members nfdim, t3P.h, t3P.gam.
   Converted to class member, Barbone 2/24/26.
*/
{
  using namespace finufft::common;
  using namespace finufft::utils;
  int nss = m.spopts.nspread + 1; // since ns may be odd
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
  m.nfdim[idim] = (BIGINT)nfd;
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (m.nfdim[idim] < 2 * m.spopts.nspread) m.nfdim[idim] = 2 * m.spopts.nspread;
  if (m.nfdim[idim] < MAX_NF)                     // otherwise will fail
    m.nfdim[idim] = next235even(m.nfdim[idim]);   // expensive at huge nf
  m.t3P.h[idim]   = TF(2.0 * PI / m.nfdim[idim]); // upsampled grid spacing
  m.t3P.gam[idim] = TF(m.nfdim[idim] / (2.0 * opts.upsampfac * Ssafe)); // x scale fac
}

// --------- setpts user guru interface driver ----------

template<typename TF>
int FINUFFT_PLAN_T<TF>::setpts(BIGINT nj, const TF *xj, const TF *yj, const TF *zj,
                               BIGINT nk, const TF *s, const TF *t, const TF *u) {
  using namespace finufft::utils;
  using namespace finufft::heuristics;
  // Method function to set NU points and do precomputations. Barnett 2020.
  // Barbone (3/4/26): removed warning_code_ plumbing (eps-too-small now throws).
  // See ../docs/cguru.doc for current documentation.
  int d = dim;       // abbrev for spatial dim
  CNTime timer;
  timer.start();
  m.nj = nj; // the user only now chooses how many NU (x,y,z) pts
  if (nj < 0) {
    fprintf(stderr, "[%s] nj (%lld) cannot be negative!\n", __func__, (long long)nj);
    throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  } else if (nj > MAX_NU_PTS) {
    fprintf(stderr, "[%s] nj (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nj);
    throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  }

  if (type != 3) { // ------------------ TYPE 1,2 SETPTS -------------------
                   // (all we can do is check and maybe bin-sort the NU pts)
    // If upsampfac is not locked by user (auto mode), choose or update it now
    // based on the actual density nj/N(). Re-plan if density changed significantly.
    if (!upsamp_locked) {
      double density   = double(nj) / double(N());
      double upsampfac = bestUpsamplingFactor<TF>(opts.nthreads, density, dim, type, m.tol);
      // Re-plan if this is the first call (upsampfac==0) or if upsampfac changed
      if (upsampfac != opts.upsampfac) {
        opts.upsampfac = upsampfac;
        if (opts.debug)
          printf("[setpts] selected best upsampfac=%.3g (density=%.3g, nj=%lld)\n",
                 opts.upsampfac, density, (long long)nj);
        setup_spreadinterp(); // throws on error
        precompute_horner_coeffs();
        // Perform the planning steps (first call or re-plan due to density change).
        init_grid_kerFT_FFT();       // throws on error
      }
    }

    check_sigma(); // throws if upsampfac too low for tol (nfdim now known)

    m.XYZ   = {xj, yj, zj}; // plan must keep pointers to user's fixed NU pts
    // Invariant: m.padded_ns must equal the runtime mirror of
    // KernelBufferLayout<TF, NS>::stride. Caught here if any path forgot to
    // call precompute_horner_coeffs (or if the trait diverges from the runtime
    // formula).
    assert(m.padded_ns ==
           finufft::spreadinterp::kernel_buffer_stride_runtime<TF>(m.spopts.nspread));
    spreadcheck();          // throws on error
    timer.restart();
    m.sortIndices.resize(nj);
    indexSort();
    if (opts.debug)
      printf("[%s] sort (didSort=%d):\t\t%.3g s\n", __func__, (int)m.didSort,
             timer.elapsedsec());

  } else { // ------------------------- TYPE 3 SETPTS -----------------------
           // (here we can precompute pre/post-phase factors and plan the t2)

    std::array<const TF *, 3> XYZ_in{xj, yj, zj};
    std::array<const TF *, 3> STU_in{s, t, u};
    if (nk < 0) {
      fprintf(stderr, "[%s] nk (%lld) cannot be negative!\n", __func__, (long long)nk);
      throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
    } else if (nk > MAX_NU_PTS) {
      fprintf(stderr, "[%s] nk (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nk);
      throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
    }
    m.nk = nk; // user set # targ freq pts
    m.STU = {s, t, u};

    // For type 3 with deferred upsampfac (not locked by user), pick and persist
    // an upsamp now using density=1.0 so that subsequent steps (set_nhg_type3 etc.)
    // have a concrete upsampfac to work with. This choice is persisted so inner
    // type-2 plans will inherit it.
    double upsampfac = bestUpsamplingFactor<TF>(opts.nthreads, 1.0, dim, type, m.tol);
    if (!upsamp_locked && (upsampfac != opts.upsampfac)) {
      opts.upsampfac = upsampfac;
      if (opts.debug > 1)
        printf("[setpts t3] selected upsampfac=%.2f (density=1 used; persisted)\n",
               opts.upsampfac);
      setup_spreadinterp(); // throws on error
      precompute_horner_coeffs();
    }

    // pick x, s intervals & shifts & # fine grid pts (nf) in each dim...
    std::array<TF, 3> S = {0, 0, 0};
    if (opts.debug) printf("\tM=%lld N=%lld\n", (long long)nj, (long long)nk);
    for (int idim = 0; idim < dim; ++idim) {
      arraywidcen(nj, XYZ_in[idim], &(m.t3P.X[idim]), &(m.t3P.C[idim]));
      arraywidcen(nk, STU_in[idim], &S[idim], &(m.t3P.D[idim]));
      set_nhg_type3(idim, S[idim], m.t3P.X[idim]); // applies twist i)
      if (opts.debug) // report on choices of shifts, centers, etc...
        printf("\tX%d=%.3g C%d=%.3g S%d=%.3g D%d=%.3g gam%d=%g nf%d=%lld h%d=%.3g\t\n",
               idim, m.t3P.X[idim], idim, m.t3P.C[idim], idim, S[idim], idim,
               m.t3P.D[idim], idim, m.t3P.gam[idim], idim, (long long)m.nfdim[idim],
               idim, m.t3P.h[idim]);
    }
    for (int idim = dim; idim < 3; ++idim)
      m.t3P.C[idim] = m.t3P.D[idim] = 0.0; // their defaults if dim 2 unused, etc

    if (nf() * batchSize > MAX_NF) {
      fprintf(stderr,
              "[%s t3] fwBatch would be bigger than MAX_NF, not attempting memory "
              "allocation!\n",
              __func__);
      throw finufft::exception(FINUFFT_ERR_MAXNALLOC);
    }

    // alloc rescaled NU src pts x'_j (in X etc), rescaled NU targ pts s'_k ...
    // We do this by resizing Xp, Yp, and Zp, and pointing X, Y, Z to their data;
    // this avoids any need for explicit cleanup.
    for (int idim = 0; idim < dim; ++idim) {
      m.XYZp[idim].resize(nj);
      m.XYZ[idim] = m.XYZp[idim].data();
      m.STUp[idim].resize(nk);
    }

    // always shift as use gam to rescale x_j to x'_j, etc (twist iii)...
    std::array<TF, 3> ig = {0, 0, 0};
    for (int idim = 0; idim < dim; ++idim) ig[idim] = 1.0 / m.t3P.gam[idim];
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT j = 0; j < nj; ++j) {
      for (int idim = 0; idim < dim; ++idim)
        m.XYZp[idim][j] = (XYZ_in[idim][j] - m.t3P.C[idim]) * ig[idim]; // rescale x_j
    }

    // set up prephase array...
    TF isign = (fftSign >= 0) ? 1 : -1;
    m.prephase.resize(nj);
    if (m.t3P.D[0] != 0.0 || m.t3P.D[1] != 0.0 || m.t3P.D[2] != 0.0) {
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
      for (BIGINT j = 0; j < nj; ++j) { // ... loop over src NU locs
        TF phase = 0;
        for (int idim = 0; idim < dim; ++idim) phase += m.t3P.D[idim] * XYZ_in[idim][j];
        m.prephase[j] = std::polar(TF(1), isign * phase); // Euler
      }
    } else
      for (BIGINT j = 0; j < nj; ++j)
        m.prephase[j] = {1.0, 0.0}; // *** or keep flag so no mult in exec??

    // create a 1D phihat evaluator
    Kernel_onedim_FT onedim_phihat(*this);

    // (old STEP 3a) Compute deconvolution post-factors array (per targ pt)...
    // (exploits that FT separates because kernel is prod of 1D funcs)
    m.deconv.resize(nk);
    // C can be nan or inf if M=0, no input NU pts
    bool Cfinite = std::isfinite(m.t3P.C[0]) && std::isfinite(m.t3P.C[1]) &&
                   std::isfinite(m.t3P.C[2]);
    bool Cnonzero = m.t3P.C[0] != 0.0 || m.t3P.C[1] != 0.0 || m.t3P.C[2] != 0.0;
    bool do_phase = Cfinite && Cnonzero;
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT k = 0; k < nk; ++k) { // .... loop over NU targ freqs
      TF phiHat = 1;
      TF phase  = 0;
      for (int idim = 0; idim < dim; ++idim) {
        auto tSTUin = STU_in[idim][k];
        // rescale the target s_k etc to s'_k etc...
        auto tSTUp =
            m.t3P.h[idim] * m.t3P.gam[idim] * (tSTUin - m.t3P.D[idim]); // |s'_k| < pi/R
        phiHat *= onedim_phihat(tSTUp);
        if (do_phase) phase += (tSTUin - m.t3P.D[idim]) * m.t3P.C[idim];
        m.STUp[idim][k] = tSTUp;
      }
      m.deconv[k] =
          do_phase ? std::polar(TF(1) / phiHat, isign * phase) : TF(1) / phiHat;
    }
    if (opts.debug)
      printf("[%s t3] phase & deconv factors:\t%.3g s\n", __func__, timer.elapsedsec());

    // Set up sort for spreading Cp (from primed NU src pts X, Y, Z) to fw...
    timer.restart();
    m.sortIndices.resize(nj);
    m.spopts.spread_direction = 1;
    indexSort();
    if (opts.debug)
      printf("[%s t3] sort (didSort=%d):\t\t%.3g s\n", __func__, (int)m.didSort,
             timer.elapsedsec());

    // Plan and setpts once, for the (repeated) inner type 2 finufft call...
    timer.restart();
    BIGINT t2nmodes[]   = {m.nfdim[0], m.nfdim[1], m.nfdim[2]}; // t2's input actually fw
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
    finufft_makeplan_t<TF>(2, d, t2nmodes, fftSign, batchSize, m.tol, &tmpplan,
                           &t2opts); // throws on error
    // Use a non-const unique_ptr to ensure cleanup if setpts throws, then
    // transfer to the const unique_ptr member.
    std::unique_ptr<FINUFFT_PLAN_T<TF>> guard(tmpplan);
    tmpplan->setpts(nk, m.STUp[0].data(), m.STUp[1].data(), m.STUp[2].data(), 0,
                    nullptr, nullptr,
                    nullptr); // note nk = # output points (not nj); throws on error
    m.innerT2plan = std::move(guard);
    if (opts.debug)
      printf("[%s t3] inner t2 plan & setpts: \t%.3g s\n", __func__, timer.elapsedsec());
  }
  return 0;
}
