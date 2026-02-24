#pragma once

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <finufft/fft.hpp>
#include <finufft/finufft_core.hpp>
#include <finufft/finufft_utils.hpp>
#include <finufft/heuristics.hpp>
#include <finufft/spreadinterp.hpp>
#include <finufft_common/kernel.h>

// ---------- local math routines for type-3 setpts: --------

namespace finufft {
namespace utils {

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
  using namespace finufft::common;
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
  if (!std::isfinite(nfd)) nfd = 0.0;
  *nf = (BIGINT)nfd;
  // printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  //  catch too small nf, and nan or +-inf, otherwise spread fails...
  if (*nf < 2 * spopts.nspread) *nf = 2 * spopts.nspread;
  if (*nf < MAX_NF)                               // otherwise will fail anyway
    *nf = next235even(*nf);                       // expensive at huge nf
  *h   = T(2.0 * PI / *nf);                       // upsampled grid spacing
  *gam = T(*nf / (2.0 * opts.upsampfac * Ssafe)); // x scale fac to x'
}

template<typename T> class Kernel_onedim_FT {
private:
  std::vector<T> z, f; // internal arrays

public:
  /*
    Approximates exact 1D Fourier transform of spreadinterp's real symmetric
    kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
    narrowness of kernel. Evaluates at set of arbitrary freqs k in [-pi, pi),
    for a kernel with x measured in grid-spacings. (See previous routine for
    FT definition.). Note: old (pre-2025) name was: onedim_nuft_kernel().

    Inputs:
    opts - spreading opts object, needed to eval kernel (must be already set up)

    Barnett 2/8/17. openmp since cos slow 2/9/17.
    11/25/25, replaced kernel_definition by evaluate_kernel_runtime, so that
    the FT of the piecewise poly approximant (not "exact" kernel) is computed.
   */

  Kernel_onedim_FT(const finufft_spread_opts &opts, const T *horner_coeffs_ptr, int nc) {
    using finufft::common::gaussquad;
    using finufft::spreadinterp::evaluate_kernel_runtime;
    // creator: uses slow kernel evals to initialize z and f arrays.
    T J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
    // # quadr nodes in z (from 0 to J/2; reflections will be added)...
    int q = (int)(2 + 2.0 * J2); // > pi/2 ratio.  cannot exceed MAX_NQUAD
    if (opts.debug) printf("q (# ker FT quadr pts) = %d\n", q);
    std::vector<double> Z(2 * q), W(2 * q);
    gaussquad(2 * q, Z.data(), W.data()); // only half the nodes used,
                                          // for (0,1)
    z.resize(q);
    f.resize(q);
    for (int n = 0; n < q; ++n) {
      z[n] = T(Z[n] * J2); // quadr nodes for [0,J/2] with weights J2 * w
      f[n] = J2 * T(W[n]) *
             evaluate_kernel_runtime<T>(z[n], opts.nspread, nc, horner_coeffs_ptr, opts);
    }
  }

  /*
    Evaluates the Fourier transform of the kernel at a single point, using
    the z and f arrays from creation time.

    Inputs:
    k - frequency, dual to the kernel's natural argument, ie exp(i.k.z)

    Outputs:
    phihat - real Fourier transform evaluated at freq k
   */
  FINUFFT_ALWAYS_INLINE T operator()(T k) {
    T x = 0;
    for (size_t n = 0; n < z.size(); ++n)
      x += f[n] * 2 * std::cos(k * z[n]); // pos & neg freq pair.  use T cos!
    return x;
  }
};

} // namespace utils
} // namespace finufft

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
    didSort =
        indexSort(sortIndices, nfdim[0], nfdim[1], nfdim[2], nj, xj, yj, zj, spopts);
    if (opts.debug)
      printf("[%s] sort (didSort=%d):\t\t%.3g s\n", __func__, didSort,
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
      set_nhg_type3(S[idim], t3P.X[idim], opts, spopts, &(nfdim[idim]), &(t3P.h[idim]),
                    &(t3P.gam[idim]));                         // applies twist i)
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
    Kernel_onedim_FT<TF> onedim_phihat(spopts, horner_coeffs.data(), nc);

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
    didSort = indexSort(sortIndices, nfdim[0], nfdim[1], nfdim[2], nj, XYZ[0], XYZ[1],
                        XYZ[2], spopts);
    if (opts.debug)
      printf("[%s t3] sort (didSort=%d):\t\t%.3g s\n", __func__, didSort,
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
