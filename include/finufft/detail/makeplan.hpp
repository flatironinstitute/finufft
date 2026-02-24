#pragma once

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <finufft/finufft_core.hpp>
#include <finufft/finufft_utils.hpp>
#include <finufft/heuristics.hpp>
#include <finufft/spreadinterp.hpp>
#include <finufft/xsimd.hpp>
#include <finufft_common/kernel.h>
#include <finufft_common/utils.h>

// ---------- local math routines (were in common.cpp; no need now): --------

template<typename TF>
int FINUFFT_PLAN_T<TF>::set_nf_type12(BIGINT ms, BIGINT *nf) const
// Type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms. Returns 0 if success, else an
// error code if nf was unreasonably big (& tell the world).
// 2/24/26 Barbone: converted from free function to method on FINUFFT_PLAN_T.
// Previous args (opts, spopts) are now plan members; only ms and nf remain.
{
  using namespace finufft::utils;
  *nf = BIGINT(std::ceil(opts.upsampfac * double(ms))); // round up to handle small cases
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

template<typename TF>
void FINUFFT_PLAN_T<TF>::onedim_fseries_kernel(BIGINT nf,
                                               std::vector<TF> &fwkerhalf) const
/*
  Approximates exact Fourier series coeffs of spreadinterp's real symmetric
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
  Reads spopts (spreading opts) from the plan, needed to eval kernel.

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
        divided by h = 2pi/n.
        (should be allocated for at least nf/2+1 TFs)

  [Compare long-gone onedim_dct_kernel which had same interface, but computed DFT
  of sampled kernel, not quite the same object. This was from 2017-ish.]

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18.
  Fixed num_threads 7/20/20. Reduced rounding error in a[n] calc 8/20/24.
  11/25/25, replaced kernel_definition by evaluate_kernel_runtime, meaning that
  the FT of the piecewise poly approximant (not "exact" kernel) is computed.
  2/24/26 Barbone: converted from free function to method on FINUFFT_PLAN_T.
  Previous arg opts (spreading opts) is now read from plan member spopts.
 */
{
  using namespace finufft::common;
  TF J2 = spopts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 3.0 * J2); // not sure why so large? (NB cannot exceed MAX_NQUAD)
  TF f[MAX_NQUAD];
  double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
  gaussquad(2 * q, z, w); // only half the nodes used, eg on (0,1)
  std::complex<TF> a[MAX_NQUAD];
  for (int n = 0; n < q; ++n) {            // set up nodes z_n and vals f_n
    z[n] *= J2;                            // rescale nodes
                                           // vals & quadr weighs
    f[n] = J2 * (TF)w[n] * evaluate_kernel_runtime((TF)z[n]);
    // phase winding rates
    a[n] = -std::exp(2 * PI * std::complex<double>(0, 1) * z[n] / double(nf));
  }
  BIGINT nout = nf / 2 + 1; // how many values we're writing to
  int nt      = std::min(nout, (BIGINT)spopts.nthreads); // how many chunks
  std::vector<BIGINT> brk(nt + 1);                       // start indices for each thread
  for (int t = 0; t <= nt; ++t) // split nout mode indices btw threads
    brk[t] = (BIGINT)(0.5 + nout * t / (double)nt);
#pragma omp parallel num_threads(nt)
  {                                                // each thread gets own chunk to do
    int t = MY_OMP_GET_THREAD_NUM();
    std::complex<TF> aj[MAX_NQUAD];                // phase rotator for this thread
    for (int n = 0; n < q; ++n)
      aj[n] = std::pow(a[n], (TF)brk[t]);          // init phase factors for chunk
    for (BIGINT j = brk[t]; j < brk[t + 1]; ++j) { // loop along output array
      TF x = 0.0;                                  // accumulator for answer at this j
      for (int n = 0; n < q; ++n) {
        x += f[n] * 2 * std::real(aj[n]);          // include the negative freq
        aj[n] *= a[n];                             // wind the phases
      }
      fwkerhalf[j] = x;
    }
  }
}

// --------------- makeplan-related member functions and free functions ----------

template<typename TF> int FINUFFT_PLAN_T<TF>::setup_spreadinterp() {
  using namespace finufft::common;
  using namespace finufft::kernel;
  /* Sets spread/interp (gridding) kernel params in spopts struct (ns, etc),
    based on:
    tol - desired user relative tolerance (a.k.a eps)
    opts.upsampfac - fixed upsampling factor (=sigma), previously set.
    opts.kerformula - kernel function type (chooses the default, override if >0)
    All of these (spopts, opts, tol) are plan class members.
    See finufft_common/spread_opts.h for docs on all spopts fields.
    Note that spopts.spread_direction is not set.
    Returns: 0  : success
              FINUFFT_WARN_EPS_TOO_SMALL : requested eps (tol) cannot be achieved,
                                           but proceed with best possible eps.
              otherwise : failure (see codes in finufft_errors.h); spreading must
                          not proceed.
    Barbone (Dec/25): ensure legacy kereval/kerpad user opts are treated as no-ops.
    1/8/26: Barnett redo (merges setup_spreader & setup_spreader_for_nufft of 2017).
  */
  spopts.nthreads     = opts.nthreads; // 0 passed in becomes OMP max avail spreadinterp
  spopts.sort         = opts.spread_sort;  // todo: could make dim or CPU choices here?
  spopts.sort_threads = 0;                 // 0:auto-choice
  spopts.debug        = opts.spread_debug; // simple pass-through
  spopts.upsampfac    = opts.upsampfac;    // "
  // sanity check sigma (upsampfac)...
  if (spopts.upsampfac <= 1.0) { // no digits would result, ns infinite
    fprintf(stderr, "[%s] error: upsampfac=%.3g is not > 1.0!\n", __func__,
            spopts.upsampfac);
    return FINUFFT_ERR_UPSAMPFAC_TOO_SMALL;
  }
  if (opts.showwarn && !opts.spreadinterponly &&
      (spopts.upsampfac < 1.15 || spopts.upsampfac > 3.0))
    fprintf(stderr,
            "%s warning: upsampfac=%.3g outside [1.15, 3.0] is unlikely to provide "
            "benefit and may break the library;\n",
            __func__, spopts.upsampfac);

  // crucial: where the default kerformula is set ....*    see kernel.{h,cpp}
  spopts.kerformula = (opts.spread_kerformula == 0) ? 8 : opts.spread_kerformula;

  constexpr TF EPSILON = std::numeric_limits<TF>::epsilon(); // 2.2e-16 or 1.2e-7
  int ier              = 0;
  if (tol < EPSILON) { // unfeasible request: no hope of beating eps_mach...
    if (opts.showwarn)
      fprintf(stderr, "%s warning: increasing tol=%.3g to eps_mach=%.3g.\n", __func__,
              (double)tol, (double)EPSILON);
    tol = EPSILON; // ... so forget the user request and target eps_mach
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }

  // choose nspread and set it in spopts...
  int ns = theoretical_kernel_ns((double)tol, dim, type, opts.debug, spopts);
  ns     = std::max(MIN_NSPREAD, ns); // clip low
  if (ns > MAX_NSPREAD) {             // clip to largest spreadinterp.cpp allows
    if (opts.showwarn)
      fprintf(stderr,
              "%s warning: at upsampfac=%.3g, tol=%.3g would need kernel "
              "width ns=%d; clipping to max %d.\n",
              __func__, spopts.upsampfac, (double)tol, ns, MAX_NSPREAD);
    ns  = MAX_NSPREAD;
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }
  // further ns reduction to prevent catastrophic cancellation in float...
  const bool singleprec = std::is_same_v<TF, float>;
  if (singleprec && spopts.upsampfac < 1.4) {
    int max_ns_CC = 8; // hacky, const, found via tolsweeptest.m (type 3 was 7)
    if (ns > max_ns_CC) {
      if (opts.showwarn)
        fprintf(stderr,
                "%s warning: ns reducing from %d to %d to prevent r_{dyn}-related"
                "catastrophic cancellation.\n",
                __func__, ns, max_ns_CC);
      ns = max_ns_CC;
    }
  }
  spopts.nspread = ns;
  set_kernel_shape_given_ns(spopts, opts.debug); // selects kernel params in spopts
  if (opts.debug || spopts.debug)
    printf("\t\t\ttol=%.3g sigma=%.3g: chose ns=%d beta=%.3g (ier=%d)\n", tol,
           spopts.upsampfac, ns, spopts.beta, ier);

  // heuristic dir=1 chunking for nthr>>1, typical for intel i7 and skylake...
  spopts.max_subproblem_size = (dim == 1) ? 10000 : 100000; // todo: revisit
  if (opts.spread_max_sp_size > 0)                          // override
    spopts.max_subproblem_size = opts.spread_max_sp_size;
  // nthr above which switch OMP critical->atomic (add_wrapped..). R Blackwell's val:
  spopts.atomic_threshold = (opts.spread_nthr_atomic >= 0) ? opts.spread_nthr_atomic : 10;

  return ier;
}

// ------------------- piecewise-poly Horner setup utility -----------------
template<typename TF> void FINUFFT_PLAN_T<TF>::precompute_horner_coeffs() {
  using namespace finufft::utils;
  using namespace finufft::common;
  using namespace finufft::kernel;
  // Solve for piecewise Horner coeffs for the function kernel.h:kernel_definition()
  // Marco Barbone, Fall 2025. Barnett & Lu edits and two bugs fixed, Jan 2026.
  // *** To-do: investigate using double when TF=float, and tol_cutoff, 1/13/26.
  const auto nspread = spopts.nspread;

  const auto nc_fit = max_nc_given_ns(nspread); // how many coeffs to fit

  // get the xsimd padding
  // (must match that used in spreadinterp.cpp: if we change horner simd_width there
  // we must also change it here)
  const auto simd_size = GetPaddedSIMDWidth<TF>(2 * nspread);
  padded_ns            = (nspread + simd_size - 1) & -simd_size;

  horner_coeffs.fill(TF(0));

  nc = MIN_NC; // a class member which will become the number of coeffs used

  CNTime timer;
  timer.start();

  // First pass: fit at max_degree (nc_fit-1), and save these coeffs,
  // then determine largest nc needed and shuffle the coeffs if nc<nc_fit.
  // Note: `poly_fit()` returns coefficients in descending-degree order
  // (highest-degree first): coeffs[0] is the highest-degree term. We store
  // them so that `horner_coeffs[k * padded_ns + j]` holds the k'th Horner
  // coefficient (k=0 -> highest-degree). `horner_coeffs` was filled with
  // zeros above, so panels that need fewer coefficients leave the rest as 0.

  for (int j = 0; j < nspread; ++j) { // ......... loop over intervals (panels)
    // affine map of x in [-1,1] to z in jth interval [-1+2j/w,-1+2(j+1)/w]
    const TF xshiftj = TF(2 * j + 1 - nspread); // jth center in [-w,w]
    // *** explore making this lambda double always, like kernel itself:
    const auto kernel_this_interval = [xshiftj, this, nspread](TF x) -> TF {
      const TF z = (x + xshiftj) / (TF)nspread;
      return (TF)kernel_definition(spopts, (double)z);
    };

    // we're fitting in float for TF=float, *** explore always double:
    const auto coeffs = poly_fit<TF>(kernel_this_interval, static_cast<int>(nc_fit));

    // Save coefficients directly into final table (transposed/padded):
    // coeffs[k] is highest->lowest, store at row k for panel j.
    for (size_t k = 0; k < coeffs.size(); ++k) {
      horner_coeffs[k * padded_ns + j] = coeffs[k];
    }

    // Truncate polynomial degree using a numerical coeff size cut-off:
    // truncation at nc is allowed if all coeffs of degree nc have magnitude
    // less than tol * coeffs_tol_cutoff. The smallest such nc is found.
    // Experiments showed with this as 0.1, ns=15 still had err bump...
    const TF coeffs_tol_cutoff = 0.05; // coeffs cut-off rel to tol: to-do make opts?
    // Note: ordering is coeffs[0] highest degree, to coeffs[nc_fit-1] const term.
    int nc_needed = 0; // initialize. then step down from highest degree...
    for (size_t k = 0; k < coeffs.size(); ++k) {            // power is nc_fit-1-k
      if (std::abs(coeffs[k]) >= tol * coeffs_tol_cutoff) { // stop when large enough
        nc_needed = static_cast<int>(coeffs.size() - k);
        break;
      }
    }
    if (nc_needed > nc) nc = nc_needed; // nc update to be max over panels j
  } // .............. end loop

  // nc = nc_fit;  // overrides truncation, useful for debugging
  //     prevent nc falling off bottom of valid range...
  nc = std::max(nc, min_nc_given_ns(nspread));
  // (we know nc cannot be larger than valid due to nc_fit initialization above)

  // If the max required degree (nc) is less than nc_fit, we must shift
  // the coefficients "left" (to lower row indices) so that the significant
  // coefficients end at row nc-1.
  if (nc < static_cast<int>(nc_fit)) {
    const size_t shift = nc_fit - nc;
    for (size_t k = 0; k < static_cast<size_t>(nc); ++k) {
      const size_t src_row = k + shift;
      const size_t dst_row = k;
      for (size_t j = 0; j < padded_ns; ++j) {
        horner_coeffs[dst_row * padded_ns + j] = horner_coeffs[src_row * padded_ns + j];
      }
    }
    // Zero out the now-unused tail rows for cleanliness
    for (size_t k = nc; k < static_cast<size_t>(nc_fit); ++k) {
      for (size_t j = 0; j < padded_ns; ++j) {
        horner_coeffs[k * padded_ns + j] = TF(0);
      }
    }
  }
  double t = timer.elapsedsec();

  if (opts.debug || spopts.debug) {
    printf("[%s] ns=%d:\t%.3g s\n", __func__, nspread, t);
    printf("\t\tnc_fit=%d (trim to nc=%d), simd_size=%d, padded_ns=%d\n", nc_fit, nc,
           (int)simd_size, (int)padded_ns);
  }
  if (opts.debug > 2) {
    // Print transposed layout: all "index 0" coeffs for intervals, then "index 1", ...
    // Note: k is the coefficient index in Horner order, with highest degree first.
    printf("dumping precomputed Horner coeffs...\n");
    for (size_t k = 0; k < static_cast<size_t>(nc); ++k) {
      printf("[%s] idx=%lu: ", __func__, k);
      for (size_t j = 0; j < padded_ns; ++j) // use padded_ns to show padding as well
        printf("%.14g ", static_cast<double>(horner_coeffs[k * padded_ns + j]));
      printf("\n");
    }
  }
}

template<typename TF>
FINUFFT_PLAN_T<TF>::FINUFFT_PLAN_T(int type_, int dim_, const BIGINT *n_modes, int iflag,
                                   int ntrans_, TF tol_, const finufft_opts *opts_,
                                   int &ier)
    : type(type_), dim(dim_), ntrans(ntrans_), tol(tol_)
// Constructor for finufft_plan object.
// opts is ptr to a finufft_opts to set options, or nullptr to use defaults.
// For some of the fields (if "auto" selected) here choose the actual setting.
// For types 1,2 allocates memory for internal working arrays,
// evaluates spreading kernel coefficients, and does FFT plan if needed.
// ier is an output written to pass out warning codes (errors now thrown in C++ style).
{
  using namespace finufft::utils;
  if (!opts_)      // use default opts
    finufft_default_opts_t(&opts);
  else             // or read from what's passed in
    opts = *opts_; // keep a deep copy; changing *opts_ now has no effect

  if (opts.debug)  // do a hello world
    printf("[%s] new plan: FINUFFT version " FINUFFT_VER " .................\n",
           __func__);

  if (!opts.spreadinterponly) { // Don't make FFTW plan if only spread/interpolate
    if (!opts.fftw_lock_fun != !opts.fftw_unlock_fun) {
      fprintf(stderr, "[%s] fftw_(un)lock functions should be both null or both set\n",
              __func__);
      throw int(FINUFFT_ERR_LOCK_FUNS_INVALID);
    }
    create_fft_plan(); // needs complete Finufft_FFT_plan type; defined in fft.cpp
  }
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

  // get stuff from args...
  fftSign = (iflag >= 0) ? 1 : -1; // clean up flag input

  CNTime timer{};
  if (opts.debug > 1) {
    timer.start();
  }
#ifdef _OPENMP
  // choose overall # threads...
  int ompmaxnthr = static_cast<int>(getOptimalThreadCount());
  int nthr       = ompmaxnthr; // default: use as many physical cores as possible
  // (the above could be set, or suggested set, to 1 for small enough problems...)
  if (opts.nthreads > 0) {
    nthr = opts.nthreads; // user override, now without limit
    if (opts.showwarn && (nthr > ompmaxnthr))
      fprintf(stderr,
              "%s warning: using opts.nthreads=%d, more than the %d physically cores "
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
  if (opts.debug > 1) {
    printf("[%s] opts.nthreads=%d\n", __func__, nthr);
  }
  if (opts.nthreads == 0) {
    fprintf(stderr,
            "[%s] error: detecting physical corers failed. Please specify the number "
            "of cores to use\n",
            __func__);
    throw int(FINUFFT_ERR_NTHREADS_NOTVALID);
  }
  if (opts.debug > 1) {
    const auto sec = timer.elapsedsec();
    fprintf(stdout, "[%s] detected %d threads in %.3g sec.\n", __func__, nthr, sec);
  }

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

  if (type != 3) { // read in user Fourier mode array sizes...
    for (int idim = 0; idim < 3; ++idim) {
      mstu[idim] = (idim < dim) ? n_modes[idim] : 1;
    }
  }

  // heuristic to choose default upsampfac: defer selection to setpts unless
  // the user explicitly forced a nonzero value in opts. In that case initialize
  // spreader/Horner internals now using the provided upsampfac.
  if (opts.upsampfac != 0.0) {
    upsamp_locked = true; // user explicitly set upsampfac, don't auto-update
    if (opts.debug) printf("\t\tuser locked upsampfac=%g\n", opts.upsampfac);
    ier = setup_spreadinterp();
    if (ier > 1) // proceed if success or warning
      throw int(ier);
    precompute_horner_coeffs();

    //  ------------------------ types 1,2: planning needed ---------------------
    if (type == 1 || type == 2) {
      int code = init_grid_kerFT_FFT();
      if (code) throw code;
    }
  } else {
    // If upsampfac was left as 0.0 (auto) we defer setup_spreader to setpts.
    // However, we may still warn the user now if tol is guaranteed unachievable:
    if (tol < std::numeric_limits<TF>::epsilon()) ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }

  if (type == 3) { // -------------------------- type 3 (no planning) ------------

    if (opts.debug) printf("[%s] %dd%d: ntrans=%d\n", __func__, dim, type, ntrans);
    // Type 3 will call finufft_makeplan for type 2; no need to init FFTW
    // Note we don't even know nj or nk yet, so can't do anything else!
  }
}

template<typename TF>
int finufft_makeplan_t(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans,
                       TF tol, FINUFFT_PLAN_T<TF> **pp, const finufft_opts *opts)
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
