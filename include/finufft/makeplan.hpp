#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

#include <finufft/plan.hpp>
#include <finufft/spreadinterp.hpp>
#include <finufft/utils.hpp>
#include <finufft_common/common.h>

// ---------- local math routines (were in common.cpp; no need now): --------

template<typename TF>
void FINUFFT_PLAN_T<TF>::set_nf_type12(BIGINT ms, BIGINT *nf) const
// Type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms. Throws on error if nf is
// unreasonably big. Previous args (opts, spopts) are now plan members.
// Converted to class member, Barbone 2/24/26.
{
  using namespace finufft::common;
  *nf = BIGINT(std::ceil(opts.upsampfac * double(ms))); // round up to handle small cases
  if (*nf < 2 * m.spopts.nspread) *nf = 2 * m.spopts.nspread; // otherwise spread fails
  if (*nf < MAX_NF) {
    *nf = next235(*nf, 2);
  } else {
    fprintf(stderr,
            "[%s] nf=%.3g exceeds MAX_NF of %.3g, so exit without attempting "
            "memory allocation\n",
            __func__, (double)*nf, (double)MAX_NF);
    throw finufft::exception(FINUFFT_ERR_MAXNALLOC);
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
  Previous arg opts (spreading opts) is now read from plan member spopts.
  Converted to class member, Barbone 2/24/26.
 */
{
  using namespace finufft::common;
  TF J2 = m.spopts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 3.0 * J2); // not sure why so large? (NB cannot exceed MAX_NQUAD)
  TF f[MAX_NQUAD];
  double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
  gaussquad(2 * q, z, w);       // only half the nodes used, eg on (0,1)
  std::complex<TF> a[MAX_NQUAD];
  for (int n = 0; n < q; ++n) { // set up nodes z_n and vals f_n
    z[n] *= J2;                 // rescale nodes
                                // vals & quadr weighs
    f[n] = J2 * (TF)w[n] * evaluate_kernel_runtime(TF(z[n]));
    // phase winding rates
    a[n] = -std::exp(2 * PI * std::complex<double>(0, 1) * z[n] / double(nf));
  }
  BIGINT nout = nf / 2 + 1; // how many values we're writing to
  int nt      = std::min(nout, (BIGINT)m.spopts.nthreads); // how many chunks
  std::vector<BIGINT> brk(nt + 1); // start indices for each thread
  for (int t = 0; t <= nt; ++t)    // split nout mode indices btw threads
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

template<typename TF> void FINUFFT_PLAN_T<TF>::setup_spreadinterp() {
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
    Throws on error (see codes in finufft_errors.h), including
    FINUFFT_ERR_EPS_TOO_SMALL if requested eps (tol) is below machine epsilon,
    unless opts.allow_eps_too_small requests clamp-and-proceed behavior.
    Barbone (Dec/25): ensure legacy kereval/kerpad user opts are treated as no-ops.
    1/8/26: Barnett redo (merges setup_spreader & setup_spreader_for_nufft of 2017).
    Barbone (3/4/26): eps-too-small is now a hard error (throw), not a warning.
  */

  m.spopts.nthreads     = opts.nthreads; // 0 passed in becomes OMP max avail spreadinterp
  m.spopts.sort         = opts.spread_sort;  // todo: could make dim or CPU choices here?
  m.spopts.sort_threads = 0;                 // 0:auto-choice
  m.spopts.debug        = opts.spread_debug; // simple pass-through
  m.spopts.upsampfac    = opts.upsampfac;    // "
  // sanity check sigma (upsampfac)...
  if (m.spopts.upsampfac <= 1.0) { // no digits would result, ns infinite
    fprintf(stderr, "[%s] error: upsampfac=%.3g is not > 1.0!\n", __func__,
            m.spopts.upsampfac);
    throw finufft::exception(FINUFFT_ERR_UPSAMPFAC_TOO_SMALL);
  }
  if (opts.showwarn && !opts.spreadinterponly &&
      (m.spopts.upsampfac < 1.15 || m.spopts.upsampfac > 3.0))
    fprintf(stderr,
            "%s warning: upsampfac=%.3g outside [1.15, 3.0] is unlikely to provide "
            "benefit and may break the library;\n",
            __func__, m.spopts.upsampfac);

  // crucial: where the default kerformula is set ....*    see kernel.{h,cpp}
  m.spopts.kerformula = (opts.spread_kerformula == 0) ? 8 : opts.spread_kerformula;

  constexpr TF EPSILON = std::numeric_limits<TF>::epsilon(); // 2.2e-16 or 1.2e-7
  if (m.tol < EPSILON) { // unfeasible request: no hope of beating eps_mach...
    if (opts.allow_eps_too_small) {
      m.tol = EPSILON;
    } else {
      fprintf(stderr, "%s error: requested tol=%.3g is below eps_mach=%.3g.\n", __func__,
              (double)m.tol, (double)EPSILON);
      throw finufft::exception(FINUFFT_ERR_EPS_TOO_SMALL);
    }
  }

  // choose nspread and set it in spopts. The width actually used comes from the shared
  // clamp_kernel_ns() (kernel.h); here we additionally report/throw when the requested
  // tol is infeasible, and warn when the float guard narrows the kernel.
  const int ns_theory = theoretical_kernel_ns((double)m.tol, dim, type, m.spopts);
  // per-precision cap: float spreadinterp is only instantiated up to
  // MAX_NSPREAD<TF> (see constants.h, issue #827)
  constexpr int max_ns = MAX_NSPREAD<TF>;
  if (ns_theory > max_ns && !opts.allow_eps_too_small) {
    fprintf(stderr,
            "%s error: at upsampfac=%.3g, tol=%.3g would need kernel "
            "width ns=%d, exceeding max %d.\n",
            __func__, m.spopts.upsampfac, (double)m.tol, ns_theory, max_ns);
    throw finufft::exception(FINUFFT_ERR_EPS_TOO_SMALL);
  }
  // catastrophic-cancellation guard narrows float below FLOAT_CC_UPSAMPFAC_LIMIT to
  // FLOAT_MAX_NS_CC (both in constants.h; the clamp is in clamp_kernel_ns; warn here
  // when it actually bites)...
  const bool singleprec = std::is_same_v<TF, float>;
  if (opts.showwarn && singleprec && m.spopts.upsampfac < FLOAT_CC_UPSAMPFAC_LIMIT &&
      std::min(ns_theory, max_ns) > FLOAT_MAX_NS_CC)
    fprintf(stderr,
            "%s warning: ns reducing from %d to %d to prevent r_{dyn}-related"
            "catastrophic cancellation.\n",
            __func__, std::min(ns_theory, max_ns), FLOAT_MAX_NS_CC);
  const int ns = clamp_kernel_ns<TF>(ns_theory, m.spopts.upsampfac);
  m.spopts.nspread = ns;
  set_kernel_shape_given_ns(m.spopts, opts.debug); // selects kernel params in spopts
  if (opts.debug || m.spopts.debug)
    printf("\t\t\ttol=%.3g sigma=%.3g: chose ns=%d beta=%.3g\n", m.tol,
           m.spopts.upsampfac, ns, m.spopts.beta);

  // heuristic dir=1 chunking for nthr>>1, typical for intel i7 and skylake...
  m.spopts.max_subproblem_size = (dim == 1) ? 10000 : 100000; // todo: revisit
  if (opts.spread_max_sp_size > 0)                            // override
    m.spopts.max_subproblem_size = opts.spread_max_sp_size;
  // nthr above which switch OMP critical->atomic (add_wrapped..). R Blackwell's val:
  m.spopts.atomic_threshold =
      (opts.spread_nthr_atomic >= 0) ? opts.spread_nthr_atomic : 10;
}

// ------------------- piecewise-poly Horner setup utility -----------------
template<typename TF> void FINUFFT_PLAN_T<TF>::precompute_horner_coeffs() {
  using namespace finufft::utils;
  using namespace finufft::common;
  using namespace finufft::kernel;
  // Solve for the piecewise Horner coeffs of the kernel in m.spopts. The fit itself
  // lives in finufft::kernel::fit_horner_coeffs (finufft_common/kernel.h), shared
  // with the GPU plan; this method only supplies the CPU buffer layout.
  // *** To-do: investigate tol_cutoff, 1/13/26.
  const auto nspread = m.spopts.nspread;
  const auto nc_fit    = max_nc_given_ns<TF>(nspread); // how many coeffs to fit

  // Both the chunk stride here and the per-chunk stride in evaluate_kernel_vector
  // flow through KernelBufferLayout<TF, NS>::stride (compile-time) and
  // kernel_buffer_stride_runtime<TF>(ns) (runtime mirror), so they provably agree.
  m.padded_ns          = finufft::spreadinterp::kernel_buffer_stride_runtime<TF>(nspread);
  const auto simd_size = get_padded_simd_width<TF>(2 * nspread);

  // Coeff cut-off relative to tol, below which a degree is dropped: to-do make opts?
  const double coeffs_tol_cutoff = 0.05;

  CNTime timer;
  timer.start();
  m.horner_coeffs.fill(TF(0));
  m.nc = fit_horner_coeffs<TF>(m.spopts, nc_fit, m.padded_ns,
                               double(m.tol) * coeffs_tol_cutoff, m.horner_coeffs.data());
  double t = timer.elapsedsec();

  if (opts.debug || m.spopts.debug) {
    printf("[%s] ns=%d:\t%.3g s\n", __func__, nspread, t);
    printf("\t\tnc_fit=%d (trim to nc=%d), simd_size=%d, padded_ns=%d\n", nc_fit, m.nc,
           (int)simd_size, (int)m.padded_ns);
  }
  if (opts.debug > 2) {
    // Print transposed layout: all "index 0" coeffs for intervals, then "index 1", ...
    // Note: k is the coefficient index in Horner order, with highest degree first.
    printf("dumping precomputed Horner coeffs...\n");
    for (size_t k = 0; k < static_cast<size_t>(m.nc); ++k) {
      printf("[%s] idx=%lld: ", __func__, (long long)k);
      for (size_t j = 0; j < m.padded_ns; ++j) // use padded_ns to show padding as well
        printf("%.14g ", static_cast<double>(m.horner_coeffs[k * m.padded_ns + j]));
      printf("\n");
    }
  }
}

template<typename TF>
FINUFFT_PLAN_T<TF>::FINUFFT_PLAN_T(int type_, int dim_, const BIGINT *n_modes, int iflag,
                                   int ntrans_, TF tol_, const finufft_opts *opts_)
    : type(type_), dim(dim_), ntrans(ntrans_)
// Constructor for finufft_plan object.
// opts is ptr to a finufft_opts to set options, or nullptr to use defaults.
// For some of the fields (if "auto" selected) here choose the actual setting.
// For types 1,2 allocates memory for internal working arrays,
// evaluates spreading kernel coefficients, and does FFT plan if needed.
// Throws finufft::exception on error.
{
  using namespace finufft::utils;
  m.tol = tol_;    // save user tolerance (setup_spreadinterp may clamp it)
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
      throw finufft::exception(FINUFFT_ERR_LOCK_FUNS_INVALID);
    }
    create_fft_plan(); // needs complete Finufft_FFT_plan type; defined in fft.cpp
  }
  if ((type != 1) && (type != 2) && (type != 3)) {
    fprintf(stderr, "[%s] Invalid type (%d), should be 1, 2 or 3.\n", __func__, type);
    throw finufft::exception(FINUFFT_ERR_TYPE_NOTVALID);
  }
  if ((dim != 1) && (dim != 2) && (dim != 3)) {
    fprintf(stderr, "[%s] Invalid dim (%d), should be 1, 2 or 3.\n", __func__, dim);
    throw finufft::exception(FINUFFT_ERR_DIM_NOTVALID);
  }
  if (ntrans < 1) {
    fprintf(stderr, "[%s] ntrans (%d) should be at least 1.\n", __func__, ntrans);
    throw finufft::exception(FINUFFT_ERR_NTRANS_NOTVALID);
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
    throw finufft::exception(FINUFFT_ERR_NTHREADS_NOTVALID);
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
  if (opts.showwarn) {
    const char *const fn = __func__; // __func__ inside the lambda is operator()
    const auto warn      = [fn](const char *name) {
      finufft::common::warn_deprecated_opt(fn, name);
    };
    FINUFFT_DIAGNOSTIC_PUSH
    FINUFFT_DISABLE_WARNING_DEPRECATED
    if (opts.spread_thread != 0) warn("spread_thread");
    if (opts.spread_kerevalmeth != 1) warn("spread_kerevalmeth");
    if (opts.spread_kerpad != 1) warn("spread_kerpad");
    FINUFFT_DIAGNOSTIC_POP
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
    setup_spreadinterp(); // throws on error
    precompute_horner_coeffs();

    //  ------------------------ types 1,2: planning needed ---------------------
    if (type == 1 || type == 2) {
      init_grid_kerFT_FFT(); // throws on error
    }
  } else {
    // If upsampfac was left as 0.0 (auto) we defer setup_spreader to setpts.
    // However, we can still reject or clamp unachievable tiny tolerances now.
    const TF eps_mach = std::numeric_limits<TF>::epsilon();
    if (m.tol < eps_mach) {
      if (opts.allow_eps_too_small) {
        m.tol = eps_mach;
      } else
        throw finufft::exception(FINUFFT_ERR_EPS_TOO_SMALL);
    }
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
// Returns 0 on success. Errors throw and are caught by safe_finufft_call at the
// C boundary.
{
  *pp = nullptr;
  *pp = new FINUFFT_PLAN_T<TF>(type, dim, n_modes, iflag, ntrans, tol, opts);
  return 0;
}
