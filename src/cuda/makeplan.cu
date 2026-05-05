// "makeplan" stage: cufinufft_plan_t<T> ctor and the helpers it calls.
// Mirrors CPU src/makeplan.cpp. Also hosts the cufft_plan RAII destructor
// and the have_pool_support warning helper, which are tied to plan setup.

#include <iostream>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

#include <cufinufft.h>
#include <cufinufft/fft.hpp>
#include <cufinufft/heuristics.hpp>
#include <cufinufft/spreadinterp.hpp>
#include <cufinufft/types.hpp>
#include <cufinufft/utils.hpp>

#include <finufft_common/constants.h>
#include <finufft_errors.h>
#include <thrust/device_vector.h>

cufft_plan::~cufft_plan() {
  if (handle_ != 0) {
    DeviceSwitcher switcher(device_id_);
    cufftDestroy(handle_);
  }
}

static bool have_pool_support(const cufinufft_opts &opts) {
  DeviceSwitcher switcher(opts.gpu_device_id);
  int supports_pools = 0;
  cudaDeviceGetAttribute(&supports_pools, cudaDevAttrMemoryPoolsSupported,
                         opts.gpu_device_id);
  static bool warned = false;
  if (!warned && !supports_pools && opts.gpu_stream != nullptr) {
    fprintf(stderr,
            "[cufinufft] Warning: cudaMallocAsync not supported on this device. Use of "
            "CUDA streams may not perform optimally.\n");
    warned = true;
  }
  return supports_pools;
}

template<typename T>
int cufinufft_plan_t<T>::setup_spreadinterp()
// Initializes spreader kernel params in this->spopts from this->tol,
// this->opts.upsampfac, and this->opts.gpu_kerevalmeth (0:exp(sqrt()),
// 1: Horner ppval). Mirrors CPU FINUFFT_PLAN_T<TF>::setup_spreadinterp().
// Returns 0, or FINUFFT_WARN_EPS_TOO_SMALL when tol was clamped up to
// eps_mach; throws on hard error.
// As of v2.5 no longer sets ES_c, ES_halfwidth, since absent from spopts.
// To do: *** update this to CPU v2.5 kernel choice, coeffs, params...
{
  using finufft::common::MAX_NSPREAD;
  using finufft::common::PI;
  T eps                      = tol;
  const T upsampfac          = T(opts.upsampfac);
  const int kerevalmeth      = opts.gpu_kerevalmeth;
  const int debug            = opts.debug;
  const int spreadinterponly = opts.gpu_spreadinterponly;

  if (upsampfac != 2.0 && upsampfac != 1.25) { // nonstandard sigma
    if (kerevalmeth == 1) {
      fprintf(
          stderr,
          "[%s] error: nonstandard upsampfac=%.3g cannot be handled by kerevalmeth=1\n",
          __func__, upsampfac);
      throw int(FINUFFT_ERR_HORNER_WRONG_BETA);
    }
    if (upsampfac <= 1.0) { // no digits would result, ns infinite
      fprintf(stderr, "[%s] error: upsampfac=%.3g\n", __func__, upsampfac);
      throw int(FINUFFT_ERR_UPSAMPFAC_TOO_SMALL);
    }
    // calling routine must abort on above errors, since spopts is garbage!
    if (!spreadinterponly && upsampfac > 4.0)
      fprintf(stderr, "[%s] warning: upsampfac=%.3g is too large to be beneficial!\n",
              __func__, upsampfac);
  }

  // defaults... (user can change after this function called)
  spopts.spread_direction = 0; // user should always set to 1 or 2 as desired
  spopts.upsampfac        = upsampfac;

  // as in FINUFFT v2.0, allow too-small-eps by truncating to eps_mach...
  int ier             = 0;
  constexpr T EPSILON = std::numeric_limits<T>::epsilon();
  if (eps < EPSILON) {
    fprintf(stderr, "[%s]: warning, increasing tol=%.3g to eps_mach=%.3g.\n", __func__,
            (double)eps, (double)EPSILON);
    eps = EPSILON;
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }

  // Set kernel width w (aka ns) and ES kernel beta parameter, in spopts...
  // To do: *** unify with new CPU kernel logic/coeffs of v2.5.
  int ns = std::ceil(-log10(eps / (T)10.0)); // 1 digit per power of ten
  if (upsampfac != 2.0)                      // override ns for custom sigma
    ns = std::ceil(-log(eps) / (T(PI) * sqrt(1 - 1 / upsampfac))); // formula,
                                                                   // gamma=1
  ns = std::max(2, ns);   // we don't have ns=1 version yet
  if (ns > MAX_NSPREAD) { // clip to match allocated arrays
    fprintf(stderr,
            "[%s] warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; "
            "clipping to max %d.\n",
            __func__, upsampfac, (double)eps, ns, MAX_NSPREAD);
    ns  = MAX_NSPREAD;
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }
  spopts.nspread = ns;

  T betaoverns = 2.30;            // gives decent betas for default sigma=2.0
  if (ns == 2) betaoverns = 2.20; // some small-width tweaks...
  if (ns == 3) betaoverns = 2.26;
  if (ns == 4) betaoverns = 2.38;
  if (upsampfac != 2.0) { // again, override beta for custom sigma
    T gamma    = 0.97;    // must match devel/gen_all_horner_C_code.m
    betaoverns = gamma * T(PI) * (1 - 1 / (2 * upsampfac));
  }
  spopts.beta = betaoverns * (T)ns; // set the kernel beta (shape) parameter
  if (debug)
    printf("[%s] (kerevalmeth=%d) eps=%.3g sigma=%.3g: chose ns=%d beta=%.3g\n", __func__,
           kerevalmeth, (double)eps, (double)upsampfac, ns, spopts.beta);
  return ier;
}
template int cufinufft_plan_t<float>::setup_spreadinterp();
template int cufinufft_plan_t<double>::setup_spreadinterp();

template<typename T>
std::tuple<CUFINUFFT_BIGINT, T, T> cufinufft_plan_t<T>::set_nhg_type3(T S, T X) const
// Mirror of CPU set_nhg_type3: choose nf, h, gam given source half-width S and
// freq half-width X, using this plan's opts/spopts.
{
  using finufft::common::PI;
  int nss = spopts.nspread + 1; // since ns may be odd
  T Xsafe = X, Ssafe = S;       // may be tweaked locally
  if (X == 0.0)                 // logic ensures XS>=1, handle X=0 a/o S=0
    if (S == 0.0) {
      Xsafe = 1.0;
      Ssafe = 1.0;
    } else
      Xsafe = std::max(Xsafe, T(1) / S);
  else
    Ssafe = std::max(Ssafe, T(1) / X);
  T nfd = 2.0 * opts.upsampfac * Ssafe * Xsafe / PI + nss;
  if (!std::isfinite(nfd)) nfd = 0.0;
  auto nf = (int)nfd;
  if (nf < 2 * spopts.nspread) nf = 2 * spopts.nspread;
  if (nf < MAX_NF) nf = cufinufft::utils::next235beven(nf, 1);
  auto h   = 2 * T(PI) / nf;
  auto gam = T(nf) / (2.0 * opts.upsampfac * Ssafe);
  return std::make_tuple(nf, h, gam);
}
template std::tuple<CUFINUFFT_BIGINT, float, float>
cufinufft_plan_t<float>::set_nhg_type3(float, float) const;
template std::tuple<CUFINUFFT_BIGINT, double, double>
cufinufft_plan_t<double>::set_nhg_type3(double, double) const;

template<typename T> void cufinufft_plan_t<T>::allocate_subprob_state() {
  cuda::std::array<int, 3> binsizes{opts.gpu_binsizex, opts.gpu_binsizey,
                                    opts.gpu_binsizez};

  switch (opts.gpu_method) {
  case 1: {
    if (opts.gpu_sort) {
      int numbins = 1;
      for (int idim = 0; idim < dim; ++idim)
        numbins *= ceil((T)nf123[idim] / binsizes[idim]);
      binsize.resize(numbins);
      binstartpts.resize(numbins);
    }
  } break;
  case 2:
  case 3: {
    int numbins = 1;
    for (int idim = 0; idim < dim; ++idim)
      numbins *= ceil((T)nf123[idim] / binsizes[idim]);
    numsubprob.resize(numbins);
    binsize.resize(numbins);
    binstartpts.resize(numbins);
    subprobstartpts.resize(numbins + 1);
  } break;
  case 4: {
    if (dim != 3) {
      std::cerr << "err: invalid method " << std::endl;
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
    cuda::std::array<int, 3> obinsizes{opts.gpu_obinsizex, opts.gpu_obinsizey,
                                       opts.gpu_obinsizez};
    int numobins_tot = 1, numbins_tot = 1;
    for (int idim = 0; idim < dim; ++idim) {
      const int numobins = (int)ceil((T)nf123[idim] / obinsizes[idim]);
      numobins_tot *= numobins;
      const int binsperobin = obinsizes[idim] / binsizes[idim];
      numbins_tot *= numobins * (binsperobin + 2);
    }

    numsubprob.resize(numobins_tot);
    binsize.resize(numbins_tot);
    binstartpts.resize(numbins_tot + 1);
    subprobstartpts.resize(numobins_tot + 1);
  } break;
  default:
    std::cerr << "[allocate] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
  if (!opts.gpu_spreadinterponly)
    for (int idim = 0; idim < dim; ++idim) fwkerhalf[idim].resize(nf123[idim] / 2 + 1);
}
template void cufinufft_plan_t<float>::allocate_subprob_state();
template void cufinufft_plan_t<double>::allocate_subprob_state();

template<typename T> void cufinufft_plan_t<T>::allocate_nupts() {
  size_t newsize_sortidx  = 0;
  size_t newsize_idxnupts = 0;

  switch (opts.gpu_method) {
  case 1: {
    if (opts.gpu_sort) newsize_sortidx = M;
    newsize_idxnupts = M;
  } break;
  case 2:
  case 3: {
    newsize_sortidx  = M;
    newsize_idxnupts = M;
  } break;
  case 4: {
    if (dim != 3) {
      std::cerr << "err: invalid method " << std::endl;
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
    newsize_sortidx = M;
  } break;
  default:
    std::cerr << "[allocate_nupts] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }

  if (newsize_sortidx != sortidx.size()) sortidx.resize(newsize_sortidx);
  if (newsize_idxnupts != idxnupts.size()) idxnupts.resize(newsize_idxnupts);
}
template void cufinufft_plan_t<float>::allocate_nupts();
template void cufinufft_plan_t<double>::allocate_nupts();

template<typename T>
cufinufft_plan_t<T>::cufinufft_plan_t(int type_, int dim_, const int *nmodes, int iflag_,
                                      int ntransf_, T tol_, const cufinufft_opts &opts_)
    : opts(opts_), supports_pools(have_pool_support(opts_)), tol(tol_), type(type_),
      dim(dim_), ntransf(ntransf_), iflag(iflag_ >= 0 ? 1 : -1) {
  /*
      "plan" stage (in single or double precision).
          See ../docs/cppdoc.md for main user-facing documentation.
          Note that *d_plan_ptr in the args list was called simply *plan there.
          This is the remaining dev-facing doc:

  This performs:
          (0) creating a new plan struct (d_plan), a pointer to which is passed
              back by writing that pointer into *d_plan_ptr.
          (1) set up the spread option, d_plan.spopts.
          (2) calculate the correction factor on cpu, copy the value from cpu to
              gpu
          (3) allocate gpu arrays with size determined by number of fourier modes
              and method related options that had been set in d_plan.opts
          (4) call cufftPlanMany and save the cufft plan inside cufinufft plan
          Variables and arrays inside the plan struct are set and allocated.

      Melody Shih 07/25/19. Use-facing moved to markdown, Barnett 2/16/21.
      Marco Barbone 07/26/24. Using SM when shared memory available is enough.
  */
  using namespace cufinufft::common;
  using namespace finufft::common;

  if (type < 1 || type > 3) {
    fprintf(stderr, "[%s] Invalid type (%d): should be 1, 2, or 3.\n", __func__, type);
    throw int(FINUFFT_ERR_TYPE_NOTVALID);
  }
  if (ntransf < 1) {
    fprintf(stderr, "[%s] Invalid ntransf (%d): should be at least 1.\n", __func__,
            ntransf);
    throw int(FINUFFT_ERR_NTRANS_NOTVALID);
  }

  // set nf1, nf2, nf3 to 1 for type 3, type 1, type 2 will overwrite this
  nf123                 = {1, 1, 1};
  opts.gpu_maxbatchsize = std::max(opts.gpu_maxbatchsize, 1);
  opts.gpu_np           = opts.gpu_method == 3 ? opts.gpu_np : 0;

  if (type != 3) {
    mstu = {nmodes[0], nmodes[1], nmodes[2]};
    if (opts.debug) {
      printf("[cufinufft] (ms,mt,mu): %d %d %d\n", mstu[0], mstu[1], mstu[2]);
    }
  } else { // type 3 turns its outer type 1 into spreading-only
    opts.gpu_spreadinterponly = 1;
  }

  batchsize = opts.gpu_maxbatchsize;
  // TODO: check if this is the right heuristic
  if (batchsize == 0)                 // implies: use a heuristic.
    batchsize = std::min(ntransf, 8); // heuristic from test codes

  stream = (cudaStream_t)opts.gpu_stream;

  // simple check to use upsampfac=1.25 if tol is big
  // FIXME: since cufft is really fast we should use 1.25 only if we run out of vram
  if (opts.upsampfac == 0.0) {         // indicates auto-choose
    opts.upsampfac = 2.0;              // default, and need for tol small
    if (tol >= (T)1E-9 && type == 3) { // the tol sigma=5/4 can reach
      opts.upsampfac = 1.25;
    }
    if (opts.debug) {
      printf("[cufinufft] upsampfac automatically set to %.3g\n", opts.upsampfac);
    }
  }
  /* Setup Spreader */
  eps_too_small = setup_spreadinterp() != 0;

  spopts.spread_direction = type;

  if (opts.debug) {
    // print the spreader options
    printf("[cufinufft] spreader options:\n");
    printf("[cufinufft] nspread: %d\n", spopts.nspread);
  }
  {
    /* Automatically set GPU method. */
    const bool auto_method = opts.gpu_method == 0;
    if (auto_method) {
      // Default to method 2 (SM) for type 1/3, otherwise method 1 (GM).
      opts.gpu_method = (type == 1 || type == 3) ? 2 : 1;
    }
    try {
      cufinufft_setup_binsize<T>(type, spopts.nspread, dim, &opts);
    } catch (const std::runtime_error &e) {
      if (auto_method) {
        // Auto-selection of SM failed, fall back to GM and try again.
        opts.gpu_method = 1;
        cufinufft_setup_binsize<T>(type, spopts.nspread, dim, &opts);
      } else {
        // User-specified method failed, or the fallback GM method failed.
        fprintf(stderr, "%s, method %d\n", e.what(), opts.gpu_method);
        throw int(FINUFFT_ERR_INSUFFICIENT_SHMEM);
      }
    }
    THROW_IF_CUDA_ERROR
  }
  // Bin size and memory info now printed in cufinufft_setup_binsize() (common.cu)
  // Additional runtime info at debug level 2
  if (opts.debug >= 2) {
    printf("[cufinufft] Runtime: grid=(%d,%d,%d), M=%d\n", nf123[0], nf123[1], nf123[2],
           M);
  }

  // dynamically request the maximum amount of shared memory available
  // for the spreader

  if (type == 1 || type == 2) {
    if (opts.gpu_spreadinterponly) {
      // spread/interp grid is precisely the user "mode" sizes, no upsampling
      for (int idim = 0; idim < dim; ++idim) nf123[idim] = mstu[idim];
      if (opts.debug) {
        printf("[cufinufft] spreadinterponly mode: (nf1,nf2,nf3) = (%d, %d, %d)\n",
               nf123[0], nf123[1], nf123[2]);
      }
    } else { // usual NUFFT with fine grid using upsampling
      std::array<int, 3> obinsize{opts.gpu_obinsizex, opts.gpu_obinsizey,
                                  opts.gpu_obinsizez};
      for (int idim = 0; idim < dim; ++idim)
        set_nf_type12(mstu[idim], &nf123[idim], obinsize[idim]);
      if (opts.debug)
        printf("[cufinufft] (nf1,nf2,nf3) = (%d, %d, %d)\n", nf123[0], nf123[1],
               nf123[2]);
    }
    nf = nf123[0] * nf123[1] * nf123[2];

    allocate_subprob_state();

    // We don't need any cuFFT plans or kernel values if we are only spreading /
    // interpolating
    if (!opts.gpu_spreadinterponly) {
      int n[3];
      int ntot = 1;
      for (int idim = 0; idim < dim; ++idim) {
        n[idim] = int(nf123[dim - idim - 1]);
        ntot *= n[idim];
      }
      fftplan.set_device_id(opts.gpu_device_id);
      cufftResult_t cufft_status =
          cufftPlanMany(fftplan.for_creation(), dim, n, n, 1, ntot, n, 1, ntot,
                        cufft_type<T>(), batchsize);

      if (cufft_status != CUFFT_SUCCESS) {
        fprintf(stderr, "[%s] cufft makeplan error: %s", __func__,
                cufftGetErrorString(cufft_status));
        throw int(FINUFFT_ERR_CUDA_FAILURE);
      }
      cufftSetStream(fftplan.get(), stream);

      // compute up to 3 * NQUAD precomputed values on CPU
      T fseries_precomp_phase[3 * MAX_NQUAD];
      T fseries_precomp_f[3 * MAX_NQUAD];
      thrust::device_vector<T> d_fseries_precomp_phase(3 * MAX_NQUAD);
      thrust::device_vector<T> d_fseries_precomp_f(3 * MAX_NQUAD);
      for (int idim = 0; idim < dim; ++idim)
        precompute_fseries_nodes(nf123[idim], fseries_precomp_f + idim * MAX_NQUAD,
                                 fseries_precomp_phase + idim * MAX_NQUAD);
      // copy the precomputed data to the device using thrust
      thrust::copy(fseries_precomp_phase, fseries_precomp_phase + 3 * MAX_NQUAD,
                   d_fseries_precomp_phase.begin());
      thrust::copy(fseries_precomp_f, fseries_precomp_f + 3 * MAX_NQUAD,
                   d_fseries_precomp_f.begin());
      // the full fseries is done on the GPU here
      fseries_kernel_compute(dim, nf123, d_fseries_precomp_f.data().get(),
                             d_fseries_precomp_phase.data().get(), fwkerhalf,
                             spopts.nspread, stream);
    }
  }
}

template cufinufft_plan_t<float>::cufinufft_plan_t(int type_, int dim_, const int *nmodes,
                                                   int iflag_, int ntransf_, float tol_,
                                                   const cufinufft_opts &opts_);
template cufinufft_plan_t<double>::cufinufft_plan_t(
    int type_, int dim_, const int *nmodes, int iflag_, int ntransf_, double tol_,
    const cufinufft_opts &opts_);
