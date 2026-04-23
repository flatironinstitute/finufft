#include <iostream>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

#include <cufinufft.h>
#include <cufinufft/common.h>
#include <cufinufft/defs.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

#include <finufft_common/constants.h>
#include <finufft_errors.h>
#include <thrust/device_vector.h>

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
static int setup_spreader(finufft_spread_opts &spopts, T eps, T upsampfac,
                          int kerevalmeth, int debug, int spreadinterponly)
// Initializes spreader kernel parameters given desired NUFFT tolerance eps,
// upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), and ker eval meth
// (etiher 0:exp(sqrt()), 1: Horner ppval).
// Also sets all default options in finufft_spread_opts.
// Must call before any kernel evals done.
// Returns: 0 success, 1 warning, >1 failure (see error codes in utils.h).
// As of v2.5 no longer sets ES_c, ES_halfwidth, since absent from spopts.
// To do: *** update this to CPU v2.5 kernel choice, coeffs, params...
{
  using finufft::common::MAX_NSPREAD;
  using finufft::common::PI;

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

template<typename T> void cufinufft_plan_t<T>::allocate() {
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

/* Kernel for copying fw to fk with amplication by prefac/ker */
// Note: assume modeord=0: CMCL-compatible mode ordering in fk (from -N/2 up
// to N/2-1), modeord=1: FFT-compatible mode ordering in fk (from 0 to N/2-1, then -N/2 up
// to -1).
template<typename T, int modeord, int ndim>
static __global__ void deconv_nd(
    cuda::std::array<int, 3> mstu, cuda::std::array<int, 3> nf123, cuda_complex<T> *fw,
    cuda_complex<T> *fk, cuda::std::array<const T *, 3> fwkerhalf, bool fw2fk) {

  cuda::std::array<int, 3> m_acc{1, mstu[0], mstu[0] * mstu[1]};
  int mtotal = m_acc[ndim - 1] * mstu[ndim - 1];
  cuda::std::array<int, 3> nf_acc{1, nf123[0], nf123[0] * nf123[1]};

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < mtotal;
       i += blockDim.x * gridDim.x) {
    cuda::std::array<int, 3> k;
    if constexpr (ndim == 1) k = {i, 0, 0};
    if constexpr (ndim == 2) k = {i % mstu[0], i / mstu[0], 0};
    if constexpr (ndim == 3)
      k = {i % mstu[0], (i / mstu[0]) % mstu[1], (i / mstu[0]) / mstu[1]};
    T kervalue = 1;
    int fkidx = 0, fwidx = 0;
    for (int idim = 0; idim < ndim; ++idim) {
      int wn, fwkerindn;
      if constexpr (modeord == 0) {
        int pivot = k[idim] - mstu[idim] / 2;
        wn        = (pivot >= 0) ? pivot : nf123[idim] + pivot;
        fwkerindn = abs(pivot);
      } else {
        int pivot = k[idim] - mstu[idim] + mstu[idim] / 2;
        wn        = (pivot >= 0) ? nf123[idim] + k[idim] - mstu[idim] : k[idim];
        fwkerindn = (pivot >= 0) ? mstu[idim] - k[idim] : k[idim];
      }
      kervalue *= fwkerhalf[idim][fwkerindn];
      fwidx += wn * nf_acc[idim];
      fkidx += k[idim] * m_acc[idim];
    }

    if (fw2fk) {
      fk[fkidx] = fw[fwidx] / kervalue;
    } else {
      fw[fwidx] = fk[fkidx] / kervalue;
    }
  }
}

template<typename T>
template<int modeord, int ndim>
void cufinufft_plan_t<T>::deconvolve_nd<modeord, ndim>(
    cuda_complex<T> *fw, cuda_complex<T> *fk, int blksize) const
/*
    wrapper for deconvolution & amplification in 1/2/3D.

    Melody Shih 11/21/21
*/
{
  int nmodes = 1, nftot = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    nmodes *= mstu[idim];
    nftot *= nf123[idim];
  }

  bool fw2fk = spopts.spread_direction == 1;
  if (!fw2fk)
    checkCudaErrors(
        cudaMemsetAsync(fw, 0, batchsize * nftot * sizeof(cuda_complex<T>), stream));

  for (int t = 0; t < blksize; t++)
    deconv_nd<T, modeord, ndim><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
        mstu, nf123, fw + t * nftot, fk + t * nmodes, dethrust(fwkerhalf), fw2fk);
}

template<typename T>
void cufinufft_plan_t<T>::deconvolve(cuda_complex<T> *fw, cuda_complex<T> *fk,
                                     int blksize) const {
  if (dim == 1)
    (opts.modeord == 0) ? deconvolve_nd<0, 1>(fw, fk, blksize)
                        : deconvolve_nd<1, 1>(fw, fk, blksize);
  if (dim == 2)
    (opts.modeord == 0) ? deconvolve_nd<0, 2>(fw, fk, blksize)
                        : deconvolve_nd<1, 2>(fw, fk, blksize);
  if (dim == 3)
    (opts.modeord == 0) ? deconvolve_nd<0, 3>(fw, fk, blksize)
                        : deconvolve_nd<1, 3>(fw, fk, blksize);
}

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
  eps_too_small = setup_spreader(spopts, tol, T(opts.upsampfac), opts.gpu_kerevalmeth,
                                 opts.debug, opts.gpu_spreadinterponly) != 0;

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
        set_nf_type12(mstu[idim], opts, spopts, &nf123[idim], obinsize[idim]);
      if (opts.debug)
        printf("[cufinufft] (nf1,nf2,nf3) = (%d, %d, %d)\n", nf123[0], nf123[1],
               nf123[2]);
    }
    nf = nf123[0] * nf123[1] * nf123[2];

    allocate();

    // We don't need any cuFFT plans or kernel values if we are only spreading /
    // interpolating
    if (!opts.gpu_spreadinterponly) {
      int n[3];
      int ntot = 1;
      for (int idim = 0; idim < dim; ++idim) {
        n[idim] = int(nf123[dim - idim - 1]);
        ntot *= n[idim];
      }
      cufftResult_t cufft_status = cufftPlanMany(&fftplan, dim, n, n, 1, ntot, n, 1, ntot,
                                                 cufft_type<T>(), batchsize);

      if (cufft_status != CUFFT_SUCCESS) {
        fprintf(stderr, "[%s] cufft makeplan error: %s", __func__,
                cufftGetErrorString(cufft_status));
        throw int(FINUFFT_ERR_CUDA_FAILURE);
      }
      cufftSetStream(fftplan, stream);

      // compute up to 3 * NQUAD precomputed values on CPU
      T fseries_precomp_phase[3 * MAX_NQUAD];
      T fseries_precomp_f[3 * MAX_NQUAD];
      thrust::device_vector<T> d_fseries_precomp_phase(3 * MAX_NQUAD);
      thrust::device_vector<T> d_fseries_precomp_f(3 * MAX_NQUAD);
      for (int idim = 0; idim < dim; ++idim)
        onedim_fseries_kernel_precomp<T>(
            nf123[idim], fseries_precomp_f + idim * MAX_NQUAD,
            fseries_precomp_phase + idim * MAX_NQUAD, spopts);
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

template<typename T>
void cufinufft_plan_t<T>::setpts_12(int M_, const T *d_kx, const T *d_ky, const T *d_kz)
/*
    "setNUpts" stage (in single or double precision).

    In this stage, we
        (1) set the number and locations of nonuniform points
        (2) allocate gpu arrays with size determined by number of nupts
        (3) rescale x,y,z coordinates for spread/interp (on gpu, rescaled
            coordinates are stored)
        (4) determine the spread/interp properties that only relates to the
            locations of nupts (see 2d/spread2d_wrapper.cu,
            3d/spread3d_wrapper.cu for what have been done in
            function spread<dim>d_<method>_prop() )

        See ../docs/cppdoc.md for main user-facing documentation.
        Here is the old developer docs, which are useful only to translate
        the argument names from the user-facing ones:

    Input:
    M                 number of nonuniform points
    d_kx, d_ky, d_kz  gpu array of x,y,z locations of sources (each a size M
                      T array) in [-pi, pi). set h_kz to "NULL" if dimension
                      is less than 3. same for h_ky for dimension 1.
    N, d_s, d_t, d_u  not used for type1, type2. set to 0 and NULL.

    Input/Output:
    d_plan            pointer to a CUFINUFFT_PLAN_S. Variables and arrays inside
                      the plan are set and allocated.

        Returned value:
        a status flag: 0 if success, otherwise an error occurred

Notes: the type T means either single or double, matching the
    precision of the library version called.

    Melody Shih 07/25/19; Barnett 2/16/21 moved out docs.
*/
{
  M = M_;

  allocate_nupts();

  kxyz = {d_kx, d_ky, d_kz};

  cufinufft::spreadinterp::cuspreadnd_prop(*this);

  if (opts.debug) {
    printf("[cufinufft] plan->M=%d\n", M);
  }
}

template<typename T>
void cufinufft_plan_t<T>::setpts(int M_, const T *d_kx, const T *d_ky, const T *d_kz,
                                 int N_, const T *d_s, const T *d_t, const T *d_u) {
  DeviceSwitcher switcher(opts.gpu_device_id);
  // type 1 and type 2 setpts
  if (type == 1 || type == 2) {
    return setpts_12(M_, d_kx, d_ky, d_kz);
  }
  // type 3 setpts

  // This code follows the same implementation of the CPU code in finufft and uses similar
  // variables names where possible. However, the use of GPU routines and paradigms make
  // it harder to follow. To understand the code, it is recommended to read the CPU code
  // first.

  cuda::std::array<const T *, 3> d_kxyz = {d_kx, d_ky, d_kz};
  cuda::std::array<const T *, 3> d_stu  = {d_s, d_t, d_u};

  M = M_;
  N = N_;
  if (N < 0) {
    fprintf(stderr, "[cufinufft] Invalid N (%d): cannot be negative.\n", N);
    throw int(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  }
  if (N > MAX_NF) {
    fprintf(stderr, "[cufinufft] Invalid N (%d): cannot be greater than %d.\n", N,
            MAX_NF);
    throw int(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  }
  if (dim > 0 && d_s == nullptr) {
    fprintf(stderr, "[%s] Error: d_s is nullptr but dim > 0.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }

  if (dim > 1 && d_t == nullptr) {
    fprintf(stderr, "[%s] Error: d_t is nullptr but dim > 1.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }

  if (dim > 2 && d_u == nullptr) {
    fprintf(stderr, "[%s] Error: d_u is nullptr but dim > 2.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }

  using namespace cufinufft::utils;
  for (int idim = 0; idim < dim; ++idim) {
    const auto [xx, cc]           = arraywidcen<T>(M, d_kxyz[idim], stream);
    type3_params.X[idim]          = xx;
    type3_params.C[idim]          = cc;
    const auto [SS, DD]           = arraywidcen<T>(N, d_stu[idim], stream);
    const auto [nfnf, hh, gamgam] = set_nhg_type3<T>(SS, xx, opts, spopts);
    nf123[idim]                   = nfnf;
    type3_params.S[idim]          = SS;
    type3_params.D[idim]          = DD;
    type3_params.h[idim]          = hh;
    type3_params.gam[idim]        = gamgam;
  }
  if (opts.debug) {
    printf("[%s]", __func__);
    printf("\tM=%d N=%d\n", M, N);
    printf("\tX1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%d h1=%.3g\t\n",
           type3_params.X[0], type3_params.C[0], type3_params.S[0], type3_params.D[0],
           type3_params.gam[0], nf123[0], type3_params.h[0]);
    if (dim > 1) {
      printf("\tX2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%d h2=%.3g\n",
             type3_params.X[1], type3_params.C[1], type3_params.S[1], type3_params.D[1],
             type3_params.gam[1], nf123[1], type3_params.h[1]);
    }
    if (dim > 2) {
      printf("\tX3=%.3g C3=%.3g S3=%.3g D3=%.3g gam3=%g nf3=%d h3=%.3g\n",
             type3_params.X[2], type3_params.C[2], type3_params.S[2], type3_params.D[2],
             type3_params.gam[2], nf123[2], type3_params.h[2]);
    }
  }
  nf = nf123[0] * nf123[1] * nf123[2];

  // FIXME: MAX_NF might be too small...
  if (nf * opts.gpu_maxbatchsize > MAX_NF) {
    fprintf(stderr,
            "[%s t3] fwBatch would be bigger than MAX_NF, not attempting malloc!\n",
            __func__);
    throw int(FINUFFT_ERR_MAXNALLOC);
  }

  for (int idim = 0; idim < dim; ++idim) {
    kxyzp[idim].resize(M);
    kxyz[idim] = dethrust(kxyzp[idim]);
    STUp[idim].resize(N);
    STU[idim] = dethrust(STUp[idim]);
  }
  prephase.resize(M);
  deconv.resize(N);

  // NOTE: init-captures are not allowed for extended __host__ __device__ lambdas

  for (int idim = 0; idim < dim; ++idim) {
    const auto ig = T(1) / type3_params.gam[idim];
    const auto C  = -type3_params.C[idim];
    thrust::transform(thrust::cuda::par.on(stream), d_kxyz[idim], d_kxyz[idim] + M,
                      dethrust(kxyzp[idim]), [ig, C] __host__ __device__(const T x) -> T {
                        return (x + C) * ig;
                      });
  }
  if (type3_params.D[0] != 0 || type3_params.D[1] != 0 || type3_params.D[2] != 0) {
    // if ky is null, use kx for ky and kz
    // this is not the most efficient implementation, but it is the most compact
    const auto iterator =
        thrust::make_zip_iterator(thrust::make_tuple(d_kx,
                                                     // to avoid out of bounds access, use
                                                     // kx if ky is null
                                                     (dim > 1) ? d_ky : d_kx,
                                                     // same idea as above
                                                     (dim > 2) ? d_kz : d_kx));
    const auto D        = type3_params.D;
    const auto realsign = iflag >= 0 ? T(1) : T(-1);
    thrust::transform(
        thrust::cuda::par.on(stream), iterator, iterator + M, dethrust(prephase),
        [D, realsign] __host__
        __device__(const thrust::tuple<T, T, T> &tuple) -> cuda_complex<T> {
          const auto x = thrust::get<0>(tuple);
          const auto y = thrust::get<1>(tuple);
          const auto z = thrust::get<2>(tuple);
          // no branching because D[1] and D[2] are 0 if dim < 2 and dim < 3
          // this is generally faster on GPU
          const auto phase = D[0] * x + D[1] * y + D[2] * z;
          // TODO: nvcc should have the sincos function
          //       check the cos + i*sin
          //       ref: https://en.wikipedia.org/wiki/Cis_(mathematics)
          return cuda_complex<T>{std::cos(phase), std::sin(phase) * realsign};
        });
  } else {
    thrust::fill(thrust::cuda::par.on(stream), dethrust(prephase), dethrust(prephase) + M,
                 cuda_complex<T>{1, 0});
  }

  for (int idim = 0; idim < dim; ++idim) {
    const auto scale = type3_params.h[idim] * type3_params.gam[idim];
    const auto D     = -type3_params.D[idim];
    thrust::transform(
        thrust::cuda::par.on(stream), d_stu[idim], d_stu[idim] + N, dethrust(STUp[idim]),
        [scale, D] __host__ __device__(const T s) -> T { return scale * (s + D); });
  }
  { // here we declare phi_hat1, phi_hat2, and phi_hat3
    // and the precomputed data for the fseries kernel
    using namespace cufinufft::common;

    std::array<T, 3 * MAX_NQUAD> nuft_precomp_z{};
    std::array<T, 3 * MAX_NQUAD> nuft_precomp_f{};
    thrust::device_vector<T> d_nuft_precomp_z(3 * MAX_NQUAD);
    thrust::device_vector<T> d_nuft_precomp_f(3 * MAX_NQUAD);
    cuda::std::array<gpu_array<T>, 3> phi_hat123(
        {gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}});
    for (int idim = 0; idim < dim; ++idim) phi_hat123[idim].resize(N);
    for (int idim = 0; idim < dim; ++idim)
      onedim_nuft_kernel_precomp<T>(nuft_precomp_f.data() + idim * MAX_NQUAD,
                                    nuft_precomp_z.data() + idim * MAX_NQUAD, spopts);
    // copy the precomputed data to the device using thrust
    thrust::copy(nuft_precomp_z.begin(), nuft_precomp_z.end(), d_nuft_precomp_z.begin());
    thrust::copy(nuft_precomp_f.begin(), nuft_precomp_f.end(), d_nuft_precomp_f.begin());
    // sync the stream before calling the kernel might be needed
    nuft_kernel_compute(dim, {N, N, N}, d_nuft_precomp_f.data().get(),
                        d_nuft_precomp_z.data().get(), STU, phi_hat123, spopts.nspread,
                        stream);

    const auto is_c_finite = std::isfinite(type3_params.C[0]) &&
                             std::isfinite(type3_params.C[1]) &&
                             std::isfinite(type3_params.C[2]);
    const auto is_c_nonzero =
        type3_params.C[0] != 0 || type3_params.C[1] != 0 || type3_params.C[2] != 0;

    const auto phi_hat_iterator = thrust::make_zip_iterator(
        thrust::make_tuple(phi_hat123[0].data(),
                           // to avoid out of bounds access, use phi_hat1 if dim < 2
                           dim > 1 ? phi_hat123[1].data() : phi_hat123[0].data(),
                           // to avoid out of bounds access, use phi_hat1 if dim < 3
                           dim > 2 ? phi_hat123[2].data() : phi_hat123[0].data()));
    auto xdim = dim;
    thrust::transform(thrust::cuda::par.on(stream), phi_hat_iterator,
                      phi_hat_iterator + N, dethrust(deconv),
                      [xdim] __host__
                      __device__(const thrust::tuple<T, T, T> tuple) -> cuda_complex<T> {
                        auto phiHat = thrust::get<0>(tuple);
                        // in case dim < 2 or dim < 3, multiply by 1
                        phiHat *= (xdim > 1) ? thrust::get<1>(tuple) : T(1);
                        phiHat *= (xdim > 2) ? thrust::get<2>(tuple) : T(1);
                        return {T(1) / phiHat, T(0)};
                      });

    if (is_c_finite && is_c_nonzero) {
      const auto c1       = type3_params.C[0];
      const auto c2       = type3_params.C[1];
      const auto c3       = type3_params.C[2];
      const auto d1       = -type3_params.D[0];
      const auto d2       = -type3_params.D[1];
      const auto d3       = -type3_params.D[2];
      const auto realsign = iflag >= 0 ? T(1) : T(-1);
      // passing d_s three times if dim == 1 because d_t and d_u are not allocated
      // passing d_s and d_t if dim == 2 because d_u is not allocated
      const auto phase_iterator = thrust::make_zip_iterator(
          thrust::make_tuple(d_s, dim > 1 ? d_t : d_s, dim > 2 ? d_u : d_s));
      thrust::transform(
          thrust::cuda::par.on(stream), phase_iterator, phase_iterator + N,
          dethrust(deconv), dethrust(deconv),
          [c1, c2, c3, d1, d2, d3, realsign] __host__
          __device__(const thrust::tuple<T, T, T> tuple,
                     cuda_complex<T> deconv) -> cuda_complex<T> {
            // d2 and d3 are 0 if dim < 2 and dim < 3
            const auto phase = c1 * (thrust::get<0>(tuple) + d1) +
                               c2 * (thrust::get<1>(tuple) + d2) +
                               c3 * (thrust::get<2>(tuple) + d3);
            return cuda_complex<T>{std::cos(phase), realsign * std::sin(phase)} * deconv;
          });
    }
    // exiting the block frees the memory allocated for phi_hat1, phi_hat2, and phi_hat3
    // and the precomputed data for the fseries kernel
    // since GPU memory is expensive, we should free it as soon as possible
  }

  allocate();

  setpts_12(M, kxyz[0], kxyz[1], kxyz[2]);
  {
    int t2modes[]               = {nf123[0], nf123[1], nf123[2]};
    cufinufft_opts t2opts       = opts;
    t2opts.gpu_spreadinterponly = 0;
    t2opts.gpu_method           = 0;
    // Release the old inner plan before allocating the new one to
    // avoid holding both in memory at the same time, which could be wasteful.
    t2_plan.reset();
    t2_plan = std::make_unique<cufinufft_plan_t<T>>(2, dim, t2modes, iflag, batchsize,
                                                    tol, t2opts);
    t2_plan->setpts_12(N, STU[0], STU[1], STU[2]);
    if (t2_plan->spopts.spread_direction != 2) {
      fprintf(stderr, "[%s] inner t2 plan cufinufft_setpts_12 wrong direction\n",
              __func__);
    }
  }
}
template void cufinufft_plan_t<float>::setpts(
    int M_, const float *d_kx, const float *d_ky, const float *d_kz, int N_,
    const float *d_s, const float *d_t, const float *d_u);
template void cufinufft_plan_t<double>::setpts(
    int M_, const double *d_kx, const double *d_ky, const double *d_kz, int N_,
    const double *d_s, const double *d_t, const double *d_u);

template<typename T>
void cufinufft_plan_t<T>::exec1(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const
/*
    1D/2D/3D Type-1 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: spread data to oversampled regular mesh using kernel
        Step 2: compute FFT on uniform mesh
        Step 3: deconvolve by division of each Fourier mode independently by the
                Fourier series coefficient of the kernel.

    Melody Shih 07/25/19
*/
{
  assert(spopts.spread_direction == 1);

  int nmodes = 1;
  for (int idim = 0; idim < dim; ++idim) nmodes *= mstu[idim];
  // We don't need this buffer if we are just spreading; so we set
  // its size to 0 in that case.
  gpu_array<cuda_complex<T>> fwp(opts.gpu_spreadinterponly ? 0 : nf * batchsize, alloc);
  auto *fw = dethrust(fwp);
  for (int i = 0; i * batchsize < ntransf; i++) {
    int blksize   = std::min(ntransf - i * batchsize, batchsize);
    const auto *c = d_c + i * batchsize * M;
    auto *fk      = d_fk + i * batchsize * nmodes; // so deconvolve will write into
                                                   // user output f
    if (opts.gpu_spreadinterponly)
      fw = fk;                                     // spread directly into the appropriate
                                                   // section of the user output f

    checkCudaErrors(
        cudaMemsetAsync(fw, 0, blksize * nf * sizeof(cuda_complex<T>), stream));

    // Step 1: Spread
    cufinufft::spreadinterp::cuspreadnd<T>(*this, c, fw, blksize);

    if (opts.gpu_spreadinterponly) continue; // skip steps 2 and 3

    // Step 2: FFT
    cufftResult cufft_status = cufft_ex(fftplan, fw, fw, iflag);
    if (cufft_status != CUFFT_SUCCESS) throw int(FINUFFT_ERR_CUDA_FAILURE);

    // Step 3: deconvolve and shuffle
    deconvolve(fw, fk, blksize);
  }
}

template<typename T>
void cufinufft_plan_t<T>::exec2(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                                int ntransf_override) const
/*
    1D/2D/3D Type-2 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel
                Fourier coeff
        Step 2: compute FFT on uniform mesh
        Step 3: interpolate data to regular mesh

    Melody Shih 07/25/19
*/
{
  assert(spopts.spread_direction == 2);
  // CAUTION: if this particular exec2() call is executed as part of
  // a type 3 transform, ntransf will be overridden!
  int ntransf_for_this_run = ntransf;
  if (ntransf_override > 0) ntransf_for_this_run = ntransf_override;

  int nmodes = 1;
  for (int idim = 0; idim < dim; ++idim) nmodes *= mstu[idim];
  // We don't need this buffer if we are just interpolating; so we set
  // its size to 0 in that case.
  gpu_array<cuda_complex<T>> fwp(opts.gpu_spreadinterponly ? 0 : nf * batchsize, alloc);
  auto *fw = dethrust(fwp);
  for (int i = 0; i * batchsize < ntransf_for_this_run; i++) {
    int blksize = std::min(ntransf_for_this_run - i * batchsize, batchsize);
    auto *c     = d_c + i * batchsize * M;
    auto *fk    = d_fk + i * batchsize * nmodes;

    // Skip steps 1 and 2 if interponly
    if (!opts.gpu_spreadinterponly) {
      // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
      deconvolve(fw, fk, blksize);

      // Step 2: FFT
      THROW_IF_CUDA_ERROR
      cufftResult cufft_status = cufft_ex(fftplan, fw, fw, iflag);
      if (cufft_status != CUFFT_SUCCESS) throw int(FINUFFT_ERR_CUDA_FAILURE);
    } else
      fw = fk; // interpolate directly from user input f

    // Step 3: Interpolate
    cufinufft::spreadinterp::cuinterpnd<T>(*this, c, fw, blksize);
  }
}

// TODO: in case data is centered, we could save GPU memory
template<typename T>
void cufinufft_plan_t<T>::exec3(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const {
  /*
    1D/2D/3D Type-3 NUFFT

  This function is called in "exec" stage (See ../cufinufft.cu).
  It includes (copied from doc in finufft library)
    Step 0: pre-phase the input strengths
    Step 1: spread data
    Step 2: Type 2 NUFFT
    Step 3: deconvolve (amplify) each Fourier mode, using kernel Fourier coeff

  Marco Barbone 08/14/2024
  */
  gpu_array<cuda_complex<T>> CpBatch(M * batchsize, alloc);
  gpu_array<cuda_complex<T>> fwp(nf * batchsize, alloc);
  auto *fw = dethrust(fwp);
  for (int i = 0; i * batchsize < ntransf; i++) {
    int blksize                = std::min(ntransf - i * batchsize, batchsize);
    cuda_complex<T> *d_cstart  = d_c + i * batchsize * M;
    cuda_complex<T> *d_fkstart = d_fk + i * batchsize * N;
    // setting input for spreader
    auto *c = dethrust(CpBatch);
    // setting output for spreader
    auto *fk = fw;
    // NOTE: fw might need to be set to 0
    checkCudaErrors(
        cudaMemsetAsync(fw, 0, blksize * nf * sizeof(cuda_complex<T>), stream));
    // Step 0: pre-phase the input strengths
    for (int block = 0; block < blksize; block++) {
      thrust::transform(thrust::cuda::par.on(stream), dethrust(prephase),
                        dethrust(prephase) + M, d_cstart + block * M, c + block * M,
                        thrust::multiplies<cuda_complex<T>>());
    }
    // Step 1: Spread
    cufinufft::spreadinterp::cuspreadnd<T>(*this, c, fk, blksize);
    // now fk = fw contains the spread values
    // Step 2: Type 2 NUFFT
    // type 2 goes from fk to c
    // saving the results directly in the user output array d_fk
    // it needs to do blksize transforms
    t2_plan->exec2(d_fkstart, fw, blksize);
    // Step 3: deconvolve
    // now we need to d_fk = d_fk*deconv
    for (int j = 0; j < blksize; j++) {
      thrust::transform(thrust::cuda::par.on(stream), dethrust(deconv),
                        dethrust(deconv) + N, d_fkstart + j * N, d_fkstart + j * N,
                        thrust::multiplies<cuda_complex<T>>());
    }
  }
}

template<typename T>
void cufinufft_plan_t<T>::exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const {
  DeviceSwitcher switcher(opts.gpu_device_id);
  switch (type) {
  case 1:
    return exec1(d_c, d_fk);
  case 2:
    return exec2(d_c, d_fk);
  case 3:
    return exec3(d_c, d_fk);
  }
}
template void cufinufft_plan_t<float>::exec(cuda_complex<float> *d_c,
                                            cuda_complex<float> *d_fk) const;
template void cufinufft_plan_t<double>::exec(cuda_complex<double> *d_c,
                                             cuda_complex<double> *d_fk) const;
