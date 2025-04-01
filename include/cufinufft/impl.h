#ifndef CUFINUFFT_IMPL_H
#define CUFINUFFT_IMPL_H

#include <iostream>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

#include <cufinufft/common.h>
#include <cufinufft/defs.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

#include <finufft_errors.h>
#include <thrust/device_vector.h>

// 1d
template<typename T>
int cufinufft1d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
template<typename T>
int cufinufft1d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
template<typename T>
int cufinufft1d3_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
// 2d
template<typename T>
int cufinufft2d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
template<typename T>
int cufinufft2d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);

template<typename T>
int cufinufft2d3_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
// 3d
template<typename T>
int cufinufft3d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
template<typename T>
int cufinufft3d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
template<typename T>
int cufinufft3d3_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);

template<typename T>
int cufinufft_makeplan_impl(int type, int dim, int *nmodes, int iflag, int ntransf, T tol,
                            cufinufft_plan_t<T> **d_plan_ptr, cufinufft_opts *opts) {
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
  int ier;
  if (type < 1 || type > 3) {
    fprintf(stderr, "[%s] Invalid type (%d): should be 1, 2, or 3.\n", __func__, type);
    return FINUFFT_ERR_TYPE_NOTVALID;
  }
  if (ntransf < 1) {
    fprintf(stderr, "[%s] Invalid ntransf (%d): should be at least 1.\n", __func__,
            ntransf);
    return FINUFFT_ERR_NTRANS_NOTVALID;
  }

  /* allocate the plan structure, assign address to user pointer. */
  auto *d_plan = new cufinufft_plan_t<T>;
  memset(d_plan, 0, sizeof(*d_plan));
  *d_plan_ptr = d_plan;

  // Zero out your struct, (sets all pointers to NULL)
  // set nf1, nf2, nf3 to 1 for type 3, type 1, type 2 will overwrite this
  d_plan->nf1 = 1;
  d_plan->nf2 = 1;
  d_plan->nf3 = 1;
  d_plan->tol = tol;
  /* If a user has not supplied their own options, assign defaults for them. */
  if (opts == nullptr) {  // use default opts
    cufinufft_default_opts(&(d_plan->opts));
  } else {                // or read from what's passed in
    d_plan->opts = *opts; // keep a deep copy; changing *opts now has no effect
  }
  d_plan->dim                   = dim;
  d_plan->opts.gpu_maxbatchsize = std::max(d_plan->opts.gpu_maxbatchsize, 1);

  if (type != 3) {
    d_plan->ms = nmodes[0];
    d_plan->mt = nmodes[1];
    d_plan->mu = nmodes[2];
    if (d_plan->opts.debug) {
      printf("[cufinufft] (ms,mt,mu): %d %d %d\n", d_plan->ms, d_plan->mt, d_plan->mu);
    }
  } else { // type 3 turns its outer type 1 into spreading-only
    d_plan->opts.gpu_spreadinterponly = 1;
  }

  int fftsign     = (iflag >= 0) ? 1 : -1;
  d_plan->iflag   = fftsign;
  d_plan->ntransf = ntransf;

  int batchsize = (opts != nullptr) ? opts->gpu_maxbatchsize : 0;
  // TODO: check if this is the right heuristic
  if (batchsize == 0)                 // implies: use a heuristic.
    batchsize = std::min(ntransf, 8); // heuristic from test codes
  d_plan->batchsize = batchsize;

  const auto stream = d_plan->stream = (cudaStream_t)d_plan->opts.gpu_stream;

  // Mult-GPU support: set the CUDA Device ID:
  const int device_id = d_plan->opts.gpu_device_id;
  const cufinufft::utils::WithCudaDevice FromID{device_id};

  // cudaMallocAsync isn't supported for all devices, regardless of cuda version. Check
  // for support
  {
    cudaDeviceGetAttribute(&d_plan->supports_pools, cudaDevAttrMemoryPoolsSupported,
                           device_id);
    static bool warned = false;
    if (!warned && !d_plan->supports_pools && d_plan->opts.gpu_stream != nullptr) {
      fprintf(stderr,
              "[cufinufft] Warning: cudaMallocAsync not supported on this device. Use of "
              "CUDA streams may not perform optimally.\n");
      warned = true;
    }
  }

  // simple check to use upsampfac=1.25 if tol is big
  // FIXME: since cufft is really fast we should use 1.25 only if we run out of vram
  if (d_plan->opts.upsampfac == 0.0) { // indicates auto-choose
    d_plan->opts.upsampfac = 2.0;      // default, and need for tol small
    if (tol >= (T)1E-9 && type == 3) { // the tol sigma=5/4 can reach
      d_plan->opts.upsampfac = 1.25;
    }
    if (d_plan->opts.debug) {
      printf("[cufinufft] upsampfac automatically set to %.3g\n", d_plan->opts.upsampfac);
    }
  }
  /* Setup Spreader */
  if ((ier = setup_spreader_for_nufft(d_plan->spopts, tol, d_plan->opts)) > 1) {
    // can return FINUFFT_WARN_EPS_TOO_SMALL=1, which is OK
    goto finalize;
  }

  d_plan->type                    = type;
  d_plan->spopts.spread_direction = d_plan->type;

  if (d_plan->opts.debug) {
    // print the spreader options
    printf("[cufinufft] spreader options:\n");
    printf("[cufinufft] nspread: %d\n", d_plan->spopts.nspread);
  }

  cufinufft_setup_binsize<T>(type, d_plan->spopts.nspread, dim, &d_plan->opts);
  if (cudaGetLastError() != cudaSuccess) {
    ier = FINUFFT_ERR_CUDA_FAILURE;
    goto finalize;
  }
  if (d_plan->opts.debug) {
    printf("[cufinufft] bin size x: %d", d_plan->opts.gpu_binsizex);
    if (dim > 1) printf(" bin size y: %d", d_plan->opts.gpu_binsizey);
    if (dim > 2) printf(" bin size z: %d", d_plan->opts.gpu_binsizez);
    printf("\n");
    // shared memory required for the spreader vs available shared memory
    int shared_mem_per_block{};
    cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device_id);
    const auto mem_required =
        shared_memory_required<T>(dim, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex,
                                  d_plan->opts.gpu_binsizey, d_plan->opts.gpu_binsizez);
    printf("[cufinufft] shared memory required for the spreader: %ld\n", mem_required);
  }

  // dynamically request the maximum amount of shared memory available
  // for the spreader

  /* Automatically set GPU method. */
  if (d_plan->opts.gpu_method == 0) {
    /* For type 1, we default to method 2 (SM) since this is generally faster
     * if there is enough shared memory available. Otherwise, we default to GM.
     * Type 3 inherits this behavior since the outer plan here is also a type 1.
     *
     * For type 2, we always default to method 1 (GM).
     */
    if (type == 2) {
      d_plan->opts.gpu_method = 1;
    } else {
      // query the device for the amount of shared memory available
      int shared_mem_per_block{};
      cudaDeviceGetAttribute(&shared_mem_per_block,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
      // compute the amount of shared memory required for the method
      const auto shared_mem_required = shared_memory_required<T>(
          dim, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex,
          d_plan->opts.gpu_binsizey, d_plan->opts.gpu_binsizez);
      if ((shared_mem_required > shared_mem_per_block)) {
        d_plan->opts.gpu_method = 1;
      } else {
        d_plan->opts.gpu_method = 2;
      }
    }
  }

  if (cudaGetLastError() != cudaSuccess) {
    ier = FINUFFT_ERR_CUDA_FAILURE;
    goto finalize;
  }

  if (type == 1 || type == 2) {
    CUFINUFFT_BIGINT nf1 = 1, nf2 = 1, nf3 = 1;
    if (d_plan->opts.gpu_spreadinterponly) {
      // spread/interp grid is precisely the user "mode" sizes, no upsampling
      nf1 = d_plan->ms;
      if (dim > 1) nf2 = d_plan->mt;
      if (dim > 2) nf3 = d_plan->mu;
      if (d_plan->opts.debug) {
        printf("[cufinufft] spreadinterponly mode: (nf1,nf2,nf3) = (%d, %d, %d)\n", nf1,
               nf2, nf3);
      }
    } else { // usual NUFFT with fine grid using upsampling
      set_nf_type12(d_plan->ms, d_plan->opts, d_plan->spopts, &nf1,
                    d_plan->opts.gpu_obinsizex);
      if (dim > 1)
        set_nf_type12(d_plan->mt, d_plan->opts, d_plan->spopts, &nf2,
                      d_plan->opts.gpu_obinsizey);
      if (dim > 2)
        set_nf_type12(d_plan->mu, d_plan->opts, d_plan->spopts, &nf3,
                      d_plan->opts.gpu_obinsizez);
      if (d_plan->opts.debug)
        printf("[cufinufft] (nf1,nf2,nf3) = (%d, %d, %d)\n", nf1, nf2, nf3);
    }
    d_plan->nf1 = nf1;
    d_plan->nf2 = nf2;
    d_plan->nf3 = nf3;
    d_plan->nf  = nf1 * nf2 * nf3;

    using namespace cufinufft::memtransfer;
    switch (d_plan->dim) {
    case 1: {
      if ((ier = allocgpumem1d_plan<T>(d_plan))) goto finalize;
    } break;
    case 2: {
      if ((ier = allocgpumem2d_plan<T>(d_plan))) goto finalize;
    } break;
    case 3: {
      if ((ier = allocgpumem3d_plan<T>(d_plan))) goto finalize;
    } break;
    }

    // We dont need any cuFFT plans or kernel values if we are only spreading /
    // interpolating
    if (!d_plan->opts.gpu_spreadinterponly) {
      cufftHandle fftplan;
      cufftResult_t cufft_status;
      switch (d_plan->dim) {
      case 1: {
        int n[]       = {(int)nf1};
        int inembed[] = {(int)nf1};

        cufft_status = cufftPlanMany(&fftplan, 1, n, inembed, 1, inembed[0], inembed, 1,
                                     inembed[0], cufft_type<T>(), batchsize);
      } break;
      case 2: {
        int n[]       = {(int)nf2, (int)nf1};
        int inembed[] = {(int)nf2, (int)nf1};

        cufft_status =
            cufftPlanMany(&fftplan, 2, n, inembed, 1, inembed[0] * inembed[1], inembed, 1,
                          inembed[0] * inembed[1], cufft_type<T>(), batchsize);
      } break;
      case 3: {
        int n[]       = {(int)nf3, (int)nf2, (int)nf1};
        int inembed[] = {(int)nf3, (int)nf2, (int)nf1};

        cufft_status = cufftPlanMany(
            &fftplan, 3, n, inembed, 1, inembed[0] * inembed[1] * inembed[2], inembed, 1,
            inembed[0] * inembed[1] * inembed[2], cufft_type<T>(), batchsize);
      } break;
      }

      if (cufft_status != CUFFT_SUCCESS) {
        fprintf(stderr, "[%s] cufft makeplan error: %s", __func__,
                cufftGetErrorString(cufft_status));
        ier = FINUFFT_ERR_CUDA_FAILURE;
        goto finalize;
      }
      cufftSetStream(fftplan, stream);

      d_plan->fftplan = fftplan;

      // compute up to 3 * NQUAD precomputed values on CPU
      T fseries_precomp_phase[3 * MAX_NQUAD];
      T fseries_precomp_f[3 * MAX_NQUAD];
      thrust::device_vector<T> d_fseries_precomp_phase(3 * MAX_NQUAD);
      thrust::device_vector<T> d_fseries_precomp_f(3 * MAX_NQUAD);
      onedim_fseries_kernel_precomp<T>(d_plan->nf1, fseries_precomp_f,
                                       fseries_precomp_phase, d_plan->spopts);
      if (d_plan->dim > 1)
        onedim_fseries_kernel_precomp<T>(d_plan->nf2, fseries_precomp_f + MAX_NQUAD,
                                         fseries_precomp_phase + MAX_NQUAD,
                                         d_plan->spopts);
      if (d_plan->dim > 2)
        onedim_fseries_kernel_precomp<T>(d_plan->nf3, fseries_precomp_f + 2 * MAX_NQUAD,
                                         fseries_precomp_phase + 2 * MAX_NQUAD,
                                         d_plan->spopts);
      // copy the precomputed data to the device using thrust
      thrust::copy(fseries_precomp_phase, fseries_precomp_phase + 3 * MAX_NQUAD,
                   d_fseries_precomp_phase.begin());
      thrust::copy(fseries_precomp_f, fseries_precomp_f + 3 * MAX_NQUAD,
                   d_fseries_precomp_f.begin());
      // the full fseries is done on the GPU here
      if ((ier = fseries_kernel_compute(
               d_plan->dim, d_plan->nf1, d_plan->nf2, d_plan->nf3,
               d_fseries_precomp_f.data().get(), d_fseries_precomp_phase.data().get(),
               d_plan->fwkerhalf1, d_plan->fwkerhalf2, d_plan->fwkerhalf3,
               d_plan->spopts.nspread, stream)))
        goto finalize;
    }
  }
finalize:
  if (ier > 1) {
    cufinufft_destroy_impl(*d_plan_ptr);
    *d_plan_ptr = nullptr;
  }
  return ier;
}

template<typename T>
int cufinufft_setpts_12_impl(int M, T *d_kx, T *d_ky, T *d_kz,
                             cufinufft_plan_t<T> *d_plan)
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
  const cufinufft::utils::WithCudaDevice FromID(d_plan->opts.gpu_device_id);

  int nf1 = d_plan->nf1;
  int nf2 = d_plan->nf2;
  int nf3 = d_plan->nf3;
  int dim = d_plan->dim;

  d_plan->M = M;

  using namespace cufinufft::memtransfer;
  int ier;
  switch (d_plan->dim) {
  case 1: {
    ier = allocgpumem1d_nupts<T>(d_plan);
  } break;
  case 2: {
    ier = allocgpumem2d_nupts<T>(d_plan);
  } break;
  case 3: {
    ier = allocgpumem3d_nupts<T>(d_plan);
  } break;
  }
  if (ier) return ier;

  d_plan->kx = d_kx;
  if (dim > 1) d_plan->ky = d_ky;
  if (dim > 2) d_plan->kz = d_kz;

  using namespace cufinufft::spreadinterp;
  switch (d_plan->dim) {
  case 1: {
    if (d_plan->opts.gpu_method == 1 &&
        (ier = cuspread1d_nuptsdriven_prop<T>(nf1, M, d_plan)))
      fprintf(stderr, "error: cuspread1d_nupts_prop, method(%d)\n",
              d_plan->opts.gpu_method);
    if (d_plan->opts.gpu_method == 2 &&
        (ier = cuspread1d_subprob_prop<T>(nf1, M, d_plan)))
      fprintf(stderr, "error: cuspread1d_subprob_prop, method(%d)\n",
              d_plan->opts.gpu_method);
  } break;
  case 2: {
    if (d_plan->opts.gpu_method == 1 &&
        (ier = cuspread2d_nuptsdriven_prop<T>(nf1, nf2, M, d_plan)))
      fprintf(stderr, "error: cuspread2d_nupts_prop, method(%d)\n",
              d_plan->opts.gpu_method);
    if (d_plan->opts.gpu_method == 2 &&
        (ier = cuspread2d_subprob_prop<T>(nf1, nf2, M, d_plan)))
      fprintf(stderr, "error: cuspread2d_subprob_prop, method(%d)\n",
              d_plan->opts.gpu_method);
  } break;
  case 3: {
    if (d_plan->opts.gpu_method == 1 &&
        (ier = cuspread3d_nuptsdriven_prop<T>(nf1, nf2, nf3, M, d_plan)))
      fprintf(stderr, "error: cuspread3d_nuptsdriven_prop, method(%d)\n",
              d_plan->opts.gpu_method);
    if (d_plan->opts.gpu_method == 2 &&
        (ier = cuspread3d_subprob_prop<T>(nf1, nf2, nf3, M, d_plan)))
      fprintf(stderr, "error: cuspread3d_subprob_prop, method(%d)\n",
              d_plan->opts.gpu_method);
    if (d_plan->opts.gpu_method == 4 &&
        (ier = cuspread3d_blockgather_prop<T>(nf1, nf2, nf3, M, d_plan)))
      fprintf(stderr, "error: cuspread3d_blockgather_prop, method(%d)\n",
              d_plan->opts.gpu_method);
  } break;
  }

  if (d_plan->opts.debug) {
    printf("[cufinufft] plan->M=%d\n", M);
  }
  return ier;
}

template<typename T>
int cufinufft_setpts_impl(int M, T *d_kx, T *d_ky, T *d_kz, int N, T *d_s, T *d_t, T *d_u,
                          cufinufft_plan_t<T> *d_plan) {
  // type 1 and type 2 setpts
  if (d_plan->type == 1 || d_plan->type == 2) {
    return cufinufft_setpts_12_impl<T>(M, d_kx, d_ky, d_kz, d_plan);
  }
  // type 3 setpts

  // This code follows the same implementation of the CPU code in finufft and uses similar
  // variables names where possible. However, the use of GPU routines and paradigms make
  // it harder to follow. To understand the code, it is recommended to read the CPU code
  // first.

  if (d_plan->type != 3) {
    fprintf(stderr, "[%s] Invalid type (%d): should be 1, 2, or 3.\n", __func__,
            d_plan->type);
    return FINUFFT_ERR_TYPE_NOTVALID;
  }
  if (N < 0) {
    fprintf(stderr, "[cufinufft] Invalid N (%d): cannot be negative.\n", N);
    return FINUFFT_ERR_NUM_NU_PTS_INVALID;
  }
  if (N > MAX_NF) {
    fprintf(stderr, "[cufinufft] Invalid N (%d): cannot be greater than %d.\n", N,
            MAX_NF);
    return FINUFFT_ERR_NUM_NU_PTS_INVALID;
  }
  const auto stream = d_plan->stream;
  d_plan->N         = N;
  if (d_plan->dim > 0 && d_s == nullptr) {
    fprintf(stderr, "[%s] Error: d_s is nullptr but dim > 0.\n", __func__);
    return FINUFFT_ERR_INVALID_ARGUMENT;
  }
  d_plan->d_Sp = d_plan->dim > 0 ? d_s : nullptr;

  if (d_plan->dim > 1 && d_t == nullptr) {
    fprintf(stderr, "[%s] Error: d_t is nullptr but dim > 1.\n", __func__);
    return FINUFFT_ERR_INVALID_ARGUMENT;
  }
  d_plan->d_Tp = d_plan->dim > 1 ? d_t : nullptr;

  if (d_plan->dim > 2 && d_u == nullptr) {
    fprintf(stderr, "[%s] Error: d_u is nullptr but dim > 2.\n", __func__);
    return FINUFFT_ERR_INVALID_ARGUMENT;
  }
  d_plan->d_Up = d_plan->dim > 2 ? d_u : nullptr;

  const auto dim = d_plan->dim;
  // no need to set the params to zero, as they are already zeroed out in the plan
  //  memset(d_plan->type3_params, 0, sizeof(d_plan->type3_params));
  using namespace cufinufft::utils;
  if (d_plan->dim > 0) {
    const auto [x1, c1]        = arraywidcen<T>(M, d_kx, stream);
    d_plan->type3_params.X1    = x1;
    d_plan->type3_params.C1    = c1;
    const auto [S1, D1]        = arraywidcen<T>(N, d_s, stream);
    const auto [nf1, h1, gam1] = set_nhg_type3<T>(S1, x1, d_plan->opts, d_plan->spopts);
    d_plan->nf1                = nf1;
    d_plan->type3_params.S1    = S1;
    d_plan->type3_params.D1    = D1;
    d_plan->type3_params.h1    = h1;
    d_plan->type3_params.gam1  = gam1;
  }
  if (d_plan->dim > 1) {
    const auto [x2, c2]        = arraywidcen<T>(M, d_ky, stream);
    d_plan->type3_params.X2    = x2;
    d_plan->type3_params.C2    = c2;
    const auto [S2, D2]        = arraywidcen<T>(N, d_t, stream);
    const auto [nf2, h2, gam2] = set_nhg_type3<T>(S2, x2, d_plan->opts, d_plan->spopts);
    d_plan->nf2                = nf2;
    d_plan->type3_params.S2    = S2;
    d_plan->type3_params.D2    = D2;
    d_plan->type3_params.h2    = h2;
    d_plan->type3_params.gam2  = gam2;
  }
  if (d_plan->dim > 2) {
    const auto [x3, c3]        = arraywidcen<T>(M, d_kz, stream);
    d_plan->type3_params.X3    = x3;
    d_plan->type3_params.C3    = c3;
    const auto [S3, D3]        = arraywidcen<T>(N, d_u, stream);
    const auto [nf3, h3, gam3] = set_nhg_type3<T>(S3, x3, d_plan->opts, d_plan->spopts);
    d_plan->nf3                = nf3;
    d_plan->type3_params.S3    = S3;
    d_plan->type3_params.D3    = D3;
    d_plan->type3_params.h3    = h3;
    d_plan->type3_params.gam3  = gam3;
  }
  if (d_plan->opts.debug) {
    printf("[%s]", __func__);
    printf("\tM=%d N=%d\n", M, N);
    printf("\tX1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%d h1=%.3g\t\n",
           d_plan->type3_params.X1, d_plan->type3_params.C1, d_plan->type3_params.S1,
           d_plan->type3_params.D1, d_plan->type3_params.gam1, d_plan->nf1,
           d_plan->type3_params.h1);
    if (d_plan->dim > 1) {
      printf("\tX2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%d h2=%.3g\n",
             d_plan->type3_params.X2, d_plan->type3_params.C2, d_plan->type3_params.S2,
             d_plan->type3_params.D2, d_plan->type3_params.gam2, d_plan->nf2,
             d_plan->type3_params.h2);
    }
    if (d_plan->dim > 2) {
      printf("\tX3=%.3g C3=%.3g S3=%.3g D3=%.3g gam3=%g nf3=%d h3=%.3g\n",
             d_plan->type3_params.X3, d_plan->type3_params.C3, d_plan->type3_params.S3,
             d_plan->type3_params.D3, d_plan->type3_params.gam3, d_plan->nf3,
             d_plan->type3_params.h3);
    }
  }
  d_plan->nf = d_plan->nf1 * d_plan->nf2 * d_plan->nf3;

  // FIXME: MAX_NF might be too small...
  if (d_plan->nf * d_plan->opts.gpu_maxbatchsize > MAX_NF) {
    fprintf(stderr,
            "[%s t3] fwBatch would be bigger than MAX_NF, not attempting malloc!\n",
            __func__);
    return FINUFFT_ERR_MAXNALLOC;
  }

  // A macro might be better as it has access to __line__ and __func__
  const auto checked_free = [stream, pool = d_plan->supports_pools](auto x) constexpr {
    if (!x) return cudaFreeWrapper(x, stream, pool);
    return cudaSuccess;
  };
  const auto checked_realloc = [checked_free, pool = d_plan->supports_pools, stream](
                                   auto &x, const auto size) constexpr {
    if (auto ier = checked_free(x); ier != cudaSuccess) return ier;
    return cudaMallocWrapper(&x, size, stream, pool);
  };
  // FIXME: check the size of the allocs for the batch interface
  if (checked_realloc(d_plan->fw, sizeof(cuda_complex<T>) * d_plan->nf *
                                      d_plan->batchsize) != cudaSuccess)
    goto finalize;
  if (checked_realloc(d_plan->CpBatch, sizeof(cuda_complex<T>) * M * d_plan->batchsize) !=
      cudaSuccess)
    goto finalize;
  if (checked_realloc(d_plan->kx, sizeof(T) * M) != cudaSuccess) goto finalize;
  if (checked_realloc(d_plan->d_Sp, sizeof(T) * N) != cudaSuccess) goto finalize;
  if (d_plan->dim > 1) {
    if (checked_realloc(d_plan->ky, sizeof(T) * M) != cudaSuccess) goto finalize;
    if (checked_realloc(d_plan->d_Tp, sizeof(T) * N) != cudaSuccess) goto finalize;
  }
  if (d_plan->dim > 2) {
    if (checked_realloc(d_plan->kz, sizeof(T) * M) != cudaSuccess) goto finalize;
    if (checked_realloc(d_plan->d_Up, sizeof(T) * N) != cudaSuccess) goto finalize;
  }
  if (checked_realloc(d_plan->prephase, sizeof(cuda_complex<T>) * M) != cudaSuccess)
    goto finalize;
  if (checked_realloc(d_plan->deconv, sizeof(cuda_complex<T>) * N) != cudaSuccess)
    goto finalize;

  // NOTE: init-captures are not allowed for extended __host__ __device__ lambdas

  if (d_plan->dim > 0) {
    const auto ig1 = T(1) / d_plan->type3_params.gam1;
    const auto C1  = -d_plan->type3_params.C1;
    thrust::transform(thrust::cuda::par.on(stream), d_kx, d_kx + M, d_plan->kx,
                      [ig1, C1] __host__
                      __device__(const T x) -> T { return (x + C1) * ig1; });
  }
  if (d_plan->dim > 1) {
    const auto ig2 = T(1) / d_plan->type3_params.gam2;
    const auto C2  = -d_plan->type3_params.C2;
    thrust::transform(thrust::cuda::par.on(stream), d_ky, d_ky + M, d_plan->ky,
                      [ig2, C2] __host__
                      __device__(const T x) -> T { return (x + C2) * ig2; });
  }
  if (d_plan->dim > 2) {
    const auto ig3 = T(1) / d_plan->type3_params.gam3;
    const auto C3  = -d_plan->type3_params.C3;
    thrust::transform(thrust::cuda::par.on(stream), d_kz, d_kz + M, d_plan->kz,
                      [ig3, C3] __host__
                      __device__(const T x) -> T { return (x + C3) * ig3; });
  }
  if (d_plan->type3_params.D1 != 0 || d_plan->type3_params.D2 != 0 ||
      d_plan->type3_params.D3 != 0) {
    // if ky is null, use kx for ky and kz
    // this is not the most efficient implementation, but it is the most compact
    const auto iterator =
        thrust::make_zip_iterator(thrust::make_tuple(d_kx,
                                                     // to avoid out of bounds access, use
                                                     // kx if ky is null
                                                     (d_plan->dim > 1) ? d_ky : d_kx,
                                                     // same idea as above
                                                     (d_plan->dim > 2) ? d_kz : d_kx));
    const auto D1       = d_plan->type3_params.D1;
    const auto D2       = d_plan->type3_params.D2; // this should be 0 if dim < 2
    const auto D3       = d_plan->type3_params.D3; // this should be 0 if dim < 3
    const auto realsign = d_plan->iflag >= 0 ? T(1) : T(-1);
    thrust::transform(
        thrust::cuda::par.on(stream), iterator, iterator + M, d_plan->prephase,
        [D1, D2, D3, realsign] __host__
        __device__(const thrust::tuple<T, T, T> &tuple) -> cuda_complex<T> {
          const auto x = thrust::get<0>(tuple);
          const auto y = thrust::get<1>(tuple);
          const auto z = thrust::get<2>(tuple);
          // no branching because D2 and D3 are 0 if dim < 2 and dim < 3
          // this is generally faster on GPU
          const auto phase = D1 * x + D2 * y + D3 * z;
          // TODO: nvcc should have the sincos function
          //       check the cos + i*sin
          //       ref: https://en.wikipedia.org/wiki/Cis_(mathematics)
          return cuda_complex<T>{std::cos(phase), std::sin(phase) * realsign};
        });
  } else {
    thrust::fill(thrust::cuda::par.on(stream), d_plan->prephase, d_plan->prephase + M,
                 cuda_complex<T>{1, 0});
  }

  if (d_plan->dim > 0) {
    const auto scale = d_plan->type3_params.h1 * d_plan->type3_params.gam1;
    const auto D1    = -d_plan->type3_params.D1;
    thrust::transform(thrust::cuda::par.on(stream), d_s, d_s + N, d_plan->d_Sp,
                      [scale, D1] __host__
                      __device__(const T s) -> T { return scale * (s + D1); });
  }
  if (d_plan->dim > 1) {
    const auto scale = d_plan->type3_params.h2 * d_plan->type3_params.gam2;
    const auto D2    = -d_plan->type3_params.D2;
    thrust::transform(thrust::cuda::par.on(stream), d_t, d_t + N, d_plan->d_Tp,
                      [scale, D2] __host__
                      __device__(const T t) -> T { return scale * (t + D2); });
  }
  if (d_plan->dim > 2) {
    const auto scale = d_plan->type3_params.h3 * d_plan->type3_params.gam3;
    const auto D3    = -d_plan->type3_params.D3;
    thrust::transform(thrust::cuda::par.on(stream), d_u, d_u + N, d_plan->d_Up,
                      [scale, D3] __host__
                      __device__(const T u) -> T { return scale * (u + D3); });
  }
  { // here we declare phi_hat1, phi_hat2, and phi_hat3
    // and the precomputed data for the fseries kernel
    using namespace cufinufft::common;

    std::array<T, 3 * MAX_NQUAD> nuft_precomp_z{};
    std::array<T, 3 * MAX_NQUAD> nuft_precomp_f{};
    thrust::device_vector<T> d_nuft_precomp_z(3 * MAX_NQUAD);
    thrust::device_vector<T> d_nuft_precomp_f(3 * MAX_NQUAD);
    thrust::device_vector<T> phi_hat1, phi_hat2, phi_hat3;
    if (d_plan->dim > 0) {
      phi_hat1.resize(N);
    }
    if (d_plan->dim > 1) {
      phi_hat2.resize(N);
    }
    if (d_plan->dim > 2) {
      phi_hat3.resize(N);
    }
    onedim_nuft_kernel_precomp<T>(nuft_precomp_f.data(), nuft_precomp_z.data(),
                                  d_plan->spopts);
    if (d_plan->dim > 1) {
      onedim_nuft_kernel_precomp<T>(nuft_precomp_f.data() + MAX_NQUAD,
                                    nuft_precomp_z.data() + MAX_NQUAD,
                                    d_plan->spopts);
    }
    if (d_plan->dim > 2) {
      onedim_nuft_kernel_precomp<T>(nuft_precomp_f.data() + 2 * MAX_NQUAD,
                                    nuft_precomp_z.data() + 2 * MAX_NQUAD,
                                    d_plan->spopts);
    }
    // copy the precomputed data to the device using thrust
    thrust::copy(nuft_precomp_z.begin(), nuft_precomp_z.end(), d_nuft_precomp_z.begin());
    thrust::copy(nuft_precomp_f.begin(), nuft_precomp_f.end(), d_nuft_precomp_f.begin());
    // sync the stream before calling the kernel might be needed
    if (nuft_kernel_compute(d_plan->dim, N, N, N, d_nuft_precomp_f.data().get(),
                            d_nuft_precomp_z.data().get(), d_plan->d_Sp, d_plan->d_Tp,
                            d_plan->d_Up, phi_hat1.data().get(), phi_hat2.data().get(),
                            phi_hat3.data().get(), d_plan->spopts.nspread, stream))
      goto finalize;

    const auto is_c_finite = std::isfinite(d_plan->type3_params.C1) &&
                             std::isfinite(d_plan->type3_params.C2) &&
                             std::isfinite(d_plan->type3_params.C3);
    const auto is_c_nonzero = d_plan->type3_params.C1 != 0 ||
                              d_plan->type3_params.C2 != 0 ||
                              d_plan->type3_params.C3 != 0;

    const auto phi_hat_iterator = thrust::make_zip_iterator(
        thrust::make_tuple(phi_hat1.begin(),
                           // to avoid out of bounds access, use phi_hat1 if dim < 2
                           dim > 1 ? phi_hat2.begin() : phi_hat1.begin(),
                           // to avoid out of bounds access, use phi_hat1 if dim < 3
                           dim > 2 ? phi_hat3.begin() : phi_hat1.begin()));
    thrust::transform(thrust::cuda::par.on(stream), phi_hat_iterator,
                      phi_hat_iterator + N, d_plan->deconv,
                      [dim] __host__
                      __device__(const thrust::tuple<T, T, T> tuple) -> cuda_complex<T> {
                        auto phiHat = thrust::get<0>(tuple);
                        // in case dim < 2 or dim < 3, multiply by 1
                        phiHat *= (dim > 1) ? thrust::get<1>(tuple) : T(1);
                        phiHat *= (dim > 2) ? thrust::get<2>(tuple) : T(1);
                        return {T(1) / phiHat, T(0)};
                      });

    if (is_c_finite && is_c_nonzero) {
      const auto c1       = d_plan->type3_params.C1;
      const auto c2       = d_plan->type3_params.C2;
      const auto c3       = d_plan->type3_params.C3;
      const auto d1       = -d_plan->type3_params.D1;
      const auto d2       = -d_plan->type3_params.D2;
      const auto d3       = -d_plan->type3_params.D3;
      const auto realsign = d_plan->iflag >= 0 ? T(1) : T(-1);
      // passing d_s three times if dim == 1 because d_t and d_u are not allocated
      // passing d_s and d_t if dim == 2 because d_u is not allocated
      const auto phase_iterator = thrust::make_zip_iterator(
          thrust::make_tuple(d_s, dim > 1 ? d_t : d_s, dim > 2 ? d_u : d_s));
      thrust::transform(
          thrust::cuda::par.on(stream), phase_iterator, phase_iterator + N,
          d_plan->deconv, d_plan->deconv,
          [c1, c2, c3, d1, d2, d3, realsign] __host__
          __device__(const thrust::tuple<T, T, T> tuple, cuda_complex<T> deconv)
          -> cuda_complex<T> {
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

  using namespace cufinufft::memtransfer;
  switch (d_plan->dim) {
  case 1: {
    if ((allocgpumem1d_plan<T>(d_plan))) goto finalize;
  } break;
  case 2: {
    if ((allocgpumem2d_plan<T>(d_plan))) goto finalize;
  } break;
  case 3: {
    if ((allocgpumem3d_plan<T>(d_plan))) goto finalize;
  } break;
  }
  if (cufinufft_setpts_12_impl(M, d_plan->kx, d_plan->ky, d_plan->kz, d_plan)) {
    fprintf(stderr, "[%s] cufinufft_setpts_12_impl failed\n", __func__);
    goto finalize;
  }
  {
    int t2modes[]               = {d_plan->nf1, d_plan->nf2, d_plan->nf3};
    cufinufft_opts t2opts       = d_plan->opts;
    t2opts.gpu_spreadinterponly = 0;
    t2opts.gpu_method           = 0;
    // Safe to ignore the return value here?
    if (d_plan->t2_plan) cufinufft_destroy_impl(d_plan->t2_plan);
    // check that maxbatchsize is correct
    if (cufinufft_makeplan_impl<T>(2, dim, t2modes, d_plan->iflag, d_plan->batchsize,
                                   d_plan->tol, &d_plan->t2_plan, &t2opts)) {
      fprintf(stderr, "[%s] inner t2 plan cufinufft_makeplan failed\n", __func__);
      goto finalize;
    }
    if (cufinufft_setpts_12_impl(N, d_plan->d_Sp, d_plan->d_Tp, d_plan->d_Up,
                                 d_plan->t2_plan)) {
      fprintf(stderr, "[%s] inner t2 plan cufinufft_setpts_12 failed\n", __func__);
      goto finalize;
    }
    if (d_plan->t2_plan->spopts.spread_direction != 2) {
      fprintf(stderr, "[%s] inner t2 plan cufinufft_setpts_12 wrong direction\n",
              __func__);
      goto finalize;
    }
  }
  return 0;
finalize:
  cufinufft_destroy_impl(d_plan);
  return FINUFFT_ERR_CUDA_FAILURE;
}

template<typename T>
int cufinufft_execute_impl(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                           cufinufft_plan_t<T> *d_plan)
/*
    "exec" stage (single and double precision versions).

    The actual transformation is done here. Type and dimension of the
    transformation are defined in d_plan in previous stages.

        See ../docs/cppdoc.md for main user-facing documentation.

    Input/Output:
    d_c   a size d_plan->M CUFINUFFT_CPX array on gpu (input for Type 1; output for Type
          2)
    d_fk  a size d_plan->ms*d_plan->mt*d_plan->mu CUFINUFFT_CPX array on gpu ((input for
          Type 2; output for Type 1)

    Notes:
        i) Here CUFINUFFT_CPX is a defined type meaning either complex<float> or
   complex<double> to match the precision of the library called. ii) All operations are
   done on the GPU device (hence the d_* names)

    Melody Shih 07/25/19; Barnett 2/16/21.
*/
{
  cufinufft::utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
  int ier;
  int type = d_plan->type;
  switch (d_plan->dim) {
  case 1: {
    if (type == 1) ier = cufinufft1d1_exec<T>(d_c, d_fk, d_plan);
    if (type == 2) ier = cufinufft1d2_exec<T>(d_c, d_fk, d_plan);
    if (type == 3) ier = cufinufft1d3_exec<T>(d_c, d_fk, d_plan);
  } break;
  case 2: {
    if (type == 1) ier = cufinufft2d1_exec<T>(d_c, d_fk, d_plan);
    if (type == 2) ier = cufinufft2d2_exec<T>(d_c, d_fk, d_plan);
    if (type == 3) ier = cufinufft2d3_exec<T>(d_c, d_fk, d_plan);
  } break;
  case 3: {
    if (type == 1) ier = cufinufft3d1_exec<T>(d_c, d_fk, d_plan);
    if (type == 2) ier = cufinufft3d2_exec<T>(d_c, d_fk, d_plan);
    if (type == 3) ier = cufinufft3d3_exec<T>(d_c, d_fk, d_plan);
  } break;
  }

  return ier;
}

template<typename T>
int cufinufft_destroy_impl(cufinufft_plan_t<T> *d_plan)
/*
    "destroy" stage (single and double precision versions).

    In this stage, we
        (1) free all the memories that have been allocated on gpu
        (2) delete the cuFFT plan
*/
{

  // Can't destroy a null pointer.
  if (!d_plan) return FINUFFT_ERR_PLAN_NOTVALID;

  cufinufft::utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);

  using namespace cufinufft::memtransfer;
  freegpumemory<T>(d_plan);

  if (d_plan->fftplan) cufftDestroy(d_plan->fftplan);

  if (d_plan->t2_plan) cufinufft_destroy_impl(d_plan->t2_plan);

  /* free/destruct the plan */
  delete d_plan;

  return 0;
} // namespace cufinufft
#endif
