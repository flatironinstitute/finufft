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
void cufinufft1d_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
// 2d
template<typename T>
void cufinufft2d_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
// 3d
template<typename T>
void cufinufft3d_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
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
  using namespace finufft::common;
  *d_plan_ptr = nullptr;

  if (type < 1 || type > 3) {
    fprintf(stderr, "[%s] Invalid type (%d): should be 1, 2, or 3.\n", __func__, type);
    throw int(FINUFFT_ERR_TYPE_NOTVALID);
  }
  if (ntransf < 1) {
    fprintf(stderr, "[%s] Invalid ntransf (%d): should be at least 1.\n", __func__,
            ntransf);
    throw int(FINUFFT_ERR_NTRANS_NOTVALID);
  }

  /* If a user has not supplied their own options, assign defaults for them. */
  cufinufft_opts planopts;
  if (opts == nullptr) {  // use default opts
    cufinufft_default_opts(&planopts);
  } else {                // or read from what's passed in
    planopts = *opts; // keep a deep copy; changing *opts now has no effect
  }

  // Multi-GPU support: set the CUDA Device ID:
  const int device_id = planopts.gpu_device_id;
  const cufinufft::utils::WithCudaDevice FromID{device_id};

  // cudaMallocAsync isn't supported for all devices, regardless of cuda version. Check
  // for support
  int supports_pools=0;
  {
    cudaDeviceGetAttribute(&supports_pools, cudaDevAttrMemoryPoolsSupported,
                           device_id);
    static bool warned = false;
    if (!warned && !supports_pools && planopts.gpu_stream != nullptr) {
      fprintf(stderr,
              "[cufinufft] Warning: cudaMallocAsync not supported on this device. Use of "
              "CUDA streams may not perform optimally.\n");
      warned = true;
    }
  }

  /* allocate the plan structure, assign address to user pointer. */
  auto *d_plan = new cufinufft_plan_t<T>(planopts, bool(supports_pools));
  *d_plan_ptr = d_plan;
try{
  // Zero out your struct, (sets all pointers to NULL)
  // set nf1, nf2, nf3 to 1 for type 3, type 1, type 2 will overwrite this
  d_plan->nf123 = {1, 1, 1};
  d_plan->tol = tol;
  d_plan->opts = planopts;
  d_plan->dim = dim;
  d_plan->opts.gpu_maxbatchsize = std::max(d_plan->opts.gpu_maxbatchsize, 1);
  d_plan->opts.gpu_np = d_plan->opts.gpu_method == 3 ? d_plan->opts.gpu_np : 0;

  if (type != 3) {
    d_plan->mstu = {nmodes[0], nmodes[1], nmodes[2]};
    if (d_plan->opts.debug) {
      printf("[cufinufft] (ms,mt,mu): %d %d %d\n", d_plan->mstu[0], d_plan->mstu[1], d_plan->mstu[2]);
    }
  } else { // type 3 turns its outer type 1 into spreading-only
    d_plan->opts.gpu_spreadinterponly = 1;
  }

  d_plan->iflag   = (iflag >= 0) ? 1 : -1;
  d_plan->ntransf = ntransf;

  int batchsize = (opts != nullptr) ? opts->gpu_maxbatchsize : 0;
  // TODO: check if this is the right heuristic
  if (batchsize == 0)                 // implies: use a heuristic.
    batchsize = std::min(ntransf, 8); // heuristic from test codes
  d_plan->batchsize = batchsize;

  const auto stream = d_plan->stream = (cudaStream_t)d_plan->opts.gpu_stream;

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
  int ier = setup_spreader_for_nufft(d_plan->spopts, tol, d_plan->opts);

  d_plan->type                    = type;
  d_plan->spopts.spread_direction = d_plan->type;

  if (d_plan->opts.debug) {
    // print the spreader options
    printf("[cufinufft] spreader options:\n");
    printf("[cufinufft] nspread: %d\n", d_plan->spopts.nspread);
  }
  {
    /* Automatically set GPU method. */
    const bool auto_method = d_plan->opts.gpu_method == 0;
    if (auto_method) {
      // Default to method 2 (SM) for type 1/3, otherwise method 1 (GM).
      d_plan->opts.gpu_method = (type == 1 || type == 3) ? 2 : 1;
    }
    try {
      cufinufft_setup_binsize<T>(type, d_plan->spopts.nspread, dim, &d_plan->opts);
    } catch (const std::runtime_error &e) {
      if (auto_method) {
        // Auto-selection of SM failed, fall back to GM and try again.
        d_plan->opts.gpu_method = 1;
        cufinufft_setup_binsize<T>(type, d_plan->spopts.nspread, dim, &d_plan->opts);
      } else {
        // User-specified method failed, or the fallback GM method failed.
        fprintf(stderr, "%s, method %d\n", e.what(), d_plan->opts.gpu_method);
        throw int(FINUFFT_ERR_INSUFFICIENT_SHMEM);
      }
    }
    if (cudaGetLastError() != cudaSuccess) {
      throw int(FINUFFT_ERR_CUDA_FAILURE);
    }
  }
  // Bin size and memory info now printed in cufinufft_setup_binsize() (common.cu)
  // Additional runtime info at debug level 2
  if (d_plan->opts.debug >= 2) {
    printf("[cufinufft] Runtime: grid=(%ld,%ld,%ld), M=%ld\n", d_plan->nf123[0], d_plan->nf123[1],
           d_plan->nf123[2], d_plan->M);
  }

  // dynamically request the maximum amount of shared memory available
  // for the spreader

  if (cudaGetLastError() != cudaSuccess) {
    throw int(FINUFFT_ERR_CUDA_FAILURE);
  }

  if (type == 1 || type == 2) {
    CUFINUFFT_BIGINT nf1 = 1, nf2 = 1, nf3 = 1;
    if (d_plan->opts.gpu_spreadinterponly) {
      // spread/interp grid is precisely the user "mode" sizes, no upsampling
      nf1 = d_plan->mstu[0];
      if (dim > 1) nf2 = d_plan->mstu[1];
      if (dim > 2) nf3 = d_plan->mstu[2];
      if (d_plan->opts.debug) {
        printf("[cufinufft] spreadinterponly mode: (nf1,nf2,nf3) = (%d, %d, %d)\n", nf1,
               nf2, nf3);
      }
    } else { // usual NUFFT with fine grid using upsampling
      set_nf_type12(d_plan->mstu[0], d_plan->opts, d_plan->spopts, &nf1,
                    d_plan->opts.gpu_obinsizex);
      if (dim > 1)
        set_nf_type12(d_plan->mstu[1], d_plan->opts, d_plan->spopts, &nf2,
                      d_plan->opts.gpu_obinsizey);
      if (dim > 2)
        set_nf_type12(d_plan->mstu[2], d_plan->opts, d_plan->spopts, &nf3,
                      d_plan->opts.gpu_obinsizez);
      if (d_plan->opts.debug)
        printf("[cufinufft] (nf1,nf2,nf3) = (%d, %d, %d)\n", nf1, nf2, nf3);
    }
    d_plan->nf123[0] = nf1;
    d_plan->nf123[1] = nf2;
    d_plan->nf123[2] = nf3;
    d_plan->nf = nf1 * nf2 * nf3;

    cufinufft::memtransfer::allocgpumem_plan(d_plan);

    // We don't need any cuFFT plans or kernel values if we are only spreading /
    // interpolating
    if (!d_plan->opts.gpu_spreadinterponly) {
      cufftHandle fftplan;
      int n[3];
      int ntot = 1;
      for (int idim=0; idim<d_plan->dim; ++idim)
        {
        n[idim] = int(d_plan->nf123[d_plan->dim-idim-1]);
        ntot *= n[idim];
        }
      cufftResult_t cufft_status = cufftPlanMany(
            &fftplan, d_plan->dim, n, n,
            1, ntot, n, 1, ntot, cufft_type<T>(), batchsize);

      if (cufft_status != CUFFT_SUCCESS) {
        fprintf(stderr, "[%s] cufft makeplan error: %s", __func__,
                cufftGetErrorString(cufft_status));
        throw int(FINUFFT_ERR_CUDA_FAILURE);
      }
      cufftSetStream(fftplan, stream);

      d_plan->fftplan = fftplan;

      // compute up to 3 * NQUAD precomputed values on CPU
      T fseries_precomp_phase[3 * MAX_NQUAD];
      T fseries_precomp_f[3 * MAX_NQUAD];
      thrust::device_vector<T> d_fseries_precomp_phase(3 * MAX_NQUAD);
      thrust::device_vector<T> d_fseries_precomp_f(3 * MAX_NQUAD);
      for (int idim=0; idim<d_plan->dim; ++idim)
        onedim_fseries_kernel_precomp<T>(d_plan->nf123[idim], fseries_precomp_f+idim*MAX_NQUAD,
                                       fseries_precomp_phase+idim*MAX_NQUAD, d_plan->spopts);
      // copy the precomputed data to the device using thrust
      thrust::copy(fseries_precomp_phase, fseries_precomp_phase + 3 * MAX_NQUAD,
                   d_fseries_precomp_phase.begin());
      thrust::copy(fseries_precomp_f, fseries_precomp_f + 3 * MAX_NQUAD,
                   d_fseries_precomp_f.begin());
      // the full fseries is done on the GPU here
      fseries_kernel_compute(
               d_plan->dim, d_plan->nf123,
               d_fseries_precomp_f.data().get(), d_fseries_precomp_phase.data().get(),
               d_plan->fwkerhalf,
               d_plan->spopts.nspread, stream);
    }
  }
  return ier;
}
catch(...) {
  delete d_plan;
  *d_plan_ptr = nullptr;
  throw;
}
}

template<typename T>
void cufinufft_setpts_12_impl(int M, T *d_kx, T *d_ky, T *d_kz,
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

  d_plan->M = M;

  cufinufft::memtransfer::allocgpumem_nupts<T>(d_plan);

  d_plan->kxyz[0] = d_kx;
  if (d_plan->dim > 1) d_plan->kxyz[1] = d_ky;
  if (d_plan->dim > 2) d_plan->kxyz[2] = d_kz;

  using namespace cufinufft::spreadinterp;
  switch (d_plan->dim) {
  case 1: {
    cuspread1d_prop(d_plan);
  } break;
  case 2: {
    cuspread2d_prop(d_plan);
  } break;
  case 3: {
    cuspread3d_prop(d_plan);
  } break;
  }

  if (d_plan->opts.debug) {
    printf("[cufinufft] plan->M=%d\n", M);
  }
}

template<typename T>
void cufinufft_setpts_impl(int M, T *d_kx, T *d_ky, T *d_kz, int N, T *d_s, T *d_t, T *d_u,
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

  cuda::std::array<T *,3> d_kxyz = {d_kx, d_ky, d_kz};
  cuda::std::array<T *,3> d_stu = {d_s, d_t, d_u};

  if (d_plan->type != 3) {
    fprintf(stderr, "[%s] Invalid type (%d): should be 1, 2, or 3.\n", __func__,
            d_plan->type);
    throw int(FINUFFT_ERR_TYPE_NOTVALID);
  }
  if (N < 0) {
    fprintf(stderr, "[cufinufft] Invalid N (%d): cannot be negative.\n", N);
    throw int(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  }
  if (N > MAX_NF) {
    fprintf(stderr, "[cufinufft] Invalid N (%d): cannot be greater than %d.\n", N,
            MAX_NF);
    throw int(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  }
  const auto stream = d_plan->stream;
  d_plan->N         = N;
  if (d_plan->dim > 0 && d_s == nullptr) {
    fprintf(stderr, "[%s] Error: d_s is nullptr but dim > 0.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }
  d_plan->STU[0] = d_plan->dim > 0 ? d_s : nullptr;

  if (d_plan->dim > 1 && d_t == nullptr) {
    fprintf(stderr, "[%s] Error: d_t is nullptr but dim > 1.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }
  d_plan->STU[1] = d_plan->dim > 1 ? d_t : nullptr;

  if (d_plan->dim > 2 && d_u == nullptr) {
    fprintf(stderr, "[%s] Error: d_u is nullptr but dim > 2.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }
  d_plan->STU[2] = d_plan->dim > 2 ? d_u : nullptr;

  const auto dim = d_plan->dim;
  // no need to set the params to zero, as they are already zeroed out in the plan
  //  memset(d_plan->type3_params, 0, sizeof(d_plan->type3_params));
  using namespace cufinufft::utils;
  for (int idim=0; idim<d_plan->dim; ++idim) {
    const auto [xx, cc]        = arraywidcen<T>(M, d_kxyz[idim], stream);
    d_plan->type3_params.X[idim]    = xx;
    d_plan->type3_params.C[idim]    = cc;
    const auto [SS, DD]        = arraywidcen<T>(N, d_stu[idim], stream);
    const auto [nfnf, hh, gamgam] = set_nhg_type3<T>(SS, xx, d_plan->opts, d_plan->spopts);
    d_plan->nf123[idim]           = nfnf;
    d_plan->type3_params.S[idim]    = SS;
    d_plan->type3_params.D[idim]    = DD;
    d_plan->type3_params.h[idim]    = hh;
    d_plan->type3_params.gam[idim]  = gamgam;
  }
  if (d_plan->opts.debug) {
    printf("[%s]", __func__);
    printf("\tM=%d N=%d\n", M, N);
    printf("\tX1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%d h1=%.3g\t\n",
           d_plan->type3_params.X[0], d_plan->type3_params.C[0], d_plan->type3_params.S[0],
           d_plan->type3_params.D[0], d_plan->type3_params.gam[0], d_plan->nf123[0],
           d_plan->type3_params.h[0]);
    if (d_plan->dim > 1) {
      printf("\tX2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%d h2=%.3g\n",
             d_plan->type3_params.X[1], d_plan->type3_params.C[1], d_plan->type3_params.S[1],
             d_plan->type3_params.D[1], d_plan->type3_params.gam[1], d_plan->nf123[1],
             d_plan->type3_params.h[1]);
    }
    if (d_plan->dim > 2) {
      printf("\tX3=%.3g C3=%.3g S3=%.3g D3=%.3g gam3=%g nf3=%d h3=%.3g\n",
             d_plan->type3_params.X[2], d_plan->type3_params.C[2], d_plan->type3_params.S[2],
             d_plan->type3_params.D[2], d_plan->type3_params.gam[2], d_plan->nf123[2],
             d_plan->type3_params.h[2]);
    }
  }
  d_plan->nf = d_plan->nf123[0] * d_plan->nf123[1] * d_plan->nf123[2];

  // FIXME: MAX_NF might be too small...
  if (d_plan->nf * d_plan->opts.gpu_maxbatchsize > MAX_NF) {
    fprintf(stderr,
            "[%s t3] fwBatch would be bigger than MAX_NF, not attempting malloc!\n",
            __func__);
    throw int(FINUFFT_ERR_MAXNALLOC);
  }

  // FIXME: check the size of the allocs for the batch interface
  d_plan->fwp.resize(d_plan->nf*d_plan->batchsize);
  d_plan->fw = dethrust(d_plan->fwp);
  d_plan->CpBatch.resize(M * d_plan->batchsize);
  for (int idim=0; idim<d_plan->dim; ++idim) {
    d_plan->kxyzp[idim].resize(M);
    d_plan->kxyz[idim] = dethrust(d_plan->kxyzp[idim]);
    d_plan->STUp[idim].resize(N);
    d_plan->STU[idim] = dethrust(d_plan->STUp[idim]);
  }
  d_plan->prephase.resize(M);
  d_plan->deconv.resize(N);

  // NOTE: init-captures are not allowed for extended __host__ __device__ lambdas

  for (int idim=0; idim<d_plan->dim; ++idim) {
    const auto ig = T(1) / d_plan->type3_params.gam[idim];
    const auto C  = -d_plan->type3_params.C[idim];
    thrust::transform(thrust::cuda::par.on(stream), d_kxyz[idim], d_kxyz[idim] + M, d_plan->kxyz[idim],
                      [ig, C] __host__
                      __device__(const T x) -> T { return (x + C) * ig; });
  }
  if (d_plan->type3_params.D[0] != 0 || d_plan->type3_params.D[1] != 0 ||
      d_plan->type3_params.D[2] != 0) {
    // if ky is null, use kx for ky and kz
    // this is not the most efficient implementation, but it is the most compact
    const auto iterator =
        thrust::make_zip_iterator(thrust::make_tuple(d_kx,
                                                     // to avoid out of bounds access, use
                                                     // kx if ky is null
                                                     (d_plan->dim > 1) ? d_ky : d_kx,
                                                     // same idea as above
                                                     (d_plan->dim > 2) ? d_kz : d_kx));
    const auto D = d_plan->type3_params.D;
    const auto realsign = d_plan->iflag >= 0 ? T(1) : T(-1);
    thrust::transform(
        thrust::cuda::par.on(stream), iterator, iterator + M, dethrust(d_plan->prephase),
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
    thrust::fill(thrust::cuda::par.on(stream), dethrust(d_plan->prephase), dethrust(d_plan->prephase) + M,
                 cuda_complex<T>{1, 0});
  }

  for (int idim=0; idim<d_plan->dim; ++idim) {
    const auto scale = d_plan->type3_params.h[idim] * d_plan->type3_params.gam[idim];
    const auto D     = -d_plan->type3_params.D[idim];
    thrust::transform(thrust::cuda::par.on(stream), d_stu[idim], d_stu[idim] + N, d_plan->STU[idim],
                      [scale, D] __host__
                      __device__(const T s) -> T { return scale * (s + D); });
  }
  { // here we declare phi_hat1, phi_hat2, and phi_hat3
    // and the precomputed data for the fseries kernel
    using namespace cufinufft::common;

    std::array<T, 3 * MAX_NQUAD> nuft_precomp_z{};
    std::array<T, 3 * MAX_NQUAD> nuft_precomp_f{};
    thrust::device_vector<T> d_nuft_precomp_z(3 * MAX_NQUAD);
    thrust::device_vector<T> d_nuft_precomp_f(3 * MAX_NQUAD);
    cuda::std::array<gpuArray<T>,3> phi_hat123({gpuArray<T>{0,d_plan->alloc},gpuArray<T>{0,d_plan->alloc},gpuArray<T>{0,d_plan->alloc}});
    for (int idim=0; idim<d_plan->dim; ++idim)
      phi_hat123[idim].resize(N);
    for (int idim=0; idim<d_plan->dim; ++idim)
      onedim_nuft_kernel_precomp<T>(nuft_precomp_f.data()+idim*MAX_NQUAD,
                                    nuft_precomp_z.data()+idim*MAX_NQUAD,
                                    d_plan->spopts);
    // copy the precomputed data to the device using thrust
    thrust::copy(nuft_precomp_z.begin(), nuft_precomp_z.end(), d_nuft_precomp_z.begin());
    thrust::copy(nuft_precomp_f.begin(), nuft_precomp_f.end(), d_nuft_precomp_f.begin());
    // sync the stream before calling the kernel might be needed
    nuft_kernel_compute(d_plan->dim, {N, N, N}, d_nuft_precomp_f.data().get(),
                            d_nuft_precomp_z.data().get(), d_plan->STU,
                            phi_hat123, d_plan->spopts.nspread, stream);

    const auto is_c_finite = std::isfinite(d_plan->type3_params.C[0]) &&
                             std::isfinite(d_plan->type3_params.C[1]) &&
                             std::isfinite(d_plan->type3_params.C[2]);
    const auto is_c_nonzero = d_plan->type3_params.C[0] != 0 ||
                              d_plan->type3_params.C[1] != 0 ||
                              d_plan->type3_params.C[2] != 0;

    const auto phi_hat_iterator = thrust::make_zip_iterator(
        thrust::make_tuple(phi_hat123[0].data(),
                           // to avoid out of bounds access, use phi_hat1 if dim < 2
                           dim > 1 ? phi_hat123[1].data() : phi_hat123[0].data(),
                           // to avoid out of bounds access, use phi_hat1 if dim < 3
                           dim > 2 ? phi_hat123[2].data() : phi_hat123[0].data()));
    thrust::transform(thrust::cuda::par.on(stream), phi_hat_iterator,
                      phi_hat_iterator + N, dethrust(d_plan->deconv),
                      [dim] __host__
                      __device__(const thrust::tuple<T, T, T> tuple) -> cuda_complex<T> {
                        auto phiHat = thrust::get<0>(tuple);
                        // in case dim < 2 or dim < 3, multiply by 1
                        phiHat *= (dim > 1) ? thrust::get<1>(tuple) : T(1);
                        phiHat *= (dim > 2) ? thrust::get<2>(tuple) : T(1);
                        return {T(1) / phiHat, T(0)};
                      });

    if (is_c_finite && is_c_nonzero) {
      const auto c1       = d_plan->type3_params.C[0];
      const auto c2       = d_plan->type3_params.C[1];
      const auto c3       = d_plan->type3_params.C[2];
      const auto d1       = -d_plan->type3_params.D[0];
      const auto d2       = -d_plan->type3_params.D[1];
      const auto d3       = -d_plan->type3_params.D[2];
      const auto realsign = d_plan->iflag >= 0 ? T(1) : T(-1);
      // passing d_s three times if dim == 1 because d_t and d_u are not allocated
      // passing d_s and d_t if dim == 2 because d_u is not allocated
      const auto phase_iterator = thrust::make_zip_iterator(
          thrust::make_tuple(d_s, dim > 1 ? d_t : d_s, dim > 2 ? d_u : d_s));
      thrust::transform(
          thrust::cuda::par.on(stream), phase_iterator, phase_iterator + N,
          dethrust(d_plan->deconv), dethrust(d_plan->deconv),
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

  cufinufft::memtransfer::allocgpumem_plan<T>(d_plan);

  cufinufft_setpts_12_impl(M, d_plan->kxyz[0], d_plan->kxyz[1], d_plan->kxyz[2], d_plan);
  {
    int t2modes[]               = {d_plan->nf123[0], d_plan->nf123[1], d_plan->nf123[2]};
    cufinufft_opts t2opts       = d_plan->opts;
    t2opts.gpu_spreadinterponly = 0;
    t2opts.gpu_method           = 0;
    // Safe to ignore the return value here?
    if (d_plan->t2_plan) cufinufft_destroy_impl(d_plan->t2_plan);
    // check that maxbatchsize is correct
    cufinufft_makeplan_impl<T>(2, dim, t2modes, d_plan->iflag, d_plan->batchsize,
                                   d_plan->tol, &d_plan->t2_plan, &t2opts);
    cufinufft_setpts_12_impl(N, d_plan->STU[0], d_plan->STU[1], d_plan->STU[2],
                                 d_plan->t2_plan);
    if (d_plan->t2_plan->spopts.spread_direction != 2) {
      fprintf(stderr, "[%s] inner t2 plan cufinufft_setpts_12 wrong direction\n",
              __func__);
    }
  }
}

template<typename T>
void cufinufft_execute_impl(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
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
  switch (d_plan->dim) {
  case 1: {
    cufinufft1d_exec<T>(d_c, d_fk, d_plan);
  } break;
  case 2: {
    cufinufft2d_exec<T>(d_c, d_fk, d_plan);
  } break;
  case 3: {
    cufinufft3d_exec<T>(d_c, d_fk, d_plan);
  } break;
  }
}

template<typename T>
void cufinufft_destroy_impl(cufinufft_plan_t<T> *d_plan)
/*
    "destroy" stage (single and double precision versions).

    In this stage, we
        (1) free all the memories that have been allocated on gpu
        (2) delete the cuFFT plan
*/
{
  // Can't destroy a null pointer.
  if (!d_plan) throw int(FINUFFT_ERR_PLAN_NOTVALID);
  delete d_plan;
} // namespace cufinufft
#endif
