#ifndef CUFINUFFT_IMPL_H
#define CUFINUFFT_IMPL_H

#include <iostream>

#include <cufinufft/contrib/helper_cuda.h>

#include <cufinufft/common.h>
#include <cufinufft/cudeconvolve.h>
#include <cufinufft/defs.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

#include <finufft_errors.h>

// 1d
template<typename T>
int cufinufft1d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
template<typename T>
int cufinufft1d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);

// 2d
template<typename T>
int cufinufft2d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
template<typename T>
int cufinufft2d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);

// 3d
template<typename T>
int cufinufft3d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
template<typename T>
int cufinufft3d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
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
  int ier;
  cuDoubleComplex *d_a = nullptr; // fseries temp data
  T *d_f               = nullptr; // fseries temp data

  if (type < 1 || type > 2) {
    fprintf(stderr, "[%s] Invalid type (%d): should be 1 or 2.\n", __func__, type);
    return FINUFFT_ERR_TYPE_NOTVALID;
  }
  if (ntransf < 1) {
    fprintf(stderr, "[%s] Invalid ntransf (%d): should be at least 1.\n", __func__,
            ntransf);
    return FINUFFT_ERR_NTRANS_NOTVALID;
  }

  // Mult-GPU support: set the CUDA Device ID:
  const int device_id = opts == nullptr ? 0 : opts->gpu_device_id;
  cufinufft::utils::WithCudaDevice device_swapper(device_id);

  /* allocate the plan structure, assign address to user pointer. */
  auto *d_plan = new cufinufft_plan_t<T>;
  *d_plan_ptr  = d_plan;
  // Zero out your struct, (sets all pointers to NULL)
  memset(d_plan, 0, sizeof(*d_plan));
  /* If a user has not supplied their own options, assign defaults for them. */
  if (opts == nullptr) {  // use default opts
    cufinufft_default_opts(&(d_plan->opts));
  } else {                // or read from what's passed in
    d_plan->opts = *opts; // keep a deep copy; changing *opts now has no effect
  }

  // cudaMallocAsync isn't supported for all devices, regardless of cuda version. Check
  // for support
  cudaDeviceGetAttribute(&d_plan->supports_pools, cudaDevAttrMemoryPoolsSupported,
                         device_id);
  static bool warned = false;
  if (!warned && !d_plan->supports_pools && d_plan->opts.gpu_stream != nullptr) {
    fprintf(stderr,
            "[cufinufft] Warning: cudaMallocAsync not supported on this device. Use of "
            "CUDA streams may not perform optimally.\n");
    warned = true;
  }

  auto &stream = d_plan->stream = (cudaStream_t)d_plan->opts.gpu_stream;
  using namespace cufinufft::common;
  /* Setup Spreader */

  // can return FINUFFT_WARN_EPS_TOO_SMALL=1, which is OK
  if ((ier = setup_spreader_for_nufft(d_plan->spopts, tol, d_plan->opts)) > 1) {
    delete *d_plan_ptr;
    *d_plan_ptr = nullptr;
    return ier;
  }

  d_plan->dim = dim;
  d_plan->ms  = nmodes[0];
  d_plan->mt  = nmodes[1];
  d_plan->mu  = nmodes[2];

  cufinufft_setup_binsize<T>(type, d_plan->spopts.nspread, dim, &d_plan->opts);
  RETURN_IF_CUDA_ERROR

  CUFINUFFT_BIGINT nf1 = 1, nf2 = 1, nf3 = 1;
  set_nf_type12(d_plan->ms, d_plan->opts, d_plan->spopts, &nf1,
                d_plan->opts.gpu_obinsizex);
  if (dim > 1)
    set_nf_type12(d_plan->mt, d_plan->opts, d_plan->spopts, &nf2,
                  d_plan->opts.gpu_obinsizey);
  if (dim > 2)
    set_nf_type12(d_plan->mu, d_plan->opts, d_plan->spopts, &nf3,
                  d_plan->opts.gpu_obinsizez);

  // dynamically request the maximum amount of shared memory available
  // for the spreader

  /* Automatically set GPU method. */
  if (d_plan->opts.gpu_method == 0) {
    /* For type 1, we default to method 2 (SM) since this is generally faster
     * if there is enough shared memory available. Otherwise, we default to GM.
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
      RETURN_IF_CUDA_ERROR
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

  int fftsign = (iflag >= 0) ? 1 : -1;

  d_plan->nf1      = nf1;
  d_plan->nf2      = nf2;
  d_plan->nf3      = nf3;
  d_plan->iflag    = fftsign;
  d_plan->ntransf  = ntransf;
  int maxbatchsize = opts ? opts->gpu_maxbatchsize : 0;
  if (maxbatchsize == 0)                 // implies: use a heuristic.
    maxbatchsize = std::min(ntransf, 8); // heuristic from test codes
  d_plan->maxbatchsize = maxbatchsize;
  d_plan->type         = type;

  if (d_plan->type == 1) d_plan->spopts.spread_direction = 1;
  if (d_plan->type == 2) d_plan->spopts.spread_direction = 2;

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

  cufftHandle fftplan;
  cufftResult_t cufft_status;
  switch (d_plan->dim) {
  case 1: {
    int n[]       = {(int)nf1};
    int inembed[] = {(int)nf1};

    cufft_status = cufftPlanMany(&fftplan, 1, n, inembed, 1, inembed[0], inembed, 1,
                                 inembed[0], cufft_type<T>(), maxbatchsize);
  } break;
  case 2: {
    int n[]       = {(int)nf2, (int)nf1};
    int inembed[] = {(int)nf2, (int)nf1};

    cufft_status =
        cufftPlanMany(&fftplan, 2, n, inembed, 1, inembed[0] * inembed[1], inembed, 1,
                      inembed[0] * inembed[1], cufft_type<T>(), maxbatchsize);
  } break;
  case 3: {
    int n[]       = {(int)nf3, (int)nf2, (int)nf1};
    int inembed[] = {(int)nf3, (int)nf2, (int)nf1};

    cufft_status = cufftPlanMany(
        &fftplan, 3, n, inembed, 1, inembed[0] * inembed[1] * inembed[2], inembed, 1,
        inembed[0] * inembed[1] * inembed[2], cufft_type<T>(), maxbatchsize);
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
  {
    std::complex<double> *a = d_plan->fseries_precomp_a;
    T *f                    = d_plan->fseries_precomp_f;

    onedim_fseries_kernel_precomp(nf1, f, a, d_plan->spopts);
    if (dim > 1)
      onedim_fseries_kernel_precomp(nf2, f + MAX_NQUAD, a + MAX_NQUAD, d_plan->spopts);
    if (dim > 2)
      onedim_fseries_kernel_precomp(nf3, f + 2 * MAX_NQUAD, a + 2 * MAX_NQUAD,
                                    d_plan->spopts);

    if ((ier = checkCudaErrors(
             cudaMallocWrapper(&d_a, dim * MAX_NQUAD * sizeof(cuDoubleComplex), stream,
                               d_plan->supports_pools))))
      goto finalize;
    if ((ier = checkCudaErrors(cudaMallocWrapper(&d_f, dim * MAX_NQUAD * sizeof(T),
                                                 stream, d_plan->supports_pools))))
      goto finalize;
    if ((ier = checkCudaErrors(
             cudaMemcpyAsync(d_a, a, dim * MAX_NQUAD * sizeof(cuDoubleComplex),
                             cudaMemcpyHostToDevice, stream))))
      goto finalize;
    if ((ier = checkCudaErrors(cudaMemcpyAsync(d_f, f, dim * MAX_NQUAD * sizeof(T),
                                               cudaMemcpyHostToDevice, stream))))
      goto finalize;
    if ((ier = cufserieskernelcompute(
             d_plan->dim, nf1, nf2, nf3, d_f, d_a, d_plan->fwkerhalf1, d_plan->fwkerhalf2,
             d_plan->fwkerhalf3, d_plan->spopts.nspread, stream)))
      goto finalize;
  }

finalize:
  cudaFreeWrapper(d_a, stream, d_plan->supports_pools);
  cudaFreeWrapper(d_f, stream, d_plan->supports_pools);

  if (ier > 1) {
    delete *d_plan_ptr;
    *d_plan_ptr = nullptr;
  }

  return ier;
}

template<typename T>
int cufinufft_setpts_impl(int M, T *d_kx, T *d_ky, T *d_kz, int N, T *d_s, T *d_t, T *d_u,
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
  cufinufft::utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);

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

  return ier;
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
    if (type == 3) {
      std::cerr << "Not Implemented yet" << std::endl;
      ier = FINUFFT_ERR_TYPE_NOTVALID;
    }
  } break;
  case 2: {
    if (type == 1) ier = cufinufft2d1_exec<T>(d_c, d_fk, d_plan);
    if (type == 2) ier = cufinufft2d2_exec<T>(d_c, d_fk, d_plan);
    if (type == 3) {
      std::cerr << "Not Implemented yet" << std::endl;
      ier = FINUFFT_ERR_TYPE_NOTVALID;
    }
  } break;
  case 3: {
    if (type == 1) ier = cufinufft3d1_exec<T>(d_c, d_fk, d_plan);
    if (type == 2) ier = cufinufft3d2_exec<T>(d_c, d_fk, d_plan);
    if (type == 3) {
      std::cerr << "Not Implemented yet" << std::endl;
      ier = FINUFFT_ERR_TYPE_NOTVALID;
    }
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

        Also see ../docs/cppdoc.md for main user-facing documentation.
*/
{
  cufinufft::utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);

  // Can't destroy a null pointer.
  if (!d_plan) return FINUFFT_ERR_PLAN_NOTVALID;

  using namespace cufinufft::memtransfer;
  freegpumemory<T>(d_plan);

  if (d_plan->fftplan) cufftDestroy(d_plan->fftplan);

  /* free/destruct the plan */
  delete d_plan;

  return 0;
} // namespace cufinufft
#endif
