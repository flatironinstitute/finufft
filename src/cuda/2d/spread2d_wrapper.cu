#include <cassert>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cufinufft/common.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>

#include "spreadinterp2d.cuh"

using namespace cufinufft::common;

namespace cufinufft {
namespace spreadinterp {

template<typename T>
int cuspread2d(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    A wrapper for different spreading methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem

    Melody Shih 07/25/19
*/
{
  int nf1 = d_plan->nf1;
  int nf2 = d_plan->nf2;
  int M   = d_plan->M;

  int ier;
  switch (d_plan->opts.gpu_method) {
  case 1: {
    ier = cuspread2d_nuptsdriven<T>(nf1, nf2, M, d_plan, blksize);
  } break;
  case 2: {
    ier = cuspread2d_subprob<T>(nf1, nf2, M, d_plan, blksize);
  } break;
  default:
    std::cerr << "[cuspread2d] error: incorrect method, should be 1 or 2\n";
    ier = FINUFFT_ERR_METHOD_NOTVALID;
  }

  return ier;
}

template<typename T>
int cuspread2d_nuptsdriven_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;

  if (d_plan->opts.gpu_sort) {
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    if (bin_size_x < 0 || bin_size_y < 0) {
      std::cerr << "[cuspread2d_nuptsdriven_prop] error: invalid binsize "
                   "(binsizex, binsizey) = (";
      std::cerr << bin_size_x << "," << bin_size_y << ")" << std::endl;
      return FINUFFT_ERR_BINSIZE_NOTVALID;
    }

    int numbins[2];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;

    int *d_binsize     = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx     = d_plan->sortidx;
    int *d_idxnupts    = d_plan->idxnupts;

    int ier;
    if ((ier = checkCudaErrors(cudaMemsetAsync(
             d_binsize, 0, numbins[0] * numbins[1] * sizeof(int), stream))))
      return ier;

    calc_bin_size_noghost_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, nf1, nf2, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binsize, d_kx,
        d_ky, d_sortidx);
    RETURN_IF_CUDA_ERROR

    int n = numbins[0] * numbins[1];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

    calc_inverse_of_global_sort_index_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binstartpts, d_sortidx, d_kx,
        d_ky, d_idxnupts, nf1, nf2);
    RETURN_IF_CUDA_ERROR
  } else {
    int *d_idxnupts = d_plan->idxnupts;

    trivial_global_sort_index_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(M,
                                                                             d_idxnupts);
    RETURN_IF_CUDA_ERROR
  }

  return 0;
}

template<typename T>
int cuspread2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize) {
  auto &stream = d_plan->stream;
  dim3 threadsPerBlock;
  dim3 blocks;

  int ns          = d_plan->spopts.nspread; // psi's support in terms of number of cells
  int *d_idxnupts = d_plan->idxnupts;
  T es_c          = d_plan->spopts.ES_c;
  T es_beta       = d_plan->spopts.ES_beta;
  T sigma         = d_plan->spopts.upsampfac;

  T *d_kx               = d_plan->kx;
  T *d_ky               = d_plan->ky;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x          = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y          = 1;
  if (d_plan->opts.gpu_kerevalmeth) {
    for (int t = 0; t < blksize; t++) {
      spread_2d_nupts_driven<T, 1><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta,
          sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      spread_2d_nupts_driven<T, 0><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta,
          sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template<typename T>
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan)
/*
    This function determines the properties for spreading that are independent
    of the strength of the nodes,  only relates to the locations of the nodes,
    which only needs to be done once.
*/
{
  auto &stream = d_plan->stream;

  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
  int bin_size_x     = d_plan->opts.gpu_binsizex;
  int bin_size_y     = d_plan->opts.gpu_binsizey;
  if (bin_size_x < 0 || bin_size_y < 0) {
    std::cerr << "[cuspread2d_subprob_prop] error: invalid binsize (binsizex, "
                 "binsizey) = (";
    std::cerr << bin_size_x << "," << bin_size_y << ")" << std::endl;
    return FINUFFT_ERR_BINSIZE_NOTVALID;
  }
  int numbins[2];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);

  T *d_kx = d_plan->kx;
  T *d_ky = d_plan->ky;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_sortidx         = d_plan->sortidx;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int *d_subprob_to_bin = NULL;

  int ier;
  if ((ier = checkCudaErrors(
           cudaMemsetAsync(d_binsize, 0, numbins[0] * numbins[1] * sizeof(int), stream))))
    return ier;

  calc_bin_size_noghost_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, nf1, nf2, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binsize, d_kx, d_ky,
      d_sortidx);
  RETURN_IF_CUDA_ERROR

  int n = numbins[0] * numbins[1];
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  calc_inverse_of_global_sort_index_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binstartpts, d_sortidx, d_kx,
      d_ky, d_idxnupts, nf1, nf2);
  RETURN_IF_CUDA_ERROR
  calc_subprob_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_binsize, d_numsubprob, maxsubprobsize, numbins[0] * numbins[1]);
  RETURN_IF_CUDA_ERROR

  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  if ((ier = checkCudaErrors(cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream))))
    return ier;

  int totalnumsubprob;
  if ((ier =
           checkCudaErrors(cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n],
                                           sizeof(int), cudaMemcpyDeviceToHost, stream))))
    return ier;
  cudaStreamSynchronize(stream);
  if ((ier = checkCudaErrors(
           cudaMallocWrapper(&d_subprob_to_bin, totalnumsubprob * sizeof(int), stream,
                             d_plan->supports_pools))))
    return ier;
  map_b_into_subprob_2d<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_subprob_to_bin, d_subprobstartpts, d_numsubprob, numbins[0] * numbins[1]);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_subprob_to_bin);
    return FINUFFT_ERR_CUDA_FAILURE;
  }

  assert(d_subprob_to_bin != NULL);
  cudaFreeWrapper(d_plan->subprob_to_bin, stream, d_plan->supports_pools);
  d_plan->subprob_to_bin  = d_subprob_to_bin;
  d_plan->totalnumsubprob = totalnumsubprob;

  return 0;
}

template<typename T>
int cuspread2d_subprob(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                       int blksize) {
  auto &stream = d_plan->stream;

  int ns    = d_plan->spopts.nspread; // psi's support in terms of number of cells
  T es_c    = d_plan->spopts.ES_c;
  T es_beta = d_plan->spopts.ES_beta;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int bin_size_y = d_plan->opts.gpu_binsizey;
  int numbins[2];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);

  T *d_kx               = d_plan->kx;
  T *d_ky               = d_plan->ky;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  T sigma = d_plan->opts.upsampfac;

  const auto sharedplanorysize =
      shared_memory_required<T>(2, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex,
                                d_plan->opts.gpu_binsizey, d_plan->opts.gpu_binsizez);

  if (d_plan->opts.gpu_kerevalmeth) {
    for (int t = 0; t < blksize; t++) {
      if (const auto finufft_err =
              cufinufft_set_shared_memory(spread_2d_subprob<T, 1>, 2, *d_plan) != 0) {
        return FINUFFT_ERR_INSUFFICIENT_SHMEM;
      }
      RETURN_IF_CUDA_ERROR
      spread_2d_subprob<T, 1><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta,
          sigma, d_binstartpts, d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin,
          d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins[0], numbins[1],
          d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      if (const auto finufft_err =
              cufinufft_set_shared_memory(spread_2d_subprob<T, 0>, 2, *d_plan) != 0) {
        return FINUFFT_ERR_INSUFFICIENT_SHMEM;
      }
      RETURN_IF_CUDA_ERROR
      spread_2d_subprob<T, 0><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta,
          sigma, d_binstartpts, d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin,
          d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins[0], numbins[1],
          d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template int cuspread2d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuspread2d<double>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cuspread2d_subprob_prop<float>(int nf1, int nf2, int M,
                                            cufinufft_plan_t<float> *d_plan);
template int cuspread2d_subprob_prop<double>(int nf1, int nf2, int M,
                                             cufinufft_plan_t<double> *d_plan);
template int cuspread2d_nuptsdriven_prop<float>(int nf1, int nf2, int M,
                                                cufinufft_plan_t<float> *d_plan);
template int cuspread2d_nuptsdriven_prop<double>(int nf1, int nf2, int M,
                                                 cufinufft_plan_t<double> *d_plan);

} // namespace spreadinterp
} // namespace cufinufft
