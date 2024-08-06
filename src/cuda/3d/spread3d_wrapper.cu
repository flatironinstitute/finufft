#include <cassert>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cufinufft/common.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>

using namespace cufinufft::common;

#include "spreadinterp3d.cuh"

namespace cufinufft {
namespace spreadinterp {

template<typename T>
int cuspread3d(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    A wrapper for different spreading methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem
    (4) Block gather

    Melody Shih 07/25/19
*/
{
  int nf1 = d_plan->nf1;
  int nf2 = d_plan->nf2;
  int nf3 = d_plan->nf3;
  int M   = d_plan->M;

  int ier = 0;
  switch (d_plan->opts.gpu_method) {
  case 1: {
    ier = cuspread3d_nuptsdriven<T>(nf1, nf2, nf3, M, d_plan, blksize);
  } break;
  case 2: {
    ier = cuspread3d_subprob<T>(nf1, nf2, nf3, M, d_plan, blksize);
  } break;
  case 4: {
    ier = cuspread3d_blockgather<T>(nf1, nf2, nf3, M, d_plan, blksize);
  } break;
  default:
    std::cerr << "[cuspread3d] error: incorrect method, should be 1,2,4" << std::endl;
    ier = FINUFFT_ERR_METHOD_NOTVALID;
  }

  return ier;
}

template<typename T>
int cuspread3d_nuptsdriven_prop(int nf1, int nf2, int nf3, int M,
                                cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;

  if (d_plan->opts.gpu_sort) {
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int bin_size_z = d_plan->opts.gpu_binsizez;
    if (bin_size_x < 0 || bin_size_y < 0 || bin_size_z < 0) {
      std::cerr << "[cuspread3d_nuptsdriven_prop] error: invalid binsize "
                   "(binsizex, binsizey, binsizez) = (";
      std::cerr << bin_size_x << "," << bin_size_y << "," << bin_size_z << ")"
                << std::endl;
      return FINUFFT_ERR_BINSIZE_NOTVALID;
    }

    int numbins[3];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);
    numbins[2] = ceil((T)nf3 / bin_size_z);

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;

    int *d_binsize     = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx     = d_plan->sortidx;
    int *d_idxnupts    = d_plan->idxnupts;

    int ier;
    if ((ier = checkCudaErrors(cudaMemsetAsync(
             d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream))))
      return ier;
    calc_bin_size_noghost_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, nf1, nf2, nf3, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1],
        numbins[2], d_binsize, d_kx, d_ky, d_kz, d_sortidx);
    RETURN_IF_CUDA_ERROR

    int n = numbins[0] * numbins[1] * numbins[2];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

    calc_inverse_of_global_sort_index_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2],
        d_binstartpts, d_sortidx, d_kx, d_ky, d_kz, d_idxnupts, nf1, nf2, nf3);
    RETURN_IF_CUDA_ERROR
  } else {
    int *d_idxnupts = d_plan->idxnupts;

    trivial_global_sort_index_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(M,
                                                                             d_idxnupts);
    RETURN_IF_CUDA_ERROR
  }

  return 0;
}

template<typename T>
int cuspread3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize) {
  auto &stream = d_plan->stream;

  dim3 threadsPerBlock;
  dim3 blocks;

  int ns    = d_plan->spopts.nspread; // psi's support in terms of number of cells
  T sigma   = d_plan->spopts.upsampfac;
  T es_c    = d_plan->spopts.ES_c;
  T es_beta = d_plan->spopts.ES_beta;

  int *d_idxnupts       = d_plan->idxnupts;
  T *d_kx               = d_plan->kx;
  T *d_ky               = d_plan->ky;
  T *d_kz               = d_plan->kz;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x          = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y          = 1;

  if (d_plan->opts.gpu_kerevalmeth == 1) {
    for (int t = 0; t < blksize; t++) {
      spread_3d_nupts_driven<T, 1><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      spread_3d_nupts_driven<T, 0><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template<typename T>
int cuspread3d_blockgather_prop(int nf1, int nf2, int nf3, int M,
                                cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;

  dim3 threadsPerBlock;
  dim3 blocks;

  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
  int o_bin_size_x   = d_plan->opts.gpu_obinsizex;
  int o_bin_size_y   = d_plan->opts.gpu_obinsizey;
  int o_bin_size_z   = d_plan->opts.gpu_obinsizez;

  int numobins[3];
  if (nf1 % o_bin_size_x != 0 || nf2 % o_bin_size_y != 0 || nf3 % o_bin_size_z != 0) {
    std::cerr << "[cuspread3d_blockgather_prop] error:\n";
    std::cerr << "       mod(nf(1|2|3), opts.gpu_obinsize(x|y|z)) != 0" << std::endl;
    std::cerr << "       (nf1, nf2, nf3) = (" << nf1 << ", " << nf2 << ", " << nf3 << ")"
              << std::endl;
    std::cerr << "       (obinsizex, obinsizey, obinsizez) = (" << o_bin_size_x << ", "
              << o_bin_size_y << ", " << o_bin_size_z << ")" << std::endl;
    return FINUFFT_ERR_BINSIZE_NOTVALID;
  }

  numobins[0] = ceil((T)nf1 / o_bin_size_x);
  numobins[1] = ceil((T)nf2 / o_bin_size_y);
  numobins[2] = ceil((T)nf3 / o_bin_size_z);

  int bin_size_x = d_plan->opts.gpu_binsizex;
  int bin_size_y = d_plan->opts.gpu_binsizey;
  int bin_size_z = d_plan->opts.gpu_binsizez;
  if (o_bin_size_x % bin_size_x != 0 || o_bin_size_y % bin_size_y != 0 ||
      o_bin_size_z % bin_size_z != 0) {
    std::cerr << "[cuspread3d_blockgather_prop] error:\n";
    std::cerr << "      mod(ops.gpu_obinsize(x|y|z), opts.gpu_binsize(x|y|z)) != 0"
              << std::endl;
    std::cerr << "      (binsizex, binsizey, binsizez) = (" << bin_size_x << ", "
              << bin_size_y << ", " << bin_size_z << ")" << std::endl;
    std::cerr << "      (obinsizex, obinsizey, obinsizez) = (" << o_bin_size_x << ", "
              << o_bin_size_y << ", " << o_bin_size_z << ")" << std::endl;
    return FINUFFT_ERR_BINSIZE_NOTVALID;
  }

  int binsperobinx, binsperobiny, binsperobinz;
  int numbins[3];
  binsperobinx = o_bin_size_x / bin_size_x + 2;
  binsperobiny = o_bin_size_y / bin_size_y + 2;
  binsperobinz = o_bin_size_z / bin_size_z + 2;
  numbins[0]   = numobins[0] * (binsperobinx);
  numbins[1]   = numobins[1] * (binsperobiny);
  numbins[2]   = numobins[2] * (binsperobinz);

  T *d_kx = d_plan->kx;
  T *d_ky = d_plan->ky;
  T *d_kz = d_plan->kz;

  int *d_binsize         = d_plan->binsize;
  int *d_sortidx         = d_plan->sortidx;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_idxnupts        = NULL;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_subprob_to_bin  = NULL;

  int ier;
  if ((ier = checkCudaErrors(cudaMemsetAsync(
           d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream))))
    return ier;

  locate_nupts_to_bins_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2],
      binsperobinx, binsperobiny, binsperobinz, d_binsize, d_kx, d_ky, d_kz, d_sortidx,
      nf1, nf2, nf3);
  RETURN_IF_CUDA_ERROR

  threadsPerBlock.x = 8;
  threadsPerBlock.y = 8;
  threadsPerBlock.z = 8;

  blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
  blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
  blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

  fill_ghost_bins<<<blocks, threadsPerBlock, 0, stream>>>(
      binsperobinx, binsperobiny, binsperobinz, numobins[0], numobins[1], numobins[2],
      d_binsize);
  RETURN_IF_CUDA_ERROR

  int n = numbins[0] * numbins[1] * numbins[2];
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  if ((ier = checkCudaErrors(cudaMemsetAsync(d_binstartpts, 0, sizeof(int), stream))))
    return ier;

  int totalNUpts;
  if ((ier = checkCudaErrors(cudaMemcpyAsync(&totalNUpts, &d_binstartpts[n], sizeof(int),
                                             cudaMemcpyDeviceToHost, stream))))
    return ier;
  cudaStreamSynchronize(stream);
  if ((ier = checkCudaErrors(cudaMallocWrapper(&d_idxnupts, totalNUpts * sizeof(int),
                                               stream, d_plan->supports_pools))))
    return ier;

  calc_inverse_of_global_sort_index_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2],
      binsperobinx, binsperobiny, binsperobinz, d_binstartpts, d_sortidx, d_kx, d_ky,
      d_kz, d_idxnupts, nf1, nf2, nf3);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_idxnupts);
    return FINUFFT_ERR_CUDA_FAILURE;
  }

  threadsPerBlock.x = 2;
  threadsPerBlock.y = 2;
  threadsPerBlock.z = 2;

  blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
  blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
  blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

  ghost_bin_pts_index<<<blocks, threadsPerBlock, 0, stream>>>(
      binsperobinx, binsperobiny, binsperobinz, numobins[0], numobins[1], numobins[2],
      d_binsize, d_idxnupts, d_binstartpts, M);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_idxnupts);
    return FINUFFT_ERR_CUDA_FAILURE;
  }

  cudaFree(d_plan->idxnupts);
  d_plan->idxnupts = d_idxnupts;

  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  n = numobins[0] * numobins[1] * numobins[2];
  calc_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
      binsperobinx, binsperobiny, binsperobinz, d_binsize, d_numsubprob, maxsubprobsize,
      numobins[0] * numobins[1] * numobins[2]);
  RETURN_IF_CUDA_ERROR

  n        = numobins[0] * numobins[1] * numobins[2];
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
  map_b_into_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_subprob_to_bin, d_subprobstartpts, d_numsubprob, n);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_subprob_to_bin);
    return FINUFFT_ERR_CUDA_FAILURE;
  }

  assert(d_subprob_to_bin != NULL);
  cudaFree(d_plan->subprob_to_bin);
  d_plan->subprob_to_bin  = d_subprob_to_bin;
  d_plan->totalnumsubprob = totalnumsubprob;

  return 0;
}

template<typename T>
int cuspread3d_blockgather(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize) {
  auto &stream = d_plan->stream;

  int ns             = d_plan->spopts.nspread;
  T es_c             = d_plan->spopts.ES_c;
  T es_beta          = d_plan->spopts.ES_beta;
  T sigma            = d_plan->spopts.upsampfac;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  int obin_size_x = d_plan->opts.gpu_obinsizex;
  int obin_size_y = d_plan->opts.gpu_obinsizey;
  int obin_size_z = d_plan->opts.gpu_obinsizez;
  int bin_size_x  = d_plan->opts.gpu_binsizex;
  int bin_size_y  = d_plan->opts.gpu_binsizey;
  int bin_size_z  = d_plan->opts.gpu_binsizez;
  int numobins[3];
  numobins[0] = ceil((T)nf1 / obin_size_x);
  numobins[1] = ceil((T)nf2 / obin_size_y);
  numobins[2] = ceil((T)nf3 / obin_size_z);

  int binsperobinx, binsperobiny, binsperobinz;
  binsperobinx = obin_size_x / bin_size_x + 2;
  binsperobiny = obin_size_y / bin_size_y + 2;
  binsperobinz = obin_size_z / bin_size_z + 2;

  T *d_kx               = d_plan->kx;
  T *d_ky               = d_plan->ky;
  T *d_kz               = d_plan->kz;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binstartpts     = d_plan->binstartpts;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  size_t sharedplanorysize =
      obin_size_x * obin_size_y * obin_size_z * sizeof(cuda_complex<T>);
  if (sharedplanorysize > 49152) {
    std::cerr << "[cuspread3d_blockgather] error: not enough shared memory" << std::endl;
    return FINUFFT_ERR_INSUFFICIENT_SHMEM;
  }

  for (int t = 0; t < blksize; t++) {
    if (d_plan->opts.gpu_kerevalmeth == 1) {
      spread_3d_block_gather<T, 1><<<totalnumsubprob, 64, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_binstartpts, obin_size_x, obin_size_y, obin_size_z,
          binsperobinx * binsperobiny * binsperobinz, d_subprob_to_bin, d_subprobstartpts,
          maxsubprobsize, numobins[0], numobins[1], numobins[2], d_idxnupts);
      RETURN_IF_CUDA_ERROR
    } else {
      spread_3d_block_gather<T, 0><<<totalnumsubprob, 64, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_binstartpts, obin_size_x, obin_size_y, obin_size_z,
          binsperobinx * binsperobiny * binsperobinz, d_subprob_to_bin, d_subprobstartpts,
          maxsubprobsize, numobins[0], numobins[1], numobins[2], d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template<typename T>
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M,
                            cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;

  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
  int bin_size_x     = d_plan->opts.gpu_binsizex;
  int bin_size_y     = d_plan->opts.gpu_binsizey;
  int bin_size_z     = d_plan->opts.gpu_binsizez;
  if (bin_size_x < 0 || bin_size_y < 0 || bin_size_z < 0) {
    std::cerr << "error: invalid binsize (binsizex, binsizey, binsizez) = (";
    std::cerr << bin_size_x << "," << bin_size_y << "," << bin_size_z << ")" << std::endl;
    return FINUFFT_ERR_BINSIZE_NOTVALID;
  }

  int numbins[3];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);
  numbins[2] = ceil((T)nf3 / bin_size_z);

  T *d_kx = d_plan->kx;
  T *d_ky = d_plan->ky;
  T *d_kz = d_plan->kz;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_sortidx         = d_plan->sortidx;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int *d_subprob_to_bin = NULL;

  int ier;
  if ((ier = checkCudaErrors(cudaMemsetAsync(
           d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream))))
    return ier;
  calc_bin_size_noghost_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, nf1, nf2, nf3, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1],
      numbins[2], d_binsize, d_kx, d_ky, d_kz, d_sortidx);
  RETURN_IF_CUDA_ERROR

  int n = numbins[0] * numbins[1] * numbins[2];
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  calc_inverse_of_global_sort_index_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2],
      d_binstartpts, d_sortidx, d_kx, d_ky, d_kz, d_idxnupts, nf1, nf2, nf3);
  RETURN_IF_CUDA_ERROR
  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  calc_subprob_3d_v2<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_binsize, d_numsubprob, maxsubprobsize, numbins[0] * numbins[1] * numbins[2]);
  RETURN_IF_CUDA_ERROR

  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);
  int totalnumsubprob;
  if (checkCudaErrors(cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream)) ||
      checkCudaErrors(cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n],
                                      sizeof(int), cudaMemcpyDeviceToHost, stream)))
    return FINUFFT_ERR_CUDA_FAILURE;
  cudaStreamSynchronize(stream);
  if (checkCudaErrors(cudaMallocWrapper(&d_subprob_to_bin, totalnumsubprob * sizeof(int),
                                        stream, d_plan->supports_pools)))
    return FINUFFT_ERR_CUDA_FAILURE;

  map_b_into_subprob_3d_v2<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024, 0,
                             stream>>>(d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
                                       numbins[0] * numbins[1] * numbins[2]);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_subprob_to_bin);
    return FINUFFT_ERR_CUDA_FAILURE;
  }

  assert(d_subprob_to_bin != NULL);
  if (d_plan->subprob_to_bin != NULL) cudaFree(d_plan->subprob_to_bin);
  d_plan->subprob_to_bin = d_subprob_to_bin;
  assert(d_plan->subprob_to_bin != nullptr);
  d_plan->totalnumsubprob = totalnumsubprob;

  return 0;
}

template<typename T>
int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                       int blksize) {
  auto &stream = d_plan->stream;

  int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int bin_size_y = d_plan->opts.gpu_binsizey;
  int bin_size_z = d_plan->opts.gpu_binsizez;
  int numbins[3];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);
  numbins[2] = ceil((T)nf3 / bin_size_z);

  T *d_kx               = d_plan->kx;
  T *d_ky               = d_plan->ky;
  T *d_kz               = d_plan->kz;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  T sigma   = d_plan->spopts.upsampfac;
  T es_c    = d_plan->spopts.ES_c;
  T es_beta = d_plan->spopts.ES_beta;
  const auto sharedplanorysize =
      shared_memory_required<T>(3, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex,
                                d_plan->opts.gpu_binsizey, d_plan->opts.gpu_binsizez);
  for (int t = 0; t < blksize; t++) {
    if (d_plan->opts.gpu_kerevalmeth) {
      if (const auto finufft_err =
              cufinufft_set_shared_memory(spread_3d_subprob<T, 1>, 3, *d_plan) != 0) {
        return FINUFFT_ERR_INSUFFICIENT_SHMEM;
      }
      RETURN_IF_CUDA_ERROR
      spread_3d_subprob<T, 1><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3,
          sigma, es_c, es_beta, d_binstartpts, d_binsize, bin_size_x, bin_size_y,
          bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
          numbins[0], numbins[1], numbins[2], d_idxnupts);
      RETURN_IF_CUDA_ERROR
    } else {
      if (const auto finufft_err =
              cufinufft_set_shared_memory(spread_3d_subprob<T, 0>, 3, *d_plan) != 0) {
        return FINUFFT_ERR_INSUFFICIENT_SHMEM;
      }
      RETURN_IF_CUDA_ERROR
      spread_3d_subprob<T, 0><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3,
          sigma, es_c, es_beta, d_binstartpts, d_binsize, bin_size_x, bin_size_y,
          bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
          numbins[0], numbins[1], numbins[2], d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template int cuspread3d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuspread3d<double>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cuspread3d_nuptsdriven_prop<float>(int nf1, int nf2, int nf3, int M,
                                                cufinufft_plan_t<float> *d_plan);
template int cuspread3d_nuptsdriven_prop<double>(int nf1, int nf2, int nf3, int M,
                                                 cufinufft_plan_t<double> *d_plan);
template int cuspread3d_subprob_prop<float>(int nf1, int nf2, int nf3, int M,
                                            cufinufft_plan_t<float> *d_plan);
template int cuspread3d_subprob_prop<double>(int nf1, int nf2, int nf3, int M,
                                             cufinufft_plan_t<double> *d_plan);
template int cuspread3d_blockgather_prop<float>(int nf1, int nf2, int nf3, int M,
                                                cufinufft_plan_t<float> *d_plan);
template int cuspread3d_blockgather_prop<double>(int nf1, int nf2, int nf3, int M,
                                                 cufinufft_plan_t<double> *d_plan);

} // namespace spreadinterp
} // namespace cufinufft
