#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

#include <cufinufft/contrib/helper_cuda.h>

namespace cufinufft {
namespace memtransfer {

template<typename T>
void allocgpumem1d_plan(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 11/21/21
*/
{
  utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
  const auto stream = d_plan->stream;

  int nf1          = d_plan->nf123[0];
  int maxbatchsize = d_plan->batchsize;

  switch (d_plan->opts.gpu_method) {
  case 1: {
    if (d_plan->opts.gpu_sort) {
      int numbins = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
      checkCudaErrors(cudaMallocWrapper(
               &d_plan->binsize, numbins * sizeof(int), stream, d_plan->supports_pools));
      checkCudaErrors(
               cudaMallocWrapper(&d_plan->binstartpts, numbins * sizeof(int), stream,
                                 d_plan->supports_pools));
    }
  } break;
  case 2:
  case 3: {
    int numbins = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
    checkCudaErrors(cudaMallocWrapper(&d_plan->numsubprob, numbins * sizeof(int),
                                               stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(&d_plan->binsize, numbins * sizeof(int),
                                                 stream, d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->binstartpts, numbins * sizeof(int), stream,
                               d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->subprobstartpts, (numbins + 1) * sizeof(int),
                               stream, d_plan->supports_pools));
  } break;
  default:
    std::cerr << "err: invalid method " << std::endl;
    throw FINUFFT_ERR_METHOD_NOTVALID;
  }

  if (!d_plan->opts.gpu_spreadinterponly) {
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->fw, maxbatchsize * nf1 * sizeof(cuda_complex<T>),
                               stream, d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->fwkerhalf[0], (nf1 / 2 + 1) * sizeof(T), stream,
                               d_plan->supports_pools));
  }
}

template<typename T>
void allocgpumem1d_nupts(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 11/21/21
*/
{
  utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
  const auto stream = d_plan->stream;

  int M = d_plan->M;
  CUDA_FREE_AND_NULL(d_plan->sortidx, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->idxnupts, stream, d_plan->supports_pools);

  switch (d_plan->opts.gpu_method) {
  case 1:
  case 2:
  case 3: {
    if (d_plan->opts.gpu_sort)
        checkCudaErrors(cudaMallocWrapper(&d_plan->sortidx, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(&d_plan->idxnupts, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
  } break;
  default:
    std::cerr << "[allocgpumem1d_nupts] error: invalid method\n";
    throw FINUFFT_ERR_METHOD_NOTVALID;
  }
}

template<typename T>
void allocgpumem2d_plan(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 07/25/19
*/
{
  utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
  const auto stream = d_plan->stream;

  int nf1          = d_plan->nf123[0];
  int nf2          = d_plan->nf123[1];
  int maxbatchsize = d_plan->batchsize;

  switch (d_plan->opts.gpu_method) {
  case 1: {
    if (d_plan->opts.gpu_sort) {
      int numbins[2];
      numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
      numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
      checkCudaErrors(
               cudaMallocWrapper(&d_plan->binsize, numbins[0] * numbins[1] * sizeof(int),
                                 stream, d_plan->supports_pools));
      checkCudaErrors(cudaMallocWrapper(&d_plan->binstartpts,
                                                   numbins[0] * numbins[1] * sizeof(int),
                                                   stream, d_plan->supports_pools));
    }
  } break;
  case 2:
  case 3: {
    int64_t numbins[2];
    numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
    numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->numsubprob, numbins[0] * numbins[1] * sizeof(int),
                               stream, d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->binsize, numbins[0] * numbins[1] * sizeof(int),
                               stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(&d_plan->binstartpts,
                                                 numbins[0] * numbins[1] * sizeof(int),
                                                 stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(
             &d_plan->subprobstartpts, (numbins[0] * numbins[1] + 1) * sizeof(int),
             stream, d_plan->supports_pools));
  } break;
  default:
    std::cerr << "[allocgpumem2d_plan] error: invalid method\n";
    throw FINUFFT_ERR_METHOD_NOTVALID;
  }

  if (!d_plan->opts.gpu_spreadinterponly) {
    checkCudaErrors(cudaMallocWrapper(
             &d_plan->fw, maxbatchsize * nf1 * nf2 * sizeof(cuda_complex<T>), stream,
             d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->fwkerhalf[0], (nf1 / 2 + 1) * sizeof(T), stream,
                               d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->fwkerhalf[1], (nf2 / 2 + 1) * sizeof(T), stream,
                               d_plan->supports_pools));
  }
}

template<typename T>
void allocgpumem2d_nupts(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 07/25/19
*/
{
  utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
  const auto stream = d_plan->stream;

  const int M = d_plan->M;

  CUDA_FREE_AND_NULL(d_plan->sortidx, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->idxnupts, stream, d_plan->supports_pools);

  switch (d_plan->opts.gpu_method) {
  case 1: {
    if (d_plan->opts.gpu_sort)
        checkCudaErrors(cudaMallocWrapper(&d_plan->sortidx, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(&d_plan->idxnupts, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
  } break;
  case 2:
  case 3: {
    checkCudaErrors(cudaMallocWrapper(&d_plan->idxnupts, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(&d_plan->sortidx, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
  } break;
  default:
    std::cerr << "[allocgpumem2d_nupts] error: invalid method\n";
    throw FINUFFT_ERR_METHOD_NOTVALID;
  }
}

template<typename T>
void allocgpumem3d_plan(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 07/25/19
*/
{
  utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
  const auto stream = d_plan->stream;

  int nf1          = d_plan->nf123[0];
  int nf2          = d_plan->nf123[1];
  int nf3          = d_plan->nf123[2];
  int maxbatchsize = d_plan->batchsize;

  switch (d_plan->opts.gpu_method) {
  case 1: {
    if (d_plan->opts.gpu_sort) {
      const int64_t nbins_tot = ceil((T)nf1 / d_plan->opts.gpu_binsizex) *
                                ceil((T)nf2 / d_plan->opts.gpu_binsizey) *
                                ceil((T)nf3 / d_plan->opts.gpu_binsizez);
      checkCudaErrors(
               cudaMallocWrapper(&d_plan->binsize, nbins_tot * sizeof(int), stream,
                                 d_plan->supports_pools));
      checkCudaErrors(
               cudaMallocWrapper(&d_plan->binstartpts, nbins_tot * sizeof(int), stream,
                                 d_plan->supports_pools));
    }
  } break;
  case 2:
  case 3: {
    const int64_t nbins_tot = ceil((T)nf1 / d_plan->opts.gpu_binsizex) *
                              ceil((T)nf2 / d_plan->opts.gpu_binsizey) *
                              ceil((T)nf3 / d_plan->opts.gpu_binsizez);

    checkCudaErrors(
             cudaMallocWrapper(&d_plan->numsubprob, nbins_tot * sizeof(int), stream,
                               d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(
             &d_plan->binsize, nbins_tot * sizeof(int), stream, d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->binstartpts, nbins_tot * sizeof(int), stream,
                               d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->subprobstartpts, (nbins_tot + 1) * sizeof(int),
                               stream, d_plan->supports_pools));
  } break;
  case 4: {
    const int numobins[3] = {(int)ceil((T)nf1 / d_plan->opts.gpu_obinsizex),
                             (int)ceil((T)nf2 / d_plan->opts.gpu_obinsizey),
                             (int)ceil((T)nf3 / d_plan->opts.gpu_obinsizez)};

    const int binsperobins[3] = {d_plan->opts.gpu_obinsizex / d_plan->opts.gpu_binsizex,
                                 d_plan->opts.gpu_obinsizey / d_plan->opts.gpu_binsizey,
                                 d_plan->opts.gpu_obinsizez / d_plan->opts.gpu_binsizez};

    const int numbins[3] = {numobins[0] * (binsperobins[0] + 2),
                            numobins[1] * (binsperobins[1] + 2),
                            numobins[2] * (binsperobins[2] + 2)};

    const int64_t numobins_tot = numobins[0] * numobins[1] * numobins[2];
    const int64_t numbins_tot  = numbins[0] * numbins[1] * numbins[2];

    checkCudaErrors(
             cudaMallocWrapper(&d_plan->numsubprob, numobins_tot * sizeof(int), stream,
                               d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->binsize, numbins_tot * sizeof(int), stream,
                               d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->binstartpts, (numbins_tot + 1) * sizeof(int),
                               stream, d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->subprobstartpts, (numobins_tot + 1) * sizeof(int),
                               stream, d_plan->supports_pools));
  } break;
  default:
    std::cerr << "[allocgpumem3d_plan] error: invalid method\n";
    throw FINUFFT_ERR_METHOD_NOTVALID;
  }

  if (!d_plan->opts.gpu_spreadinterponly) {
    checkCudaErrors(cudaMallocWrapper(
             &d_plan->fw, maxbatchsize * nf1 * nf2 * nf3 * sizeof(cuda_complex<T>),
             stream, d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->fwkerhalf[0], (nf1 / 2 + 1) * sizeof(T), stream,
                               d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->fwkerhalf[1], (nf2 / 2 + 1) * sizeof(T), stream,
                               d_plan->supports_pools));
    checkCudaErrors(
             cudaMallocWrapper(&d_plan->fwkerhalf[2], (nf3 / 2 + 1) * sizeof(T), stream,
                               d_plan->supports_pools));
  }
}

template<typename T>
void allocgpumem3d_nupts(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 07/25/19
*/
{
  utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
  const auto stream = d_plan->stream;
  int M = d_plan->M;

  CUDA_FREE_AND_NULL(d_plan->sortidx, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->idxnupts, stream, d_plan->supports_pools);

  switch (d_plan->opts.gpu_method) {
  case 1: {
    if (d_plan->opts.gpu_sort)
        checkCudaErrors(cudaMallocWrapper(&d_plan->sortidx, M * sizeof(int),
                                                  stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(&d_plan->idxnupts, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
  } break;
  case 2: {
    checkCudaErrors(cudaMallocWrapper(&d_plan->idxnupts, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(&d_plan->sortidx, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
  } break;
  case 3: {
    checkCudaErrors(cudaMallocWrapper(&d_plan->idxnupts, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
    checkCudaErrors(cudaMallocWrapper(&d_plan->sortidx, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
  } break;
  case 4: {
    checkCudaErrors(cudaMallocWrapper(&d_plan->sortidx, M * sizeof(int),
                                                 stream, d_plan->supports_pools));
  } break;
  default:
    std::cerr << "[allocgpumem3d_nupts] error: invalid method\n";
    throw FINUFFT_ERR_METHOD_NOTVALID;
  }
}

template<typename T>
void freegpumemory(cufinufft_plan_t<T> *d_plan)
/*
    wrapper for freeing gpu memory.

    Melody Shih 11/21/21
*/
{
  utils::WithCudaDevice device_swapper(d_plan->opts.gpu_device_id);
  // Fixes a crash whewre the plan itself is deleted before the stream
  const auto stream = d_plan->stream;
  // Dont clear fw if spreadinterponly for type 1 and 2 as fw belongs to original program
  // (it is d_fk)
  if (!d_plan->opts.gpu_spreadinterponly || d_plan->type == 3)
    CUDA_FREE_AND_NULL(d_plan->fw, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->fwkerhalf[0], stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->fwkerhalf[1], stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->fwkerhalf[2], stream, d_plan->supports_pools);

  CUDA_FREE_AND_NULL(d_plan->idxnupts, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->sortidx, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->numsubprob, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->binsize, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->binstartpts, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->subprob_to_bin, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->subprobstartpts, stream, d_plan->supports_pools);

  CUDA_FREE_AND_NULL(d_plan->numnupts, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->numsubprob, stream, d_plan->supports_pools);

  if (d_plan->type != 3) {
    return;
  }

  CUDA_FREE_AND_NULL(d_plan->kxyz[0], stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->d_STUp[0], stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->kxyz[1], stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->d_STUp[1], stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->kxyz[2], stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->d_STUp[2], stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->prephase, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->deconv, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->fwbatch, stream, d_plan->supports_pools);
  CUDA_FREE_AND_NULL(d_plan->CpBatch, stream, d_plan->supports_pools);
}

template void allocgpumem1d_plan<float>(cufinufft_plan_t<float> *d_plan);
template void allocgpumem1d_plan<double>(cufinufft_plan_t<double> *d_plan);
template void allocgpumem1d_nupts<float>(cufinufft_plan_t<float> *d_plan);
template void allocgpumem1d_nupts<double>(cufinufft_plan_t<double> *d_plan);

template void freegpumemory<float>(cufinufft_plan_t<float> *d_plan);
template void freegpumemory<double>(cufinufft_plan_t<double> *d_plan);

template void allocgpumem2d_plan<float>(cufinufft_plan_t<float> *d_plan);
template void allocgpumem2d_plan<double>(cufinufft_plan_t<double> *d_plan);
template void allocgpumem2d_nupts<float>(cufinufft_plan_t<float> *d_plan);
template void allocgpumem2d_nupts<double>(cufinufft_plan_t<double> *d_plan);

template void allocgpumem3d_plan<float>(cufinufft_plan_t<float> *d_plan);
template void allocgpumem3d_plan<double>(cufinufft_plan_t<double> *d_plan);
template void allocgpumem3d_nupts<float>(cufinufft_plan_t<float> *d_plan);
template void allocgpumem3d_nupts<double>(cufinufft_plan_t<double> *d_plan);

} // namespace memtransfer
} // namespace cufinufft
