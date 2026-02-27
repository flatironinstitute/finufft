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
      d_plan->binsize.resize(numbins);
      d_plan->binstartpts.resize(numbins);
    }
  } break;
  case 2:
  case 3: {
    int numbins = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
    d_plan->numsubprob.resize(numbins);
    d_plan->binsize.resize(numbins);
    d_plan->binstartpts.resize(numbins);
    d_plan->subprobstartpts.resize(numbins+1);
  } break;
  default:
    std::cerr << "err: invalid method " << std::endl;
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }

  if (!d_plan->opts.gpu_spreadinterponly) {
    d_plan->fwp.resize(maxbatchsize * nf1);
    d_plan->fw = dethrust(d_plan->fwp);
    d_plan->fwkerhalf[0].resize(nf1 / 2 + 1);
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
      d_plan->binsize.resize(numbins[0] * numbins[1]);
      d_plan->binstartpts.resize(numbins[0] * numbins[1]);
    }
  } break;
  case 2:
  case 3: {
    int64_t numbins[2];
    numbins[0] = ceil((T)nf1 / d_plan->opts.gpu_binsizex);
    numbins[1] = ceil((T)nf2 / d_plan->opts.gpu_binsizey);
    d_plan->numsubprob.resize(numbins[0] * numbins[1]);
    d_plan->binsize.resize(numbins[0] * numbins[1]);
    d_plan->binstartpts.resize(numbins[0] * numbins[1]);
    d_plan->subprobstartpts.resize(numbins[0] * numbins[1] + 1);
  } break;
  default:
    std::cerr << "[allocgpumem2d_plan] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }

  if (!d_plan->opts.gpu_spreadinterponly) {
    d_plan->fwp.resize(maxbatchsize * nf1 * nf2);
    d_plan->fw = dethrust(d_plan->fwp);
    d_plan->fwkerhalf[0].resize(nf1 / 2 + 1);
    d_plan->fwkerhalf[1].resize(nf2 / 2 + 1);
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
      d_plan->binsize.resize(nbins_tot);
      d_plan->binstartpts.resize(nbins_tot);
    }
  } break;
  case 2:
  case 3: {
    const int64_t nbins_tot = ceil((T)nf1 / d_plan->opts.gpu_binsizex) *
                              ceil((T)nf2 / d_plan->opts.gpu_binsizey) *
                              ceil((T)nf3 / d_plan->opts.gpu_binsizez);

    d_plan->numsubprob.resize(nbins_tot);
    d_plan->binsize.resize(nbins_tot);
    d_plan->binstartpts.resize(nbins_tot);
    d_plan->subprobstartpts.resize(nbins_tot + 1);
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

    d_plan->numsubprob.resize(numobins_tot);
    d_plan->binsize.resize(numbins_tot);
    d_plan->binstartpts.resize(numbins_tot + 1);
    d_plan->subprobstartpts.resize(numobins_tot + 1);
  } break;
  default:
    std::cerr << "[allocgpumem3d_plan] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }

  if (!d_plan->opts.gpu_spreadinterponly) {
    d_plan->fwp.resize(maxbatchsize * nf1 * nf2 * nf3);
    d_plan->fw = dethrust(d_plan->fwp);
    d_plan->fwkerhalf[0].resize(nf1 / 2 + 1);
    d_plan->fwkerhalf[1].resize(nf2 / 2 + 1);
    d_plan->fwkerhalf[2].resize(nf3 / 2 + 1);
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

  d_plan->sortidx.clear();
  d_plan->idxnupts.clear();

  switch (d_plan->opts.gpu_method) {
  case 1:
  case 2:
  case 3: {
    if (d_plan->opts.gpu_sort)
      d_plan->sortidx.resize(d_plan->M);
    d_plan->idxnupts.resize(d_plan->M);
  } break;
  default:
    std::cerr << "[allocgpumem1d_nupts] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
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

  d_plan->sortidx.clear();
  d_plan->idxnupts.clear();

  switch (d_plan->opts.gpu_method) {
  case 1: {
    if (d_plan->opts.gpu_sort)
      d_plan->sortidx.resize(d_plan->M);
    d_plan->idxnupts.resize(d_plan->M);
  } break;
  case 2:
  case 3: {
    d_plan->idxnupts.resize(d_plan->M);
    d_plan->sortidx.resize(d_plan->M);
  } break;
  default:
    std::cerr << "[allocgpumem2d_nupts] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
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

  switch (d_plan->opts.gpu_method) {
  case 1:
  case 2:
  case 3: {
    d_plan->idxnupts.resize(d_plan->M);
    d_plan->sortidx.resize(d_plan->M);
  } break;
  case 4: {
    d_plan->idxnupts.clear();
    d_plan->sortidx.resize(d_plan->M);
  } break;
  default:
    std::cerr << "[allocgpumem3d_nupts] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
}

template<typename T> void allocgpumem_plan(cufinufft_plan_t<T> *d_plan)
{
if (d_plan->dim==1) allocgpumem1d_plan(d_plan);
if (d_plan->dim==2) allocgpumem2d_plan(d_plan);
if (d_plan->dim==3) allocgpumem3d_plan(d_plan);
}
template void allocgpumem_plan(cufinufft_plan_t<float> *d_plan);
template void allocgpumem_plan(cufinufft_plan_t<double> *d_plan);

template<typename T> void allocgpumem_nupts(cufinufft_plan_t<T> *d_plan)
{
if (d_plan->dim==1) allocgpumem1d_nupts(d_plan);
if (d_plan->dim==2) allocgpumem2d_nupts(d_plan);
if (d_plan->dim==3) allocgpumem3d_nupts(d_plan);
}
template void allocgpumem_nupts(cufinufft_plan_t<float> *d_plan);
template void allocgpumem_nupts(cufinufft_plan_t<double> *d_plan);

} // namespace memtransfer
} // namespace cufinufft
