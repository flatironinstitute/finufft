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
  d_plan->fwp.clear();
  d_plan->fwkerhalf[0].clear();
  d_plan->fwkerhalf[1].clear();
  d_plan->fwkerhalf[2].clear();

  d_plan->idxnupts.clear();
  d_plan->sortidx.clear();
  d_plan->numsubprob.clear();
  d_plan->binsize.clear();
  d_plan->binstartpts.clear();
  d_plan->subprob_to_bin.clear();
  d_plan->subprobstartpts.clear();

  d_plan->numnupts.clear();
  d_plan->numsubprob.clear();

  if (d_plan->type != 3) {
    return;
  }

  d_plan->kxyzp[0].clear();
  d_plan->kxyzp[1].clear();
  d_plan->kxyzp[2].clear();
  d_plan->STUp[0].clear();
  d_plan->STUp[1].clear();
  d_plan->STUp[2].clear();
  d_plan->prephase.clear();
  d_plan->deconv.clear();
  d_plan->CpBatch.clear();
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
