#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/cufinufft_plan_t.h>
#include <cufinufft/utils.h>

#include <cufinufft/contrib/helper_cuda.h>

template<typename T>
void cufinufft_plan_t<T>::alloc1d()
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 11/21/21
*/
{
  int nf1 = nf123[0];

  switch (opts.gpu_method) {
  case 1: {
    if (opts.gpu_sort) {
      int numbins = ceil((T)nf1 / opts.gpu_binsizex);
      binsize.resize(numbins);
      binstartpts.resize(numbins);
    }
  } break;
  case 2:
  case 3: {
    int numbins = ceil((T)nf1 / opts.gpu_binsizex);
    numsubprob.resize(numbins);
    binsize.resize(numbins);
    binstartpts.resize(numbins);
    subprobstartpts.resize(numbins + 1);
  } break;
  default:
    std::cerr << "err: invalid method " << std::endl;
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }

  if (!opts.gpu_spreadinterponly) {
    fwkerhalf[0].resize(nf1 / 2 + 1);
  }
}

template<typename T>
void cufinufft_plan_t<T>::alloc2d()
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 07/25/19
*/
{
  int nf1 = nf123[0];
  int nf2 = nf123[1];

  switch (opts.gpu_method) {
  case 1: {
    if (opts.gpu_sort) {
      int numbins[2];
      numbins[0] = ceil((T)nf1 / opts.gpu_binsizex);
      numbins[1] = ceil((T)nf2 / opts.gpu_binsizey);
      binsize.resize(numbins[0] * numbins[1]);
      binstartpts.resize(numbins[0] * numbins[1]);
    }
  } break;
  case 2:
  case 3: {
    int64_t numbins[2];
    numbins[0] = ceil((T)nf1 / opts.gpu_binsizex);
    numbins[1] = ceil((T)nf2 / opts.gpu_binsizey);
    numsubprob.resize(numbins[0] * numbins[1]);
    binsize.resize(numbins[0] * numbins[1]);
    binstartpts.resize(numbins[0] * numbins[1]);
    subprobstartpts.resize(numbins[0] * numbins[1] + 1);
  } break;
  default:
    std::cerr << "[allocgpumem2d_plan] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }

  if (!opts.gpu_spreadinterponly) {
    fwkerhalf[0].resize(nf1 / 2 + 1);
    fwkerhalf[1].resize(nf2 / 2 + 1);
  }
}

template<typename T>
void cufinufft_plan_t<T>::alloc3d()
/*
    wrapper for gpu memory allocation in "plan" stage.

    Melody Shih 07/25/19
*/
{
  int nf1 = nf123[0];
  int nf2 = nf123[1];
  int nf3 = nf123[2];

  switch (opts.gpu_method) {
  case 1: {
    if (opts.gpu_sort) {
      const int64_t nbins_tot = ceil((T)nf1 / opts.gpu_binsizex) *
                                ceil((T)nf2 / opts.gpu_binsizey) *
                                ceil((T)nf3 / opts.gpu_binsizez);
      binsize.resize(nbins_tot);
      binstartpts.resize(nbins_tot);
    }
  } break;
  case 2:
  case 3: {
    const int64_t nbins_tot = ceil((T)nf1 / opts.gpu_binsizex) *
                              ceil((T)nf2 / opts.gpu_binsizey) *
                              ceil((T)nf3 / opts.gpu_binsizez);

    numsubprob.resize(nbins_tot);
    binsize.resize(nbins_tot);
    binstartpts.resize(nbins_tot);
    subprobstartpts.resize(nbins_tot + 1);
  } break;
  case 4: {
    const int numobins[3] = {(int)ceil((T)nf1 / opts.gpu_obinsizex),
                             (int)ceil((T)nf2 / opts.gpu_obinsizey),
                             (int)ceil((T)nf3 / opts.gpu_obinsizez)};

    const int binsperobins[3] = {opts.gpu_obinsizex / opts.gpu_binsizex,
                                 opts.gpu_obinsizey / opts.gpu_binsizey,
                                 opts.gpu_obinsizez / opts.gpu_binsizez};

    const int numbins[3] = {numobins[0] * (binsperobins[0] + 2),
                            numobins[1] * (binsperobins[1] + 2),
                            numobins[2] * (binsperobins[2] + 2)};

    const int64_t numobins_tot = numobins[0] * numobins[1] * numobins[2];
    const int64_t numbins_tot  = numbins[0] * numbins[1] * numbins[2];

    numsubprob.resize(numobins_tot);
    binsize.resize(numbins_tot);
    binstartpts.resize(numbins_tot + 1);
    subprobstartpts.resize(numobins_tot + 1);
  } break;
  default:
    std::cerr << "[allocgpumem3d_plan] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }

  if (!opts.gpu_spreadinterponly) {
    fwkerhalf[0].resize(nf1 / 2 + 1);
    fwkerhalf[1].resize(nf2 / 2 + 1);
    fwkerhalf[2].resize(nf3 / 2 + 1);
  }
}

template<typename T> void cufinufft_plan_t<T>::allocate() {
  if (dim == 1) alloc1d();
  if (dim == 2) alloc2d();
  if (dim == 3) alloc3d();
}
template void cufinufft_plan_t<float>::allocate();
template void cufinufft_plan_t<double>::allocate();

template<typename T>
void cufinufft_plan_t<T>::alloc1d_nupts()
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 11/21/21
*/
{
  sortidx.clear();
  idxnupts.clear();

  switch (opts.gpu_method) {
  case 1:
  case 2:
  case 3: {
    if (opts.gpu_sort) sortidx.resize(M);
    idxnupts.resize(M);
  } break;
  default:
    std::cerr << "[allocgpumem1d_nupts] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
}

template<typename T>
void cufinufft_plan_t<T>::alloc2d_nupts()
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 07/25/19
*/
{
  sortidx.clear();
  idxnupts.clear();

  switch (opts.gpu_method) {
  case 1: {
    if (opts.gpu_sort) sortidx.resize(M);
    idxnupts.resize(M);
  } break;
  case 2:
  case 3: {
    idxnupts.resize(M);
    sortidx.resize(M);
  } break;
  default:
    std::cerr << "[allocgpumem2d_nupts] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
}

template<typename T>
void cufinufft_plan_t<T>::alloc3d_nupts()
/*
    wrapper for gpu memory allocation in "setNUpts" stage.

    Melody Shih 07/25/19
*/
{
  switch (opts.gpu_method) {
  case 1:
  case 2:
  case 3: {
    idxnupts.resize(M);
    sortidx.resize(M);
  } break;
  case 4: {
    idxnupts.clear();
    sortidx.resize(M);
  } break;
  default:
    std::cerr << "[allocgpumem3d_nupts] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
}

template<typename T> void cufinufft_plan_t<T>::allocate_nupts() {
  if (dim == 1) alloc1d_nupts();
  if (dim == 2) alloc2d_nupts();
  if (dim == 3) alloc3d_nupts();
}
template void cufinufft_plan_t<float>::allocate_nupts();
template void cufinufft_plan_t<double>::allocate_nupts();
