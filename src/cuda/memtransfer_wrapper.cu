#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/cufinufft_plan_t.h>
#include <cufinufft/utils.h>

#include <cufinufft/contrib/helper_cuda.h>

template<typename T>
void cufinufft_plan_t<T>::allocate() {
  cuda::std::array<int, 3> binsizes {opts.gpu_binsizex, opts.gpu_binsizey, opts.gpu_binsizez};

  switch (opts.gpu_method) {
  case 1: {
    if (opts.gpu_sort) {
      int numbins = 1;
      for (int idim=0; idim<dim; ++idim)
        numbins *= ceil((T)nf123[idim] / binsizes[idim]);
      binsize.resize(numbins);
      binstartpts.resize(numbins);
    }
  } break;
  case 2:
  case 3: {
    int numbins = 1;
    for (int idim=0; idim<dim; ++idim)
      numbins *= ceil((T)nf123[idim] / binsizes[idim]);
    numsubprob.resize(numbins);
    binsize.resize(numbins);
    binstartpts.resize(numbins);
    subprobstartpts.resize(numbins + 1);
  } break;
  case 4: {
    if (dim!=3) {
      std::cerr << "err: invalid method " << std::endl;
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
    cuda::std::array<int, 3> obinsizes {opts.gpu_obinsizex, opts.gpu_obinsizey, opts.gpu_obinsizez};
    int numobins_tot=1, numbins_tot=1;
    for (int idim=0; idim<dim; ++idim) {
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
    std::cerr << "[allocgpumem3d_plan] error: invalid method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
  if (!opts.gpu_spreadinterponly)
    for (int idim=0; idim<dim; ++idim)
      fwkerhalf[idim].resize(nf123[idim] / 2 + 1);
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
