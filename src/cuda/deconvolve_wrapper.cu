#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cuda.h>
#include <cufinufft/contrib/helper_cuda.h>

#include <cufinufft/cudeconvolve.h>

namespace cufinufft {
namespace deconvolve {
/* Kernel for copying fw to fk with amplication by prefac/ker */
// Note: assume modeord=0: CMCL-compatible mode ordering in fk (from -N/2 up
// to N/2-1), modeord=1: FFT-compatible mode ordering in fk (from 0 to N/2-1, then -N/2 up
// to -1).
template<typename T, int modeord, int ndim>
__global__ static void deconvolve_nd(cuda::std::array<int,3> mstu, cuda::std::array<int,3> nf123, cuda_complex<T> *fw, cuda_complex<T> *fk,
                              cuda::std::array<T *, 3> fwkerhalf, bool fw2fk) {

  cuda::std::array<int, 3> m_acc {1, mstu[0], mstu[0]*mstu[1]};
  int mtotal = m_acc[ndim-1]*mstu[ndim-1];
  cuda::std::array<int, 3> nf_acc {1, nf123[0], nf123[0]*nf123[1]};

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < mtotal;
       i += blockDim.x * gridDim.x) {
    cuda::std::array<int,3> k;
    if constexpr (ndim==1)
      k = {i,0,0};
    if constexpr (ndim==2)
      k = {i%mstu[0], i/mstu[0], 0};
    if constexpr (ndim==3)
      k = {i%mstu[0], (i/mstu[0])%mstu[1], (i/mstu[0])/mstu[1]};
    T kervalue=1;
    int fkidx=0, fwidx=0;
    for (int idim=0; idim<ndim; ++idim) {
      int wn, fwkerindn;
      if constexpr (modeord == 0) {
        int pivot = k[idim] - mstu[idim] / 2;
        wn        = (pivot >= 0) ? pivot : nf123[idim] + pivot;
        fwkerindn = abs(pivot);
      } else {
        int pivot = k[idim] - mstu[idim] + mstu[idim] / 2;
        wn        = (pivot >= 0) ? nf123[idim] + k[idim] - mstu[idim] : k[idim];
        fwkerindn = (pivot >= 0) ? mstu[idim] - k[idim] : k[idim];
      }
      kervalue *= fwkerhalf[idim][fwkerindn];
      fwidx += wn*nf_acc[idim];
      fkidx += k[idim]*m_acc[idim];
    }

    if (fw2fk) {
      fk[fkidx].x  = fw[fwidx].x / kervalue;
      fk[fkidx].y  = fw[fwidx].y / kervalue;
    } else {
      fw[fwidx].x  = fk[fkidx].x / kervalue;
      fw[fwidx].y  = fk[fkidx].y / kervalue;
    }
  }
}

template<typename T, int modeord, int ndim>
static void cudeconvolve_nd(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    wrapper for deconvolution & amplification in 1/2/3D.

    Melody Shih 11/21/21
*/
{
  int nmodes=1, nftot=1;
  for (int idim=0; idim<ndim; ++idim) {
    nmodes *= d_plan->mstu[idim];
    nftot *= d_plan->nf123[idim];
  }

  bool fw2fk = d_plan->spopts.spread_direction == 1;
  if (!fw2fk)
    checkCudaErrors(cudaMemsetAsync(
        d_plan->fw, 0, d_plan->batchsize * nftot * sizeof(cuda_complex<T>), d_plan->stream));

  for (int t = 0; t < blksize; t++)
    deconvolve_nd<T, modeord, ndim><<<(nmodes + 256 - 1) / 256, 256, 0, d_plan->stream>>>(
      d_plan->mstu, d_plan->nf123, d_plan->fw + t * nftot, d_plan->fk + t * nmodes, dethrust(d_plan->fwkerhalf), fw2fk);
}

template<typename T>
void cudeconvolve(cufinufft_plan_t<T> *d_plan, int blksize)
  {
  if (d_plan->dim==1)
    (d_plan->opts.modeord == 0) ? cudeconvolve_nd<T,0,1>(d_plan, blksize)
                                : cudeconvolve_nd<T,1,1>(d_plan, blksize);
  if (d_plan->dim==2)
    (d_plan->opts.modeord == 0) ? cudeconvolve_nd<T,0,2>(d_plan, blksize)
                                : cudeconvolve_nd<T,1,2>(d_plan, blksize);
  if (d_plan->dim==3)
    (d_plan->opts.modeord == 0) ? cudeconvolve_nd<T,0,3>(d_plan, blksize)
                                : cudeconvolve_nd<T,1,3>(d_plan, blksize);
  }

template void cudeconvolve<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template void cudeconvolve<double>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace deconvolve
} // namespace cufinufft
