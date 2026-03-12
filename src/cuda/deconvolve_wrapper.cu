#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cuda.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/cufinufft_plan_t.h>

/* Kernel for copying fw to fk with amplication by prefac/ker */
// Note: assume modeord=0: CMCL-compatible mode ordering in fk (from -N/2 up
// to N/2-1), modeord=1: FFT-compatible mode ordering in fk (from 0 to N/2-1, then -N/2 up
// to -1).
template<typename T, int modeord, int ndim>
static __global__ void deconv_nd(
    cuda::std::array<int, 3> mstu, cuda::std::array<int, 3> nf123, cuda_complex<T> *fw,
    cuda_complex<T> *fk, cuda::std::array<const T *, 3> fwkerhalf, bool fw2fk) {

  cuda::std::array<int, 3> m_acc{1, mstu[0], mstu[0] * mstu[1]};
  int mtotal = m_acc[ndim - 1] * mstu[ndim - 1];
  cuda::std::array<int, 3> nf_acc{1, nf123[0], nf123[0] * nf123[1]};

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < mtotal;
       i += blockDim.x * gridDim.x) {
    cuda::std::array<int, 3> k;
    if constexpr (ndim == 1) k = {i, 0, 0};
    if constexpr (ndim == 2) k = {i % mstu[0], i / mstu[0], 0};
    if constexpr (ndim == 3)
      k = {i % mstu[0], (i / mstu[0]) % mstu[1], (i / mstu[0]) / mstu[1]};
    T kervalue = 1;
    int fkidx = 0, fwidx = 0;
    for (int idim = 0; idim < ndim; ++idim) {
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
      fwidx += wn * nf_acc[idim];
      fkidx += k[idim] * m_acc[idim];
    }

    if (fw2fk) {
      fk[fkidx].x = fw[fwidx].x / kervalue;
      fk[fkidx].y = fw[fwidx].y / kervalue;
    } else {
      fw[fwidx].x = fk[fkidx].x / kervalue;
      fw[fwidx].y = fk[fkidx].y / kervalue;
    }
  }
}

template<typename T>
template<int modeord, int ndim>
void cufinufft_plan_t<T>::deconvolve_nd<modeord, ndim>(cuda_complex<T> *fw, cuda_complex<T> *fk, int blksize) const
/*
    wrapper for deconvolution & amplification in 1/2/3D.

    Melody Shih 11/21/21
*/
{
  int nmodes = 1, nftot = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    nmodes *= mstu[idim];
    nftot *= nf123[idim];
  }

  bool fw2fk = spopts.spread_direction == 1;
  if (!fw2fk)
    checkCudaErrors(
        cudaMemsetAsync(fw, 0, batchsize * nftot * sizeof(cuda_complex<T>), stream));

  for (int t = 0; t < blksize; t++)
    deconv_nd<T, modeord, ndim><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
        mstu, nf123, fw + t * nftot, fk + t * nmodes, dethrust(fwkerhalf), fw2fk);
}

template<typename T> void cufinufft_plan_t<T>::deconvolve(cuda_complex<T> *fw, cuda_complex<T> *fk, int blksize) const {
  if (dim == 1)
    (opts.modeord == 0) ? deconvolve_nd<0, 1>(fw, fk, blksize) : deconvolve_nd<1, 1>(fw, fk, blksize);
  if (dim == 2)
    (opts.modeord == 0) ? deconvolve_nd<0, 2>(fw, fk, blksize) : deconvolve_nd<1, 2>(fw, fk, blksize);
  if (dim == 3)
    (opts.modeord == 0) ? deconvolve_nd<0, 3>(fw, fk, blksize) : deconvolve_nd<1, 3>(fw, fk, blksize);
}

template void cufinufft_plan_t<float>::deconvolve(cuda_complex<float> *fw, cuda_complex<float> *fk, int blksize) const;
template void cufinufft_plan_t<double>::deconvolve(cuda_complex<double> *fw, cuda_complex<double> *fk, int blksize) const;
