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
template<typename T, int modeord>
__global__ void deconvolve_1d(int ms, int nf1, cuda_complex<T> *fw, cuda_complex<T> *fk,
                              T *fwkerhalf1) {
  int pivot1, w1, fwkerind1;
  T kervalue;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < ms;
       i += blockDim.x * gridDim.x) {
    if (modeord == 0) {
      pivot1    = i - ms / 2;
      w1        = (pivot1 >= 0) ? pivot1 : nf1 + pivot1;
      fwkerind1 = abs(pivot1);
    } else {
      pivot1    = i - ms + ms / 2;
      w1        = (pivot1 >= 0) ? nf1 + i - ms : i;
      fwkerind1 = (pivot1 >= 0) ? ms - i : i;
    }

    kervalue = fwkerhalf1[fwkerind1];
    fk[i].x  = fw[w1].x / kervalue;
    fk[i].y  = fw[w1].y / kervalue;
  }
}

template<typename T, int modeord>
__global__ void deconvolve_2d(int ms, int mt, int nf1, int nf2, cuda_complex<T> *fw,
                              cuda_complex<T> *fk, T *fwkerhalf1, T *fwkerhalf2) {
  int pivot1, pivot2, w1, w2, fwkerind1, fwkerind2;
  int k1, k2, inidx, outidx;
  T kervalue;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < ms * mt;
       i += blockDim.x * gridDim.x) {
    k1     = i % ms;
    k2     = i / ms;
    outidx = k1 + k2 * ms;

    if (modeord == 0) {
      pivot1    = k1 - ms / 2;
      pivot2    = k2 - mt / 2;
      w1        = (pivot1 >= 0) ? pivot1 : nf1 + pivot1;
      w2        = (pivot2 >= 0) ? pivot2 : nf2 + pivot2;
      fwkerind1 = abs(pivot1);
      fwkerind2 = abs(pivot2);
    } else {
      pivot1    = k1 - ms + ms / 2;
      pivot2    = k2 - mt + mt / 2;
      w1        = (pivot1 >= 0) ? nf1 + k1 - ms : k1;
      w2        = (pivot2 >= 0) ? nf2 + k2 - mt : k2;
      fwkerind1 = (pivot1 >= 0) ? ms - k1 : k1;
      fwkerind2 = (pivot2 >= 0) ? mt - k2 : k2;
    }

    inidx        = w1 + w2 * nf1;
    kervalue     = fwkerhalf1[fwkerind1] * fwkerhalf2[fwkerind2];
    fk[outidx].x = fw[inidx].x / kervalue;
    fk[outidx].y = fw[inidx].y / kervalue;
  }
}

template<typename T, int modeord>
__global__ void deconvolve_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3,
                              cuda_complex<T> *fw, cuda_complex<T> *fk, T *fwkerhalf1,
                              T *fwkerhalf2, T *fwkerhalf3) {
  int pivot1, pivot2, pivot3, w1, w2, w3, fwkerind1, fwkerind2, fwkerind3;
  int k1, k2, k3, inidx, outidx;
  T kervalue;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < ms * mt * mu;
       i += blockDim.x * gridDim.x) {
    k1     = i % ms;
    k2     = (i / ms) % mt;
    k3     = (i / ms / mt);
    outidx = k1 + k2 * ms + k3 * ms * mt;

    if (modeord == 0) {
      pivot1    = k1 - ms / 2;
      pivot2    = k2 - mt / 2;
      pivot3    = k3 - mu / 2;
      w1        = (pivot1 >= 0) ? pivot1 : nf1 + pivot1;
      w2        = (pivot2 >= 0) ? pivot2 : nf2 + pivot2;
      w3        = (pivot3 >= 0) ? pivot3 : nf3 + pivot3;
      fwkerind1 = abs(pivot1);
      fwkerind2 = abs(pivot2);
      fwkerind3 = abs(pivot3);
    } else {
      pivot1    = k1 - ms + ms / 2;
      pivot2    = k2 - mt + mt / 2;
      pivot3    = k3 - mu + mu / 2;
      w1        = (pivot1 >= 0) ? nf1 + k1 - ms : k1;
      w2        = (pivot2 >= 0) ? nf2 + k2 - mt : k2;
      w3        = (pivot3 >= 0) ? nf3 + k3 - mu : k3;
      fwkerind1 = (pivot1 >= 0) ? ms - k1 : k1;
      fwkerind2 = (pivot2 >= 0) ? mt - k2 : k2;
      fwkerind3 = (pivot3 >= 0) ? mu - k3 : k3;
    }

    inidx        = w1 + w2 * nf1 + w3 * nf1 * nf2;
    kervalue     = fwkerhalf1[fwkerind1] * fwkerhalf2[fwkerind2] * fwkerhalf3[fwkerind3];
    fk[outidx].x = fw[inidx].x / kervalue;
    fk[outidx].y = fw[inidx].y / kervalue;
  }
}

/* Kernel for copying fk to fw with same amplication */
template<typename T, int modeord>
__global__ void amplify_1d(int ms, int nf1, cuda_complex<T> *fw, cuda_complex<T> *fk,
                           T *fwkerhalf1) {
  int pivot1, w1, fwkerind1;
  T kervalue;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < ms;
       i += blockDim.x * gridDim.x) {
    if (modeord == 0) {
      pivot1    = i - ms / 2;
      w1        = (pivot1 >= 0) ? pivot1 : nf1 + pivot1;
      fwkerind1 = abs(pivot1);
    } else {
      pivot1    = i - ms + ms / 2;
      w1        = (pivot1 >= 0) ? nf1 + i - ms : i;
      fwkerind1 = (pivot1 >= 0) ? ms - i : i;
    }

    kervalue = fwkerhalf1[fwkerind1];
    fw[w1].x = fk[i].x / kervalue;
    fw[w1].y = fk[i].y / kervalue;
  }
}

template<typename T, int modeord>
__global__ void amplify_2d(int ms, int mt, int nf1, int nf2, cuda_complex<T> *fw,
                           cuda_complex<T> *fk, T *fwkerhalf1, T *fwkerhalf2) {
  int pivot1, pivot2, w1, w2, fwkerind1, fwkerind2;
  int k1, k2, inidx, outidx;
  T kervalue;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < ms * mt;
       i += blockDim.x * gridDim.x) {
    k1    = i % ms;
    k2    = i / ms;
    inidx = k1 + k2 * ms;

    if (modeord == 0) {
      pivot1    = k1 - ms / 2;
      pivot2    = k2 - mt / 2;
      w1        = (pivot1 >= 0) ? pivot1 : nf1 + pivot1;
      w2        = (pivot2 >= 0) ? pivot2 : nf2 + pivot2;
      fwkerind1 = abs(pivot1);
      fwkerind2 = abs(pivot2);
    } else {
      pivot1    = k1 - ms + ms / 2;
      pivot2    = k2 - mt + mt / 2;
      w1        = (pivot1 >= 0) ? nf1 + k1 - ms : k1;
      w2        = (pivot2 >= 0) ? nf2 + k2 - mt : k2;
      fwkerind1 = (pivot1 >= 0) ? ms - k1 : k1;
      fwkerind2 = (pivot2 >= 0) ? mt - k2 : k2;
    }

    outidx       = w1 + w2 * nf1;
    kervalue     = fwkerhalf1[fwkerind1] * fwkerhalf2[fwkerind2];
    fw[outidx].x = fk[inidx].x / kervalue;
    fw[outidx].y = fk[inidx].y / kervalue;
  }
}

template<typename T, int modeord>
__global__ void amplify_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3,
                           cuda_complex<T> *fw, cuda_complex<T> *fk, T *fwkerhalf1,
                           T *fwkerhalf2, T *fwkerhalf3) {
  int pivot1, pivot2, pivot3, w1, w2, w3, fwkerind1, fwkerind2, fwkerind3;
  int k1, k2, k3, inidx, outidx;
  T kervalue;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < ms * mt * mu;
       i += blockDim.x * gridDim.x) {
    k1    = i % ms;
    k2    = (i / ms) % mt;
    k3    = (i / ms / mt);
    inidx = k1 + k2 * ms + k3 * ms * mt;

    if (modeord == 0) {
      pivot1    = k1 - ms / 2;
      pivot2    = k2 - mt / 2;
      pivot3    = k3 - mu / 2;
      w1        = (pivot1 >= 0) ? pivot1 : nf1 + pivot1;
      w2        = (pivot2 >= 0) ? pivot2 : nf2 + pivot2;
      w3        = (pivot3 >= 0) ? pivot3 : nf3 + pivot3;
      fwkerind1 = abs(pivot1);
      fwkerind2 = abs(pivot2);
      fwkerind3 = abs(pivot3);
    } else {
      pivot1    = k1 - ms + ms / 2;
      pivot2    = k2 - mt + mt / 2;
      pivot3    = k3 - mu + mu / 2;
      w1        = (pivot1 >= 0) ? nf1 + k1 - ms : k1;
      w2        = (pivot2 >= 0) ? nf2 + k2 - mt : k2;
      w3        = (pivot3 >= 0) ? nf3 + k3 - mu : k3;
      fwkerind1 = (pivot1 >= 0) ? ms - k1 : k1;
      fwkerind2 = (pivot2 >= 0) ? mt - k2 : k2;
      fwkerind3 = (pivot3 >= 0) ? mu - k3 : k3;
    }

    outidx       = w1 + w2 * nf1 + w3 * nf1 * nf2;
    kervalue     = fwkerhalf1[fwkerind1] * fwkerhalf2[fwkerind2] * fwkerhalf3[fwkerind3];
    fw[outidx].x = fk[inidx].x / kervalue;
    fw[outidx].y = fk[inidx].y / kervalue;
  }
}

template<typename T, int modeord>
int cudeconvolve1d(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    wrapper for deconvolution & amplication in 1D.

    Melody Shih 11/21/21
*/
{
  auto &stream = d_plan->stream;

  int ms           = d_plan->ms;
  int nf1          = d_plan->nf1;
  int nmodes       = ms;
  int maxbatchsize = d_plan->maxbatchsize;

  if (d_plan->spopts.spread_direction == 1) {
    for (int t = 0; t < blksize; t++) {
      deconvolve_1d<T, modeord><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
          ms, nf1, d_plan->fw + t * nf1, d_plan->fk + t * nmodes, d_plan->fwkerhalf1);
    }
  } else {
    checkCudaErrors(cudaMemsetAsync(
        d_plan->fw, 0, maxbatchsize * nf1 * sizeof(cuda_complex<T>), stream));
    for (int t = 0; t < blksize; t++) {
      amplify_1d<T, modeord><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
          ms, nf1, d_plan->fw + t * nf1, d_plan->fk + t * nmodes, d_plan->fwkerhalf1);
    }
  }
  return 0;
}

template<typename T, int modeord>
int cudeconvolve2d(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    wrapper for deconvolution & amplication in 2D.

    Melody Shih 07/25/19
*/
{
  auto &stream = d_plan->stream;

  int ms           = d_plan->ms;
  int mt           = d_plan->mt;
  int nf1          = d_plan->nf1;
  int nf2          = d_plan->nf2;
  int nmodes       = ms * mt;
  int maxbatchsize = d_plan->maxbatchsize;

  if (d_plan->spopts.spread_direction == 1) {
    for (int t = 0; t < blksize; t++) {
      deconvolve_2d<T, modeord><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
          ms, mt, nf1, nf2, d_plan->fw + t * nf1 * nf2, d_plan->fk + t * nmodes,
          d_plan->fwkerhalf1, d_plan->fwkerhalf2);
    }
  } else {
    checkCudaErrors(cudaMemsetAsync(
        d_plan->fw, 0, maxbatchsize * nf1 * nf2 * sizeof(cuda_complex<T>), stream));
    for (int t = 0; t < blksize; t++) {
      amplify_2d<T, modeord><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
          ms, mt, nf1, nf2, d_plan->fw + t * nf1 * nf2, d_plan->fk + t * nmodes,
          d_plan->fwkerhalf1, d_plan->fwkerhalf2);
    }
  }
  return 0;
}

template<typename T, int modeord>
int cudeconvolve3d(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    wrapper for deconvolution & amplication in 3D.

    Melody Shih 07/25/19
*/
{
  auto &stream = d_plan->stream;

  int ms           = d_plan->ms;
  int mt           = d_plan->mt;
  int mu           = d_plan->mu;
  int nf1          = d_plan->nf1;
  int nf2          = d_plan->nf2;
  int nf3          = d_plan->nf3;
  int nmodes       = ms * mt * mu;
  int maxbatchsize = d_plan->maxbatchsize;
  if (d_plan->spopts.spread_direction == 1) {
    for (int t = 0; t < blksize; t++) {
      deconvolve_3d<T, modeord><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
          ms, mt, mu, nf1, nf2, nf3, d_plan->fw + t * nf1 * nf2 * nf3,
          d_plan->fk + t * nmodes, d_plan->fwkerhalf1, d_plan->fwkerhalf2,
          d_plan->fwkerhalf3);
    }
  } else {
    checkCudaErrors(cudaMemsetAsync(
        d_plan->fw, 0, maxbatchsize * nf1 * nf2 * nf3 * sizeof(cuda_complex<T>), stream));
    for (int t = 0; t < blksize; t++) {
      amplify_3d<T, modeord><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
          ms, mt, mu, nf1, nf2, nf3, d_plan->fw + t * nf1 * nf2 * nf3,
          d_plan->fk + t * nmodes, d_plan->fwkerhalf1, d_plan->fwkerhalf2,
          d_plan->fwkerhalf3);
    }
  }
  return 0;
}

template int cudeconvolve1d<float, 0>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cudeconvolve1d<float, 1>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cudeconvolve1d<double, 0>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cudeconvolve1d<double, 1>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cudeconvolve2d<float, 0>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cudeconvolve2d<float, 1>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cudeconvolve2d<double, 0>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cudeconvolve2d<double, 1>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cudeconvolve3d<float, 0>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cudeconvolve3d<float, 1>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cudeconvolve3d<double, 0>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cudeconvolve3d<double, 1>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace deconvolve
} // namespace cufinufft
