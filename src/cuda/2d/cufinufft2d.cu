#include <cassert>
#include <cmath>
#include <complex>
#include <cufft.h>
#include <cufinufft/contrib/helper_cuda.h>

#include <cufinufft/cudeconvolve.h>
#include <cufinufft/spreadinterp.h>

using namespace cufinufft::deconvolve;
using namespace cufinufft::spreadinterp;
using std::min;

template<typename T>
int cufinufft2d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan)
/*
    2D Type-1 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: spread data to oversampled regular mesh using kernel
        Step 2: compute FFT on uniform mesh
        Step 3: deconvolve by division of each Fourier mode independently by the
                Fourier series coefficient of the kernel.

    Melody Shih 07/25/19
*/
{
  assert(d_plan->spopts.spread_direction == 1);

  int ier;
  cuda_complex<T> *d_fkstart;
  cuda_complex<T> *d_cstart;

  auto &stream = d_plan->stream;
  for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
    int blksize = min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
    d_cstart    = d_c + i * d_plan->maxbatchsize * d_plan->M;
    d_fkstart   = d_fk + i * d_plan->maxbatchsize * d_plan->ms * d_plan->mt;
    d_plan->c   = d_cstart;
    d_plan->fk  = d_fkstart;
    if (d_plan->opts.gpu_spreadinterponly) d_plan->fw = d_fkstart;

    // this is needed
    if ((ier = checkCudaErrors(cudaMemsetAsync(
             d_plan->fw, 0,
             d_plan->maxbatchsize * d_plan->nf1 * d_plan->nf2 * sizeof(cuda_complex<T>),
             stream))))
      return ier;

    // Step 1: Spread
    if ((ier = cuspread2d<T>(d_plan, blksize))) return ier;

    // Step 1.5: if spreadonly, skip the rest
    if (d_plan->opts.gpu_spreadinterponly) continue;

    // Step 2: FFT
    cufftResult cufft_status =
        cufft_ex(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
    if (cufft_status != CUFFT_SUCCESS) return FINUFFT_ERR_CUDA_FAILURE;

    // Step 3: deconvolve and shuffle
    if (d_plan->opts.modeord == 0) {
      if ((ier = cudeconvolve2d<T, 0>(d_plan, blksize))) return ier;
    } else {
      if ((ier = cudeconvolve2d<T, 1>(d_plan, blksize))) return ier;
    }
  }

  return 0;
}

template<typename T>
int cufinufft2d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan)
/*
    2D Type-2 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel
                Fourier coeff
        Step 2: compute FFT on uniform mesh
        Step 3: interpolate data to regular mesh

    Melody Shih 07/25/19
*/
{
  assert(d_plan->spopts.spread_direction == 2);

  int ier;
  cuda_complex<T> *d_fkstart;
  cuda_complex<T> *d_cstart;
  for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
    int blksize = min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
    d_cstart    = d_c + i * d_plan->maxbatchsize * d_plan->M;
    d_fkstart   = d_fk + i * d_plan->maxbatchsize * d_plan->ms * d_plan->mt;

    d_plan->c  = d_cstart;
    d_plan->fk = d_fkstart;

    // Skip steps 1 and 2 if interponly
    if (!d_plan->opts.gpu_spreadinterponly) {
      // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
      if (d_plan->opts.modeord == 0) {
        if ((ier = cudeconvolve2d<T, 0>(d_plan, blksize))) return ier;
      } else {
        if ((ier = cudeconvolve2d<T, 1>(d_plan, blksize))) return ier;
      }

      // Step 2: FFT
      cufftResult cufft_status =
          cufft_ex(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
      if (cufft_status != CUFFT_SUCCESS) return FINUFFT_ERR_CUDA_FAILURE;
    } else
      d_plan->fw = d_fkstart;

    // Step 3: Interpolate
    if ((ier = cuinterp2d<T>(d_plan, blksize))) return ier;
  }

  return 0;
}

template int cufinufft2d1_exec<float>(cuda_complex<float> *d_c, cuda_complex<float> *d_fk,
                                      cufinufft_plan_t<float> *d_plan);
template int cufinufft2d1_exec<double>(cuda_complex<double> *d_c,
                                       cuda_complex<double> *d_fk,
                                       cufinufft_plan_t<double> *d_plan);
template int cufinufft2d2_exec<float>(cuda_complex<float> *d_c, cuda_complex<float> *d_fk,
                                      cufinufft_plan_t<float> *d_plan);
template int cufinufft2d2_exec<double>(cuda_complex<double> *d_c,
                                       cuda_complex<double> *d_fk,
                                       cufinufft_plan_t<double> *d_plan);
