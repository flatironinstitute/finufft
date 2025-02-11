#include <cassert>
#include <cmath>
#include <complex>
#include <cufft.h>

#include <thrust/extrema.h>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

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
  for (int i = 0; i * d_plan->batchsize < d_plan->ntransf; i++) {
    int blksize = min(d_plan->ntransf - i * d_plan->batchsize, d_plan->batchsize);
    d_cstart    = d_c + i * d_plan->batchsize * d_plan->M;
    d_fkstart   = d_fk + i * d_plan->batchsize * d_plan->ms * d_plan->mt;
    d_plan->c   = d_cstart;
    d_plan->fk  = d_fkstart;  // so deconvolve will write into user output f
    if (d_plan->opts.gpu_spreadinterponly)
      d_plan->fw = d_fkstart; // spread directly into user output f

    // this is needed
    if ((ier = checkCudaErrors(cudaMemsetAsync(
             d_plan->fw, 0,
             d_plan->batchsize * d_plan->nf1 * d_plan->nf2 * sizeof(cuda_complex<T>),
             stream))))
      return ier;

    // Step 1: Spread
    if ((ier = cuspread2d<T>(d_plan, blksize))) return ier;

    if (d_plan->opts.gpu_spreadinterponly) continue; // skip steps 2 and 3

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
  for (int i = 0; i * d_plan->batchsize < d_plan->ntransf; i++) {
    int blksize = min(d_plan->ntransf - i * d_plan->batchsize, d_plan->batchsize);
    d_cstart    = d_c + i * d_plan->batchsize * d_plan->M;
    d_fkstart   = d_fk + i * d_plan->batchsize * d_plan->ms * d_plan->mt;

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
      d_plan->fw = d_fkstart; // interpolate directly from user input f

    // Step 3: Interpolate
    if ((ier = cuinterp2d<T>(d_plan, blksize))) return ier;
  }

  return 0;
}

template<typename T>
int cufinufft2d3_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan) {
  /*
    2D Type-3 NUFFT

  This function is called in "exec" stage (See ../cufinufft.cu).
  It includes (copied from doc in finufft library)
    Step 0: pre-phase the input strengths
    Step 1: spread data
    Step 2: Type 2 NUFFT
    Step 3: deconvolve (amplify) each Fourier mode, using kernel Fourier coeff

  Marco Barbone 08/14/2024
  */
  int ier;
  cuda_complex<T> *d_cstart;
  cuda_complex<T> *d_fkstart;
  const auto stream = d_plan->stream;
  for (int i = 0; i * d_plan->batchsize < d_plan->ntransf; i++) {
    int blksize = min(d_plan->ntransf - i * d_plan->batchsize, d_plan->batchsize);
    d_cstart    = d_c + i * d_plan->batchsize * d_plan->M;
    d_fkstart   = d_fk + i * d_plan->batchsize * d_plan->N;
    // setting input for spreader
    d_plan->c = d_plan->CpBatch;
    // setting output for spreader
    d_plan->fk = d_plan->fw;
    if ((ier = checkCudaErrors(cudaMemsetAsync(
             d_plan->fw, 0, d_plan->batchsize * d_plan->nf * sizeof(cuda_complex<T>),
             stream))))
      return ier;
    // NOTE: fw might need to be set to 0
    // Step 0: pre-phase the input strengths
    for (int i = 0; i < blksize; i++) {
      thrust::transform(thrust::cuda::par.on(stream), d_plan->prephase,
                        d_plan->prephase + d_plan->M, d_cstart + i * d_plan->M,
                        d_plan->c + i * d_plan->M, thrust::multiplies<cuda_complex<T>>());
    }
    // Step 1: Spread
    if ((ier = cuspread2d<T>(d_plan, blksize))) return ier;
    // now d_plan->fk = d_plan->fw contains the spread values
    // Step 2: Type 2 NUFFT
    // type 2 goes from fk to c
    // saving the results directly in the user output array d_fk
    // it needs to do blksize transforms
    d_plan->t2_plan->ntransf = blksize;
    if ((ier = cufinufft2d2_exec<T>(d_fkstart, d_plan->fw, d_plan->t2_plan))) return ier;
    // Step 3: deconvolve
    // now we need to d_fk = d_fk*d_plan->deconv
    for (int i = 0; i < blksize; i++) {
      thrust::transform(thrust::cuda::par.on(stream), d_plan->deconv,
                        d_plan->deconv + d_plan->N, d_fkstart + i * d_plan->N,
                        d_fkstart + i * d_plan->N, thrust::multiplies<cuda_complex<T>>());
    }
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
template int cufinufft2d3_exec<float>(cuda_complex<float> *d_c, cuda_complex<float> *d_fk,
                                      cufinufft_plan_t<float> *d_plan);
template int cufinufft2d3_exec<double>(cuda_complex<double> *d_c,
                                       cuda_complex<double> *d_fk,
                                       cufinufft_plan_t<double> *d_plan);
