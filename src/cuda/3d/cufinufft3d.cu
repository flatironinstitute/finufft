#include <cmath>
#include <complex>

#include <cufft.h>
#include <cufinufft/contrib/helper_cuda.h>

#include <cufinufft/cudeconvolve.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>

using namespace cufinufft::deconvolve;
using namespace cufinufft::spreadinterp;
using std::min;

template<typename T>
int cufinufft3d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan)
/*
    3D Type-1 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: spread data to oversampled regular mesh using kernel
        Step 2: compute FFT on uniform mesh
        Step 3: deconvolve by division of each Fourier mode independently by the
                Fourier series coefficient of the kernel.

    Melody Shih 07/25/19
*/
{
  auto &stream = d_plan->stream;
  int ier;
  cuda_complex<T> *d_fkstart;
  cuda_complex<T> *d_cstart;
  for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
    int blksize = min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
    d_cstart    = d_c + i * d_plan->maxbatchsize * d_plan->M;
    d_fkstart   = d_fk + i * d_plan->maxbatchsize * d_plan->ms * d_plan->mt * d_plan->mu;

    d_plan->c  = d_cstart;
    d_plan->fk = d_fkstart;

    if ((ier = checkCudaErrors(
             cudaMemsetAsync(d_plan->fw, 0,
                             d_plan->maxbatchsize * d_plan->nf1 * d_plan->nf2 *
                                 d_plan->nf3 * sizeof(cuda_complex<T>),
                             stream))))
      return ier;

    // Step 1: Spread
    if ((ier = cuspread3d<T>(d_plan, blksize))) return ier;

    // Step 2: FFT
    cufftResult cufft_status =
        cufft_ex(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
    if (cufft_status != CUFFT_SUCCESS) return FINUFFT_ERR_CUDA_FAILURE;

    // Step 3: deconvolve and shuffle
    if (d_plan->opts.modeord == 0) {
      if ((ier = cudeconvolve3d<T, 0>(d_plan, blksize))) return ier;
    } else {
      if ((ier = cudeconvolve3d<T, 1>(d_plan, blksize))) return ier;
    }
  }

  return 0;
}

template<typename T>
int cufinufft3d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan)
/*
    3D Type-2 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel
                Fourier coeff
        Step 2: compute FFT on uniform mesh
        Step 3: interpolate data to regular mesh

    Melody Shih 07/25/19
*/
{
  int ier;
  cuda_complex<T> *d_fkstart;
  cuda_complex<T> *d_cstart;
  for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
    int blksize = min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
    d_cstart    = d_c + i * d_plan->maxbatchsize * d_plan->M;
    d_fkstart   = d_fk + i * d_plan->maxbatchsize * d_plan->ms * d_plan->mt * d_plan->mu;

    d_plan->c  = d_cstart;
    d_plan->fk = d_fkstart;

    // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
    if (d_plan->opts.modeord == 0) {
      if ((ier = cudeconvolve3d<T, 0>(d_plan, blksize))) return ier;
    } else {
      if ((ier = cudeconvolve3d<T, 1>(d_plan, blksize))) return ier;
    }

    // Step 2: FFT
    RETURN_IF_CUDA_ERROR
    cufftResult cufft_status =
        cufft_ex(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
    if (cufft_status != CUFFT_SUCCESS) return FINUFFT_ERR_CUDA_FAILURE;

    // Step 3: deconvolve and shuffle
    if ((ier = cuinterp3d<T>(d_plan, blksize))) return ier;
  }

  return 0;
}

template int cufinufft3d1_exec<float>(cuda_complex<float> *d_c, cuda_complex<float> *d_fk,
                                      cufinufft_plan_t<float> *d_plan);
template int cufinufft3d1_exec<double>(cuda_complex<double> *d_c,
                                       cuda_complex<double> *d_fk,
                                       cufinufft_plan_t<double> *d_plan);

template int cufinufft3d2_exec<float>(cuda_complex<float> *d_c, cuda_complex<float> *d_fk,
                                      cufinufft_plan_t<float> *d_plan);
template int cufinufft3d2_exec<double>(cuda_complex<double> *d_c,
                                       cuda_complex<double> *d_fk,
                                       cufinufft_plan_t<double> *d_plan);
