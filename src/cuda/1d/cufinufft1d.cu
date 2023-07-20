#include <cmath>
#include <complex>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <type_traits>

#include <assert.h>
#include <cufft.h>

#include <cufinufft/cudeconvolve.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>

using namespace cufinufft::deconvolve;
using namespace cufinufft::spreadinterp;

template <typename T>
int cufinufft1d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan)
/*
    1D Type-1 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: spread data to oversampled regular mesh using kernel
        Step 2: compute FFT on uniform mesh
        Step 3: deconvolve by division of each Fourier mode independently by the
                Fourier series coefficient of the kernel.

    Melody Shih 11/21/21
*/
{
    assert(d_plan->spopts.spread_direction == 1);
    auto &stream = d_plan->streams[d_plan->curr_stream];

    int blksize;
    int ier;
    cuda_complex<T> *d_fkstart;
    cuda_complex<T> *d_cstart;
    for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
        blksize = std::min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
        d_cstart = d_c + i * d_plan->maxbatchsize * d_plan->M;
        d_fkstart = d_fk + i * d_plan->maxbatchsize * d_plan->ms;
        d_plan->c = d_cstart;
        d_plan->fk = d_fkstart;

        checkCudaErrors(cudaMemsetAsync(d_plan->fw, 0, d_plan->maxbatchsize * d_plan->nf1 * sizeof(cuda_complex<T>),
                                        stream)); // this is needed

        // Step 1: Spread
        ier = cuspread1d<T>(d_plan, blksize);

        if (ier != 0) {
            printf("error: cuspread1d, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }

        // Step 2: FFT
        cufft_ex(d_plan->fftplans[d_plan->curr_stream], d_plan->fw, d_plan->fw, d_plan->iflag);

        // Step 3: deconvolve and shuffle
        cudeconvolve1d<T>(d_plan, blksize);
    }

    return ier;
}

template <typename T>
int cufinufft1d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan)
/*
    1D Type-2 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel
                Fourier coeff
        Step 2: compute FFT on uniform mesh
        Step 3: interpolate data to regular mesh

    Melody Shih 11/21/21
*/
{
    assert(d_plan->spopts.spread_direction == 2);

    int blksize;
    int ier;
    cuda_complex<T> *d_fkstart;
    cuda_complex<T> *d_cstart;
    for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
        blksize = std::min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
        d_cstart = d_c + i * d_plan->maxbatchsize * d_plan->M;
        d_fkstart = d_fk + i * d_plan->maxbatchsize * d_plan->ms;

        d_plan->c = d_cstart;
        d_plan->fk = d_fkstart;

        // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
        cudeconvolve1d<T>(d_plan, blksize);

        // Step 2: FFT
        cufft_ex(d_plan->fftplans[d_plan->curr_stream], d_plan->fw, d_plan->fw, d_plan->iflag);

        // Step 3: deconvolve and shuffle
        ier = cuinterp1d<T>(d_plan, blksize);

        if (ier != 0) {
            printf("error: cuinterp1d, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }
    return ier;
}

template int cufinufft1d1_exec<float>(cuda_complex<float> *d_c, cuda_complex<float> *d_fk,
                                      cufinufft_plan_t<float> *d_plan);
template int cufinufft1d1_exec<double>(cuda_complex<double> *d_c, cuda_complex<double> *d_fk,
                                       cufinufft_plan_t<double> *d_plan);
template int cufinufft1d2_exec<float>(cuda_complex<float> *d_c, cuda_complex<float> *d_fk,
                                      cufinufft_plan_t<float> *d_plan);
template int cufinufft1d2_exec<double>(cuda_complex<double> *d_c, cuda_complex<double> *d_fk,
                                       cufinufft_plan_t<double> *d_plan);
