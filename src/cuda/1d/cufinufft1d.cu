#include <complex>
#include <cufft.h>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <math.h>

#include <cufinufft/cudeconvolve.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft_eitherprec.h>

using namespace cufinufft::deconvolve;
using namespace cufinufft::spreadinterp;
using std::min;

int CUFINUFFT1D1_EXEC(CUCPX *d_c, CUCPX *d_fk, CUFINUFFT_PLAN d_plan)
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

    int blksize;
    int ier;
    CUCPX *d_fkstart;
    CUCPX *d_cstart;
    for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
        blksize = std::min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
        d_cstart = d_c + i * d_plan->maxbatchsize * d_plan->M;
        d_fkstart = d_fk + i * d_plan->maxbatchsize * d_plan->ms;
        d_plan->c = d_cstart;
        d_plan->fk = d_fkstart;

        checkCudaErrors(
            cudaMemset(d_plan->fw, 0, d_plan->maxbatchsize * d_plan->nf1 * sizeof(CUCPX))); // this is needed

        // Step 1: Spread
        ier = CUSPREAD1D(d_plan, blksize);
        if (ier != 0) {
            printf("error: cuspread1d, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }

        // Step 2: FFT
        CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);

        // Step 3: deconvolve and shuffle
        CUDECONVOLVE1D(d_plan, blksize);
    }

    return ier;
}

int CUFINUFFT1D2_EXEC(CUCPX *d_c, CUCPX *d_fk, CUFINUFFT_PLAN d_plan)
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
    CUCPX *d_fkstart;
    CUCPX *d_cstart;
    for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
        blksize = std::min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
        d_cstart = d_c + i * d_plan->maxbatchsize * d_plan->M;
        d_fkstart = d_fk + i * d_plan->maxbatchsize * d_plan->ms;

        d_plan->c = d_cstart;
        d_plan->fk = d_fkstart;

        // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
        CUDECONVOLVE1D(d_plan, blksize);

        // Step 2: FFT
        cudaDeviceSynchronize();
        CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);

        // Step 3: deconvolve and shuffle
        ier = CUINTERP1D(d_plan, blksize);
        if (ier != 0) {
            printf("error: cuinterp1d, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }
    return ier;
}
