#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>

#include <cufft.h>
#include <helper_cuda.h>

#include <cufinufft/cudeconvolve.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft_eitherprec.h>

using namespace cufinufft::deconvolve;
using namespace cufinufft::spreadinterp;
using std::min;

int CUFINUFFT3D1_EXEC(CUCPX *d_c, CUCPX *d_fk, CUFINUFFT_PLAN d_plan)
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
    int blksize;
    int ier;
    CUCPX *d_fkstart;
    CUCPX *d_cstart;
    for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
        blksize = min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
        d_cstart = d_c + i * d_plan->maxbatchsize * d_plan->M;
        d_fkstart = d_fk + i * d_plan->maxbatchsize * d_plan->ms * d_plan->mt * d_plan->mu;

        d_plan->c = d_cstart;
        d_plan->fk = d_fkstart;

        checkCudaErrors(
            cudaMemset(d_plan->fw, 0, d_plan->maxbatchsize * d_plan->nf1 * d_plan->nf2 * d_plan->nf3 * sizeof(CUCPX)));

        // Step 1: Spread
        ier = CUSPREAD3D(d_plan, blksize);
        if (ier != 0) {
            printf("error: cuspread3d, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }

        // Step 2: FFT
        CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);

        // Step 3: deconvolve and shuffle
        CUDECONVOLVE3D(d_plan, blksize);
    }

    return 0;
}

int CUFINUFFT3D2_EXEC(CUCPX *d_c, CUCPX *d_fk, CUFINUFFT_PLAN d_plan)
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
    int blksize;
    int ier;
    CUCPX *d_fkstart;
    CUCPX *d_cstart;
    for (int i = 0; i * d_plan->maxbatchsize < d_plan->ntransf; i++) {
        blksize = min(d_plan->ntransf - i * d_plan->maxbatchsize, d_plan->maxbatchsize);
        d_cstart = d_c + i * d_plan->maxbatchsize * d_plan->M;
        d_fkstart = d_fk + i * d_plan->maxbatchsize * d_plan->ms * d_plan->mt * d_plan->mu;

        d_plan->c = d_cstart;
        d_plan->fk = d_fkstart;

        // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
        CUDECONVOLVE3D(d_plan, blksize);

        // Step 2: FFT
        cudaDeviceSynchronize();
        CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);

        // Step 3: deconvolve and shuffle
        ier = CUINTERP3D(d_plan, blksize);
        if (ier != 0) {
            printf("error: cuinterp3d, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    return 0;
}
