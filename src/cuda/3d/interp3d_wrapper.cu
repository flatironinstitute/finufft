#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <helper_cuda.h>

#include <cufinufft/memtransfer.h>
#include <cufinufft/profile.h>
#include <cufinufft/spreadinterp.h>
using namespace cufinufft::memtransfer;

namespace cufinufft {
namespace spreadinterp {

int CUFINUFFT_INTERP3D(int nf1, int nf2, int nf3, CUCPX *d_fw, int M, CUFINUFFT_FLT *d_kx, CUFINUFFT_FLT *d_ky,
                       CUFINUFFT_FLT *d_kz, CUCPX *d_c, CUFINUFFT_PLAN d_plan)
/*
    This c function is written for only doing 3D interpolation. See
    test/interp3d_test.cu for usage.

    Melody Shih 07/25/19
    not allocate,transfer and free memories on gpu. Shih 09/24/20
*/
{
    int ier;
    d_plan->kx = d_kx;
    d_plan->ky = d_ky;
    d_plan->kz = d_kz;
    d_plan->c = d_c;
    d_plan->fw = d_fw;

    d_plan->nf1 = nf1;
    d_plan->nf2 = nf2;
    d_plan->nf3 = nf3;
    d_plan->M = M;
    d_plan->maxbatchsize = 1;

    ier = ALLOCGPUMEM3D_PLAN(d_plan);
    ier = ALLOCGPUMEM3D_NUPTS(d_plan);

    if (d_plan->opts.gpu_method == 1) {
        ier = CUSPREAD3D_NUPTSDRIVEN_PROP(nf1, nf2, nf3, M, d_plan);
        if (ier != 0) {
            printf("error: cuinterp3d_nuptsdriven_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }
    if (d_plan->opts.gpu_method == 2) {
        ier = CUSPREAD3D_SUBPROB_PROP(nf1, nf2, nf3, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread3d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    ier = CUINTERP3D(d_plan, 1);

    FREEGPUMEMORY3D(d_plan);

    return ier;
}

int CUINTERP3D(CUFINUFFT_PLAN d_plan, int blksize)
/*
    A wrapper for different interpolation methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem

    Melody Shih 07/25/19
*/
{
    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int nf3 = d_plan->nf3;
    int M = d_plan->M;

    int ier;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        ier = CUINTERP3D_NUPTSDRIVEN(nf1, nf2, nf3, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread3d_gpu_nuptsdriven" << std::endl;
            return 1;
        }
    } break;
    case 2: {
        ier = CUINTERP3D_SUBPROB(nf1, nf2, nf3, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread3d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    default:
        std::cout << "error: incorrect method, should be 1,2" << std::endl;
        return 2;
    }

    return ier;
}

int CUINTERP3D_NUPTSDRIVEN(int nf1, int nf2, int nf3, int M, CUFINUFFT_PLAN d_plan, int blksize) {
    dim3 threadsPerBlock;
    dim3 blocks;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    CUFINUFFT_FLT es_c = d_plan->spopts.ES_c;
    CUFINUFFT_FLT es_beta = d_plan->spopts.ES_beta;
    CUFINUFFT_FLT sigma = d_plan->spopts.upsampfac;
    int pirange = d_plan->spopts.pirange;

    int *d_idxnupts = d_plan->idxnupts;

    CUFINUFFT_FLT *d_kx = d_plan->kx;
    CUFINUFFT_FLT *d_ky = d_plan->ky;
    CUFINUFFT_FLT *d_kz = d_plan->kz;
    CUCPX *d_c = d_plan->c;
    CUCPX *d_fw = d_plan->fw;

    threadsPerBlock.x = 16;
    threadsPerBlock.y = 1;
    blocks.x = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = 1;

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            interp_3d_nupts_driven_horner<<<blocks, threadsPerBlock, 0, 0>>>(d_kx, d_ky, d_kz, d_c + t * M,
                                                                             d_fw + t * nf1 * nf2 * nf3, M, ns, nf1,
                                                                             nf2, nf3, sigma, d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            interp_3d_nupts_driven<<<blocks, threadsPerBlock, 0, 0>>>(d_kx, d_ky, d_kz, d_c + t * M,
                                                                      d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3,
                                                                      es_c, es_beta, d_idxnupts, pirange);
        }
    }

    return 0;
}

int CUINTERP3D_SUBPROB(int nf1, int nf2, int nf3, int M, CUFINUFFT_PLAN d_plan, int blksize) {
    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

    // assume that bin_size_x > ns/2;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int bin_size_z = d_plan->opts.gpu_binsizez;
    int numbins[3];
    numbins[0] = ceil((CUFINUFFT_FLT)nf1 / bin_size_x);
    numbins[1] = ceil((CUFINUFFT_FLT)nf2 / bin_size_y);
    numbins[2] = ceil((CUFINUFFT_FLT)nf3 / bin_size_z);

    CUFINUFFT_FLT *d_kx = d_plan->kx;
    CUFINUFFT_FLT *d_ky = d_plan->ky;
    CUFINUFFT_FLT *d_kz = d_plan->kz;
    CUCPX *d_c = d_plan->c;
    CUCPX *d_fw = d_plan->fw;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;
    int totalnumsubprob = d_plan->totalnumsubprob;

    CUFINUFFT_FLT sigma = d_plan->spopts.upsampfac;
    CUFINUFFT_FLT es_c = d_plan->spopts.ES_c;
    CUFINUFFT_FLT es_beta = d_plan->spopts.ES_beta;
    int pirange = d_plan->spopts.pirange;
    size_t sharedplanorysize = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) *
                               (bin_size_z + 2 * ceil(ns / 2.0)) * sizeof(CUCPX);
    if (sharedplanorysize > 49152) {
        std::cout << "error: not enough shared memory" << std::endl;
        return 1;
    }

    for (int t = 0; t < blksize; t++) {
        if (d_plan->opts.gpu_kerevalmeth == 1) {
            interp_3d_subprob_horner<<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, sigma, d_binstartpts,
                d_binsize, bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
                maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, pirange);
        } else {
            interp_3d_subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c, es_beta,
                d_binstartpts, d_binsize, bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts,
                d_numsubprob, maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, pirange);
        }
    }

    return 0;
}

} // namespace spreadinterp
} // namespace cufinufft
