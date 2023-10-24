#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>

#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>

#include "spreadinterp3d.cuh"

using namespace cufinufft::memtransfer;

namespace cufinufft {
namespace spreadinterp {

template <typename T>
int cuinterp3d(cufinufft_plan_t<T> *d_plan, int blksize)
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
        ier = cuinterp3d_nuptsdriven<T>(nf1, nf2, nf3, M, d_plan, blksize);
    } break;
    case 2: {
        ier = cuinterp3d_subprob<T>(nf1, nf2, nf3, M, d_plan, blksize);
    } break;
    default:
        std::cerr << "[cuinterp3d] error: incorrect method, should be 1,2\n";
        ier = FINUFFT_ERR_METHOD_NOTVALID;
    }

    return ier;
}

template <typename T>
int cuinterp3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    auto &stream = d_plan->stream;

    dim3 threadsPerBlock;
    dim3 blocks;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    T sigma = d_plan->spopts.upsampfac;
    int pirange = d_plan->spopts.pirange;

    int *d_idxnupts = d_plan->idxnupts;

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    const char *tpbstr = getenv("CUFINUFFT_INTERP3D_TPB");
    threadsPerBlock.x = tpbstr ? std::atoi(tpbstr) : 64;
    threadsPerBlock.y = 1;
    blocks.x = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = 1;

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            interp_3d_nupts_driven<T, 1>
                <<<blocks, threadsPerBlock, 0, stream>>>(d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M,
                                                         ns, nf1, nf2, nf3, es_c, es_beta, sigma, d_idxnupts, pirange);
            RETURN_IF_CUDA_ERROR
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            interp_3d_nupts_driven<T, 0>
                <<<blocks, threadsPerBlock, 0, stream>>>(d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M,
                                                         ns, nf1, nf2, nf3, es_c, es_beta, sigma, d_idxnupts, pirange);
            RETURN_IF_CUDA_ERROR
        }
    }

    return 0;
}

template <typename T>
int cuinterp3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    auto &stream = d_plan->stream;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

    // assume that bin_size_x > ns/2;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int bin_size_z = d_plan->opts.gpu_binsizez;
    int numbins[3];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);
    numbins[2] = ceil((T)nf3 / bin_size_z);

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;
    int totalnumsubprob = d_plan->totalnumsubprob;

    T sigma = d_plan->spopts.upsampfac;
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    int pirange = d_plan->spopts.pirange;
    size_t sharedplanorysize = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) *
                               (bin_size_z + 2 * ceil(ns / 2.0)) * sizeof(cuda_complex<T>);
    if (sharedplanorysize > 49152) {
        std::cerr << "[cuinterp3d_subprob] error: not enough shared memory\n";
        return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }

    for (int t = 0; t < blksize; t++) {
        if (d_plan->opts.gpu_kerevalmeth == 1) {
            interp_3d_subprob<T, 1><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c, es_beta, sigma,
                d_binstartpts, d_binsize, bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts,
                d_numsubprob, maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, pirange);
            RETURN_IF_CUDA_ERROR
        } else {
            interp_3d_subprob<T, 0><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c, es_beta, sigma,
                d_binstartpts, d_binsize, bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts,
                d_numsubprob, maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, pirange);
            RETURN_IF_CUDA_ERROR
        }
    }

    return 0;
}

template int cuinterp3d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp3d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

template int cuinterp3d_nuptsdriven<float>(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<float> *d_plan,
                                           int blksize);
template int cuinterp3d_nuptsdriven<double>(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<double> *d_plan,
                                            int blksize);

template int cuinterp3d_subprob<float>(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp3d_subprob<double>(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<double> *d_plan,
                                        int blksize);

} // namespace spreadinterp
} // namespace cufinufft
