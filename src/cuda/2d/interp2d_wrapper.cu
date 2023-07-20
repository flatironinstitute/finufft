#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <helper_cuda.h>

#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>

using namespace cufinufft::memtransfer;

#include "spreadinterp2d.cuh"

namespace cufinufft {
namespace spreadinterp {

template <typename T>
int cufinufft_interp2d(int nf1, int nf2, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, cuda_complex<T> *d_c,
                       cufinufft_plan_t<T> *d_plan)
/*
    This c function is written for only doing 2D interpolation. See
    test/interp2d_test.cu for usage.

    Melody Shih 07/25/19
    not allocate,transfer and free memories on gpu. Shih 09/24/20
*/
{
    d_plan->nf1 = nf1;
    d_plan->nf2 = nf2;
    d_plan->M = M;
    d_plan->maxbatchsize = 1;

    d_plan->kx = d_kx;
    d_plan->ky = d_ky;
    d_plan->c = d_c;
    d_plan->fw = d_fw;

    int ier;
    ier = allocgpumem2d_plan<T>(d_plan);
    ier = allocgpumem2d_nupts<T>(d_plan);

    if (d_plan->opts.gpu_method == 1) {
        ier = cuspread2d_nuptsdriven_prop<T>(nf1, nf2, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread2d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }
    if (d_plan->opts.gpu_method == 2) {
        ier = cuspread2d_subprob_prop<T>(nf1, nf2, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread2d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    ier = cuinterp2d<T>(d_plan, 1);
    freegpumemory2d<T>(d_plan);

    return ier;
}

template <typename T>
int cuinterp2d(cufinufft_plan_t<T> *d_plan, int blksize)
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
    int M = d_plan->M;

    int ier;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        ier = cuinterp2d_nuptsdriven<T>(nf1, nf2, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread2d_gpu_nuptsdriven" << std::endl;
            return 1;
        }
    } break;
    case 2: {
        ier = cuinterp2d_subprob<T>(nf1, nf2, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cuinterp2d_subprob" << std::endl;
            return 1;
        }
    } break;
    default:
        std::cout << "error: incorrect method, should be 1 or 2" << std::endl;
        return 2;
    }

    return ier;
}

template <typename T>
int cuinterp2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    auto &stream = d_plan->streams[d_plan->curr_stream];

    dim3 threadsPerBlock;
    dim3 blocks;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    T sigma = d_plan->opts.upsampfac;
    int pirange = d_plan->spopts.pirange;
    int *d_idxnupts = d_plan->idxnupts;

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    blocks.x = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = 1;

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            interp_2d_nupts_driven_horner<<<blocks, threadsPerBlock, 0, stream>>>(
                d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, sigma, d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            interp_2d_nupts_driven<<<blocks, threadsPerBlock, 0, stream>>>(
                d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta, d_idxnupts, pirange);
        }
    }

    return 0;
}

template <typename T>
int cuinterp2d_subprob(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    auto &stream = d_plan->streams[d_plan->curr_stream];

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

    // assume that bin_size_x > ns/2;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int numbins[2];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;
    int totalnumsubprob = d_plan->totalnumsubprob;
    int pirange = d_plan->spopts.pirange;

    T sigma = d_plan->opts.upsampfac;
    size_t sharedplanorysize =
        (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) * sizeof(cuda_complex<T>);

    if (sharedplanorysize > 49152) {
        std::cout << "error: not enough shared memory" << std::endl;
        return 1;
    }

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            interp_2d_subprob_horner<<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
                d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, sigma, d_binstartpts, d_binsize,
                bin_size_x, bin_size_y, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins[0],
                numbins[1], d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            interp_2d_subprob<<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
                d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta, sigma, d_binstartpts,
                d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
                numbins[0], numbins[1], d_idxnupts, pirange);
        }
    }

    return 0;
}

template int cufinufft_interp2d(int nf1, int nf2, cuda_complex<float> *d_fw, int M, float *d_kx, float *d_ky,
                                cuda_complex<float> *d_c, cufinufft_plan_t<float> *d_plan);
template int cufinufft_interp2d(int nf1, int nf2, cuda_complex<double> *d_fw, int M, double *d_kx, double *d_ky,
                                cuda_complex<double> *d_c, cufinufft_plan_t<double> *d_plan);

template int cuinterp2d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp2d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft
