#include <cassert>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cufinufft/memtransfer.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>

using namespace cufinufft::common;
using namespace cufinufft::memtransfer;

namespace cufinufft {
namespace spreadinterp {

int CUFINUFFT_SPREAD1D(int nf1, CUCPX *d_fw, int M, CUFINUFFT_FLT *d_kx, CUCPX *d_c, CUFINUFFT_PLAN d_plan)
/*
    This c function is written for only doing 1D spreading. See
    test/spread1d_test.cu for usage.

    note: not allocate,transfer and free memories on gpu.
    Melody Shih 11/21/21
*/
{
    d_plan->kx = d_kx;
    d_plan->c = d_c;
    d_plan->fw = d_fw;

    int ier;
    d_plan->nf1 = nf1;
    d_plan->M = M;
    d_plan->maxbatchsize = 1;

    ier = ALLOCGPUMEM1D_PLAN(d_plan);
    ier = ALLOCGPUMEM1D_NUPTS(d_plan);

    if (d_plan->opts.gpu_method == 1) {
        ier = CUSPREAD1D_NUPTSDRIVEN_PROP(nf1, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread1d_nuptsdriven_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    if (d_plan->opts.gpu_method == 2) {
        ier = CUSPREAD1D_SUBPROB_PROP(nf1, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread1d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    ier = CUSPREAD1D(d_plan, 1);
    FREEGPUMEMORY1D(d_plan);

    return ier;
}

int CUSPREAD1D(CUFINUFFT_PLAN d_plan, int blksize)
/*
    A wrapper for different spreading methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem

    Melody Shih 11/21/21
*/
{
    int nf1 = d_plan->nf1;
    int M = d_plan->M;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int ier;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        cudaEventRecord(start);
        ier = CUSPREAD1D_NUPTSDRIVEN(nf1, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread1d_gpu_nuptsdriven" << std::endl;
            return 1;
        }
    } break;
    case 2: {
        cudaEventRecord(start);
        ier = CUSPREAD1D_SUBPROB(nf1, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread1d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    default:
        std::cout << "error: incorrect method, should be 1,2" << std::endl;
        return 2;
    }
    return ier;
}

int CUSPREAD1D_NUPTSDRIVEN_PROP(int nf1, int M, CUFINUFFT_PLAN d_plan) {
    if (d_plan->opts.gpu_sort) {
        int bin_size_x = d_plan->opts.gpu_binsizex;
        if (bin_size_x < 0) {
            std::cout << "error: invalid binsize (binsizex) = (" << bin_size_x << ")" << std::endl;
            return 1;
        }

        int numbins = ceil((CUFINUFFT_FLT)nf1 / bin_size_x);

        CUFINUFFT_FLT *d_kx = d_plan->kx;
        int *d_binsize = d_plan->binsize;
        int *d_binstartpts = d_plan->binstartpts;
        int *d_sortidx = d_plan->sortidx;
        int *d_idxnupts = d_plan->idxnupts;

        int pirange = d_plan->spopts.pirange;

        checkCudaErrors(cudaMemset(d_binsize, 0, numbins * sizeof(int)));
        calc_bin_size_noghost_1d<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, bin_size_x, numbins, d_binsize, d_kx,
                                                                  d_sortidx, pirange);

        int n = numbins;
        thrust::device_ptr<int> d_ptr(d_binsize);
        thrust::device_ptr<int> d_result(d_binstartpts);
        thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);

        calc_inverse_of_global_sort_idx_1d<<<(M + 1024 - 1) / 1024, 1024>>>(M, bin_size_x, numbins, d_binstartpts,
                                                                            d_sortidx, d_kx, d_idxnupts, pirange, nf1);
    } else {
        int *d_idxnupts = d_plan->idxnupts;
        trivial_global_sort_index_1d<<<(M + 1024 - 1) / 1024, 1024>>>(M, d_idxnupts);
    }

    return 0;
}

int CUSPREAD1D_NUPTSDRIVEN(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize) {
    dim3 threadsPerBlock;
    dim3 blocks;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    int pirange = d_plan->spopts.pirange;
    int *d_idxnupts = d_plan->idxnupts;
    CUFINUFFT_FLT es_c = d_plan->spopts.ES_c;
    CUFINUFFT_FLT es_beta = d_plan->spopts.ES_beta;
    CUFINUFFT_FLT sigma = d_plan->spopts.upsampfac;

    CUFINUFFT_FLT *d_kx = d_plan->kx;
    CUCPX *d_c = d_plan->c;
    CUCPX *d_fw = d_plan->fw;

    threadsPerBlock.x = 16;
    threadsPerBlock.y = 1;
    blocks.x = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = 1;

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            spread_1d_nuptsdriven_horner<<<blocks, threadsPerBlock>>>(d_kx, d_c + t * M, d_fw + t * nf1, M, ns, nf1,
                                                                      sigma, d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            spread_1d_nuptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_c + t * M, d_fw + t * nf1, M, ns, nf1, es_c,
                                                               es_beta, d_idxnupts, pirange);
        }
    }

    return 0;
}

int CUSPREAD1D_SUBPROB_PROP(int nf1, int M, CUFINUFFT_PLAN d_plan)
/*
    This function determines the properties for spreading that are independent
    of the strength of the nodes,  only relates to the locations of the nodes,
    which only needs to be done once.
*/
{
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    if (bin_size_x < 0) {
        std::cout << "error: invalid binsize (binsizex) = (";
        std::cout << bin_size_x << ")" << std::endl;
        return 1;
    }
    int numbins = ceil((CUFINUFFT_FLT)nf1 / bin_size_x);

    CUFINUFFT_FLT *d_kx = d_plan->kx;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx = d_plan->sortidx;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int *d_subprob_to_bin = NULL;

    int pirange = d_plan->spopts.pirange;

    checkCudaErrors(cudaMemset(d_binsize, 0, numbins * sizeof(int)));
    calc_bin_size_noghost_1d<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, bin_size_x, numbins, d_binsize, d_kx, d_sortidx,
                                                              pirange);

    int n = numbins;
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);

    calc_inverse_of_global_sort_idx_1d<<<(M + 1024 - 1) / 1024, 1024>>>(M, bin_size_x, numbins, d_binstartpts,
                                                                        d_sortidx, d_kx, d_idxnupts, pirange, nf1);

    calc_sub_prob_1d<<<(M + 1024 - 1) / 1024, 1024>>>(d_binsize, d_numsubprob, maxsubprobsize, numbins);

    d_ptr = thrust::device_pointer_cast(d_numsubprob);
    d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemset(d_subprobstartpts, 0, sizeof(int)));

    int totalnumsubprob;
    checkCudaErrors(cudaMemcpy(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMalloc(&d_subprob_to_bin, totalnumsubprob * sizeof(int)));
    map_b_into_sub_prob_1d<<<(numbins + 1024 - 1) / 1024, 1024>>>(d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
                                                                  numbins);
    assert(d_subprob_to_bin != NULL);
    if (d_plan->subprob_to_bin != NULL)
        cudaFree(d_plan->subprob_to_bin);
    d_plan->subprob_to_bin = d_subprob_to_bin;
    assert(d_plan->subprob_to_bin != NULL);
    d_plan->totalnumsubprob = totalnumsubprob;

    return 0;
}

int CUSPREAD1D_SUBPROB(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize) {
    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    CUFINUFFT_FLT es_c = d_plan->spopts.ES_c;
    CUFINUFFT_FLT es_beta = d_plan->spopts.ES_beta;
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

    // assume that bin_size_x > ns/2;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int numbins = ceil((CUFINUFFT_FLT)nf1 / bin_size_x);

    CUFINUFFT_FLT *d_kx = d_plan->kx;
    CUCPX *d_c = d_plan->c;
    CUCPX *d_fw = d_plan->fw;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int totalnumsubprob = d_plan->totalnumsubprob;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;

    int pirange = d_plan->spopts.pirange;

    CUFINUFFT_FLT sigma = d_plan->opts.upsampfac;

    size_t sharedplanorysize = (bin_size_x + 2 * (int)ceil(ns / 2.0)) * sizeof(CUCPX);
    if (sharedplanorysize > 49152) {
        std::cout << "error: not enough shared memory" << std::endl;
        return 1;
    }

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            spread_1d_subprob_horner<<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_c + t * M, d_fw + t * nf1, M, ns, nf1, sigma, d_binstartpts, d_binsize, bin_size_x,
                d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins, d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            spread_1d_subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_c + t * M, d_fw + t * nf1, M, ns, nf1, es_c, es_beta, sigma, d_binstartpts, d_binsize,
                bin_size_x, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins, d_idxnupts,
                pirange);
        }
    }

    return 0;
}

} // namespace spreadinterp
} // namespace cufinufft
