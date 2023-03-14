#include <cassert>
#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cufinufft/memtransfer.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>

#include "spreadinterp2d.cuh"

using namespace cufinufft::common;
using namespace cufinufft::memtransfer;

namespace cufinufft {
namespace spreadinterp {

template <typename T>
int cufinufft_spread2d(int nf1, int nf2, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, cuda_complex<T> *d_c,
                       cufinufft_plan_template<T> d_plan)
/*
    This c function is written for only doing 2D spreading. See
    test/spread2d_test.cu for usage.

    Melody Shih 07/25/19
    not allocate,transfer and free memories on gpu. Shih 09/24/20
*/
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    d_plan->kx = d_kx;
    d_plan->ky = d_ky;
    d_plan->c = d_c;
    d_plan->fw = d_fw;

    int ier;
    d_plan->nf1 = nf1;
    d_plan->nf2 = nf2;
    d_plan->M = M;
    d_plan->maxbatchsize = 1;

    cudaEventRecord(start);
    ier = ALLOCGPUMEM2D_PLAN(d_plan);
    ier = ALLOCGPUMEM2D_NUPTS(d_plan);

    if (d_plan->opts.gpu_method == 1) {
        ier = CUSPREAD2D_NUPTSDRIVEN_PROP(nf1, nf2, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread2d_nuptsdriven_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    if (d_plan->opts.gpu_method == 2) {
        ier = CUSPREAD2D_SUBPROB_PROP(nf1, nf2, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread2d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

#ifdef TIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] Obtain Spread Prop\t %.3g ms\n", milliseconds);
#endif

    cudaEventRecord(start);
    ier = CUSPREAD2D(d_plan, 1);
#ifdef TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] Spread (%d)\t\t %5.3f ms\n", d_plan->opts.gpu_method, milliseconds);
#endif

    cudaEventRecord(start);
    FREEGPUMEMORY2D(d_plan);
#ifdef TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
    return ier;
}

template <typename T>
int cuspread2d(cufinufft_plan_template<T> d_plan, int blksize)
/*
    A wrapper for different spreading methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem

    Melody Shih 07/25/19
*/
{
    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int M = d_plan->M;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int ier;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        cudaEventRecord(start);
        ier = cuspread2d_nuptsdriven<T>(nf1, nf2, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread2d_gpu_nuptsdriven" << std::endl;
            return 1;
        }
    } break;
    case 2: {
        cudaEventRecord(start);
        ier = cuspread2d_subprob<T>(nf1, nf2, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread2d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    default:
        std::cout << "error: incorrect method, should be 1,2,3" << std::endl;
        return 2;
    }
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[time  ]"
              << " Spread " << milliseconds << " ms" << std::endl;
#endif
    return ier;
}

template <typename T>
int cuspread2d_nuptsdriven_prop(int nf1, int nf2, int M, cufinufft_plan_template<T> d_plan) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (d_plan->opts.gpu_sort) {

        int bin_size_x = d_plan->opts.gpu_binsizex;
        int bin_size_y = d_plan->opts.gpu_binsizey;
        if (bin_size_x < 0 || bin_size_y < 0) {
            std::cout << "error: invalid binsize (binsizex, binsizey) = (";
            std::cout << bin_size_x << "," << bin_size_y << ")" << std::endl;
            return 1;
        }

        int numbins[2];
        numbins[0] = ceil((T)nf1 / bin_size_x);
        numbins[1] = ceil((T)nf2 / bin_size_y);

#ifdef DEBUG
        std::cout << "[debug ] Dividing the uniform grids to bin size[" << d_plan->opts.gpu_binsizex << "x"
                  << d_plan->opts.gpu_binsizey << "]" << std::endl;
        std::cout << "[debug ] numbins = [" << numbins[0] << "x" << numbins[1] << "]" << std::endl;
#endif

        T *d_kx = d_plan->kx;
        T *d_ky = d_plan->ky;
#ifdef DEBUG
        T *h_kx;
        T *h_ky;
        h_kx = (T *)malloc(M * sizeof(T));
        h_ky = (T *)malloc(M * sizeof(T));

        checkCudaErrors(cudaMemcpy(h_kx, d_kx, M * sizeof(T), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_ky, d_ky, M * sizeof(T), cudaMemcpyDeviceToHost));
        for (int i = M - 10; i < M; i++) {
            std::cout << "[debug ] ";
            std::cout << "(" << setw(3) << h_kx[i] << "," << setw(3) << h_ky[i] << ")" << std::endl;
        }
#endif
        int *d_binsize = d_plan->binsize;
        int *d_binstartpts = d_plan->binstartpts;
        int *d_sortidx = d_plan->sortidx;
        int *d_idxnupts = d_plan->idxnupts;

        int pirange = d_plan->spopts.pirange;

        cudaEventRecord(start);
        checkCudaErrors(cudaMemset(d_binsize, 0, numbins[0] * numbins[1] * sizeof(int)));
        CalcBinSize_noghost_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0],
                                                                numbins[1], d_binsize, d_kx, d_ky, d_sortidx, pirange);
#ifdef SPREADTIME
        float milliseconds = 0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] \tKernel CalcBinSize_noghost_2d \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
        int *h_binsize; // For debug
        h_binsize = (int *)malloc(numbins[0] * numbins[1] * sizeof(int));
        checkCudaErrors(
            cudaMemcpy(h_binsize, d_binsize, numbins[0] * numbins[1] * sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "[debug ] bin size:" << std::endl;
        for (int j = 0; j < numbins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                std::cout << " bin[" << setw(1) << i << "," << setw(1) << j << "]=" << h_binsize[i + j * numbins[0]];
            }
            std::cout << std::endl;
        }
        free(h_binsize);
        std::cout << "[debug ] ------------------------------------------------" << std::endl;

        int *h_sortidx;
        h_sortidx = (int *)malloc(M * sizeof(int));

        checkCudaErrors(cudaMemcpy(h_sortidx, d_sortidx, M * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < M; i++) {
            if (h_sortidx[i] < 0) {
                std::cout << "[debug ] ";
                std::cout << "point[" << setw(3) << i << "]=" << setw(3) << h_sortidx[i] << std::endl;
                std::cout << "[debug ] ";
                printf("(%10.10f, %10.10f) ", RESCALE(h_kx[i], nf1, pirange), RESCALE(h_ky[i], nf1, pirange));
                printf("(%10.10f, %10.10f) ", RESCALE(h_kx[i], nf1, pirange) / 32, RESCALE(h_ky[i], nf1, pirange) / 32);
                printf("(%f, %f)\n", floor(RESCALE(h_kx[i], nf1, pirange) / 32),
                       floor(RESCALE(h_ky[i], nf1, pirange) / 32));
            }
        }
#endif
        cudaEventRecord(start);
        int n = numbins[0] * numbins[1];
        thrust::device_ptr<int> d_ptr(d_binsize);
        thrust::device_ptr<int> d_result(d_binstartpts);
        thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
#ifdef SPREADTIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] \tKernel BinStartPts_2d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
        int *h_binstartpts;
        h_binstartpts = (int *)malloc((numbins[0] * numbins[1]) * sizeof(int));
        checkCudaErrors(
            cudaMemcpy(h_binstartpts, d_binstartpts, (numbins[0] * numbins[1]) * sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "[debug ] Result of scan bin_size array:" << std::endl;
        for (int j = 0; j < numbins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                std::cout << " bin[" << setw(1) << i << "," << setw(1) << j
                          << "]=" << h_binstartpts[i + j * numbins[0]];
            }
            std::cout << std::endl;
        }
        free(h_binstartpts);
        std::cout << "[debug ] ------------------------------------------------" << std::endl;
#endif
        cudaEventRecord(start);
        CalcInvertofGlobalSortIdx_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, bin_size_x, bin_size_y, numbins[0], numbins[1],
                                                                      d_binstartpts, d_sortidx, d_kx, d_ky, d_idxnupts,
                                                                      pirange, nf1, nf2);
#ifdef SPREADTIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] \tKernel CalcInvertofGlobalSortIdx_2d \t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
        int *h_idxnupts;
        h_idxnupts = (int *)malloc(M * sizeof(int));
        checkCudaErrors(cudaMemcpy(h_idxnupts, d_idxnupts, M * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 10; i++) {
            std::cout << "[debug ] idx=" << h_idxnupts[i] << std::endl;
        }
        free(h_idxnupts);
#endif
    } else {
        int *d_idxnupts = d_plan->idxnupts;

        cudaEventRecord(start);
        TrivialGlobalSortIdx_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, d_idxnupts);
#ifdef SPREADTIME
        float milliseconds = 0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] \tKernel TrivialGlobalSortIDx_2d \t\t%.3g ms\n", milliseconds);
#endif
    }
    return 0;
}

template <typename T>
int cuspread2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_template<T> d_plan, int blksize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock;
    dim3 blocks;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    int pirange = d_plan->spopts.pirange;
    int *d_idxnupts = d_plan->idxnupts;
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    T sigma = d_plan->spopts.upsampfac;

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    threadsPerBlock.x = 16;
    threadsPerBlock.y = 1;
    blocks.x = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = 1;
    cudaEventRecord(start);
    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            Spread_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M,
                                                                      ns, nf1, nf2, sigma, d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            Spread_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns,
                                                               nf1, nf2, es_c, es_beta, d_idxnupts, pirange);
        }
    }

#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Spread_2d_NUptsdriven (%d)\t%.3g ms\n", milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
    return 0;
}

template <typename T>
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan_template<T> d_plan)
/*
    This function determines the properties for spreading that are independent
    of the strength of the nodes,  only relates to the locations of the nodes,
    which only needs to be done once.
*/
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    if (bin_size_x < 0 || bin_size_y < 0) {
        std::cout << "error: invalid binsize (binsizex, binsizey) = (";
        std::cout << bin_size_x << "," << bin_size_y << ")" << std::endl;
        return 1;
    }
    int numbins[2];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);
#ifdef DEBUG
    std::cout << "[debug  ] Dividing the uniform grids to bin size[" << d_plan->opts.gpu_binsizex << "x"
              << d_plan->opts.gpu_binsizey << "]" << std::endl;
    std::cout << "[debug  ] numbins = [" << numbins[0] << "x" << numbins[1] << "]" << std::endl;
#endif

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;

#ifdef DEBUG
    T *h_kx;
    T *h_ky;
    h_kx = (T *)malloc(M * sizeof(T));
    h_ky = (T *)malloc(M * sizeof(T));

    checkCudaErrors(cudaMemcpy(h_kx, d_kx, M * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_ky, d_ky, M * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; i++) {
        std::cout << "[debug ]";
        std::cout << "(" << setw(3) << h_kx[i] << "," << setw(3) << h_ky[i] << ")" << std::endl;
    }
#endif
    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx = d_plan->sortidx;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int *d_subprob_to_bin = NULL;

    int pirange = d_plan->spopts.pirange;

    cudaEventRecord(start);
    checkCudaErrors(cudaMemset(d_binsize, 0, numbins[0] * numbins[1] * sizeof(int)));
    CalcBinSize_noghost_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0], numbins[1],
                                                            d_binsize, d_kx, d_ky, d_sortidx, pirange);
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel CalcBinSize_noghost_2d \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_binsize; // For debug
    h_binsize = (int *)malloc(numbins[0] * numbins[1] * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_binsize, d_binsize, numbins[0] * numbins[1] * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "[debug ] bin size:" << std::endl;
    for (int j = 0; j < numbins[1]; j++) {
        std::cout << "[debug ] ";
        for (int i = 0; i < numbins[0]; i++) {
            if (i != 0)
                std::cout << " ";
            std::cout << " bin[" << setw(3) << i << "," << setw(3) << j << "]=" << h_binsize[i + j * numbins[0]];
        }
        std::cout << std::endl;
    }
    free(h_binsize);
    std::cout << "[debug ] ----------------------------------------------------" << std::endl;
#endif
#ifdef DEBUG
    int *h_sortidx;
    h_sortidx = (int *)malloc(M * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_sortidx, d_sortidx, M * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "[debug ]";
    for (int i = 0; i < M; i++) {
        std::cout << "[debug] point[" << setw(3) << i << "]=" << setw(3) << h_sortidx[i] << std::endl;
    }

#endif

    cudaEventRecord(start);
    int n = numbins[0] * numbins[1];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel BinStartPts_2d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_binstartpts;
    h_binstartpts = (int *)malloc((numbins[0] * numbins[1]) * sizeof(int));
    checkCudaErrors(
        cudaMemcpy(h_binstartpts, d_binstartpts, (numbins[0] * numbins[1]) * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "[debug ] Result of scan bin_size array:" << std::endl;
    for (int j = 0; j < numbins[1]; j++) {
        std::cout << "[debug ] ";
        for (int i = 0; i < numbins[0]; i++) {
            if (i != 0)
                std::cout << " ";
            std::cout << "bin[" << setw(3) << i << "," << setw(3) << j << "] = " << setw(2)
                      << h_binstartpts[i + j * numbins[0]];
        }
        std::cout << std::endl;
    }
    free(h_binstartpts);
    std::cout << "[debug ] ---------------------------------------------------" << std::endl;
#endif
    cudaEventRecord(start);
    CalcInvertofGlobalSortIdx_2d<<<(M + 1024 - 1) / 1024, 1024>>>(M, bin_size_x, bin_size_y, numbins[0], numbins[1],
                                                                  d_binstartpts, d_sortidx, d_kx, d_ky, d_idxnupts,
                                                                  pirange, nf1, nf2);
#ifdef DEBUG
    int *h_idxnupts;
    h_idxnupts = (int *)malloc(M * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_idxnupts, d_idxnupts, M * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; i++) {
        std::cout << "[debug ] idx=" << h_idxnupts[i] << std::endl;
    }
    free(h_idxnupts);
#endif
    cudaEventRecord(start);
    CalcSubProb_2d<<<(M + 1024 - 1) / 1024, 1024>>>(d_binsize, d_numsubprob, maxsubprobsize, numbins[0] * numbins[1]);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel CalcSubProb_2d\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_numsubprob;
    h_numsubprob = (int *)malloc(n * sizeof(int));
    checkCudaErrors(
        cudaMemcpy(h_numsubprob, d_numsubprob, numbins[0] * numbins[1] * sizeof(int), cudaMemcpyDeviceToHost));
    for (int j = 0; j < numbins[1]; j++) {
        std::cout << "[debug ] ";
        for (int i = 0; i < numbins[0]; i++) {
            if (i != 0)
                std::cout << " ";
            std::cout << "nsub[" << setw(3) << i << "," << setw(3) << j << "] = " << setw(2)
                      << h_numsubprob[i + j * numbins[0]];
        }
        std::cout << std::endl;
    }
    free(h_numsubprob);
#endif
    d_ptr = thrust::device_pointer_cast(d_numsubprob);
    d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemset(d_subprobstartpts, 0, sizeof(int)));
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Scan Subprob array\t\t%.3g ms\n", milliseconds);
#endif

#ifdef DEBUG
    printf("[debug ] Subproblem start points\n");
    int *h_subprobstartpts;
    h_subprobstartpts = (int *)malloc((n + 1) * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_subprobstartpts, d_subprobstartpts, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int j = 0; j < numbins[1]; j++) {
        std::cout << "[debug ] ";
        for (int i = 0; i < numbins[0]; i++) {
            if (i != 0)
                std::cout << " ";
            std::cout << "nsub[" << setw(3) << i << "," << setw(3) << j << "] = " << setw(2)
                      << h_subprobstartpts[i + j * numbins[0]];
        }
        std::cout << std::endl;
    }
    printf("[debug ] Total number of subproblems = %d\n", h_subprobstartpts[n]);
    free(h_subprobstartpts);
#endif
    cudaEventRecord(start);
    int totalnumsubprob;
    checkCudaErrors(cudaMemcpy(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMalloc(&d_subprob_to_bin, totalnumsubprob * sizeof(int)));
    MapBintoSubProb_2d<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024>>>(d_subprob_to_bin, d_subprobstartpts,
                                                                              d_numsubprob, numbins[0] * numbins[1]);
    assert(d_subprob_to_bin != NULL);
    if (d_plan->subprob_to_bin != NULL)
        cudaFree(d_plan->subprob_to_bin);
    d_plan->subprob_to_bin = d_subprob_to_bin;
    assert(d_plan->subprob_to_bin != NULL);
    d_plan->totalnumsubprob = totalnumsubprob;
#ifdef DEBUG
    printf("[debug ] Map Subproblem to Bins\n");
    int *h_subprob_to_bin;
    h_subprob_to_bin = (int *)malloc((totalnumsubprob) * sizeof(int));
    checkCudaErrors(
        cudaMemcpy(h_subprob_to_bin, d_subprob_to_bin, (totalnumsubprob) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int j = 0; j < totalnumsubprob; j++) {
        std::cout << "[debug ] ";
        std::cout << "nsub[" << j << "] = " << setw(2) << h_subprob_to_bin[j];
        std::cout << std::endl;
    }
    free(h_subprob_to_bin);
#endif
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Subproblem to Bin map\t\t%.3g ms\n", milliseconds);
#endif
    return 0;
}

template <typename T>
int cuspread2d_subprob(int nf1, int nf2, int M, cufinufft_plan_template<T> d_plan, int blksize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
#ifdef INFO
    std::cout << "[info  ] Dividing the uniform grids to bin size[" << d_plan->opts.gpu_binsizex << "x"
              << d_plan->opts.gpu_binsizey << "]" << std::endl;
    std::cout << "[info  ] numbins = [" << numbins[0] << "x" << numbins[1] << "]" << std::endl;
#endif

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int totalnumsubprob = d_plan->totalnumsubprob;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;

    int pirange = d_plan->spopts.pirange;

    T sigma = d_plan->opts.upsampfac;
    cudaEventRecord(start);

    size_t sharedplanorysize =
        (bin_size_x + 2 * (int)ceil(ns / 2.0)) * (bin_size_y + 2 * (int)ceil(ns / 2.0)) * sizeof(cuda_complex<T>);
    if (sharedplanorysize > 49152) {
        std::cout << "error: not enough shared memory" << std::endl;
        return 1;
    }

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            Spread_2d_Subprob_Horner<<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, sigma, d_binstartpts, d_binsize,
                bin_size_x, bin_size_y, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins[0],
                numbins[1], d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            Spread_2d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta, sigma, d_binstartpts,
                d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
                numbins[0], numbins[1], d_idxnupts, pirange);
        }
    }
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Spread_2d_Subprob (%d)\t\t%.3g ms\n", milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
    return 0;
}
template int cuspread2d<float>(cufinufft_plan_template<float> d_plan, int blksize);
template int cuspread2d<double>(cufinufft_plan_template<double> d_plan, int blksize);
template int cuspread2d_subprob_prop<float>(int nf1, int nf2, int M, cufinufft_plan_template<float> d_plan);
template int cuspread2d_subprob_prop<double>(int nf1, int nf2, int M, cufinufft_plan_template<double> d_plan);
template int cuspread2d_nuptsdriven_prop<float>(int nf1, int nf2, int M, cufinufft_plan_template<float> d_plan);
template int cuspread2d_nuptsdriven_prop<double>(int nf1, int nf2, int M, cufinufft_plan_template<double> d_plan);

} // namespace spreadinterp
} // namespace cufinufft
