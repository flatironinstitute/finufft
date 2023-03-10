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
using namespace cufinufft::common;

namespace cufinufft {
namespace spreadinterp {
// only relates to the locations of the nodes, which only needs to be done once
template <typename T>
int cuspread2d_paul_prop(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int ns = d_plan->spopts.nspread;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
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
    for (int i = 0; i < M; i++) {
        std::cout << "[debug ]";
        std::cout << " (" << setw(3) << h_kx[i] << "," << setw(3) << h_ky[i] << ")" << std::endl;
    }
#endif
    int *d_binsize = d_plan->binsize;
    int *d_finegridsize = d_plan->finegridsize;
    int *d_sortidx = d_plan->sortidx;
    int *d_fgstartpts = d_plan->fgstartpts;
    int *d_idxnupts = d_plan->idxnupts;
    int *d_numsubprob = d_plan->numsubprob;

    int pirange = d_plan->spopts.pirange;

    void *d_temp_storage = NULL;

    cudaEventRecord(start);
    checkCudaErrors(cudaMemset(d_finegridsize, 0, nf1 * nf2 * sizeof(int)));
    LocateFineGridPos_Paul<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0], numbins[1],
                                                            d_binsize, ns, d_kx, d_ky, d_sortidx, d_finegridsize,
                                                            pirange);
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel LocateFineGridPos \t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    printf("[debug ] ns = %d\n", ns);
    int binx, biny, binidx;
    int *h_finegridsize;
    h_finegridsize = (int *)malloc(nf1 * nf2 * sizeof(int));

    checkCudaErrors(cudaMemcpy(h_finegridsize, d_finegridsize, nf1 * nf2 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int j = 0; j < nf2; j++) {
        if (j % d_plan->opts.gpu_binsizey == 0)
            printf("\n");
        biny = floor(j / bin_size_y);
        std::cout << "[debug ] ";
        for (int i = 0; i < nf1; i++) {
            if (i % d_plan->opts.gpu_binsizex == 0 && i != 0)
                printf(" |");
            binx = floor(i / bin_size_x);
            binidx = binx + biny * numbins[0];
            if (i != 0)
                std::cout << " ";
            std::cout << setw(2)
                      << h_finegridsize[binidx * bin_size_x * bin_size_y + (i - binx * bin_size_x) +
                                        (j - bin_size_y * biny) * bin_size_x];
        }
        std::cout << std::endl;
    }
    std::cout << "[debug ] ------------------------------------------------" << std::endl;

    free(h_finegridsize);
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
#endif
#ifdef DEBUG
    std::cout << "[debug ] ------------------------------------------------" << std::endl;
    int *h_sortidx;
    h_sortidx = (int *)malloc(M * sizeof(int));

    checkCudaErrors(cudaMemcpy(h_sortidx, d_sortidx, M * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "[debug ]";
    for (int i = 0; i < M; i++) {
        std::cout << "point[" << setw(3) << i << "]=" << setw(3) << h_sortidx[i] << std::endl;
    }
#endif
    int n = nf1 * nf2;
    cudaEventRecord(start);
    thrust::device_ptr<int> d_ptr(d_finegridsize);
    thrust::device_ptr<int> d_result(d_fgstartpts);
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Scan fingridsize array\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_fgstartpts;
    h_fgstartpts = (int *)malloc((nf1 * nf2) * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_fgstartpts, d_fgstartpts, (nf1 * nf2) * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "[debug ] Result of scan finegridsize array:" << std::endl;
    for (int j = 0; j < nf2; j++) {
        if (j % d_plan->opts.gpu_binsizey == 0)
            printf("\n");
        biny = floor(j / bin_size_y);
        std::cout << "[debug ] ";
        for (int i = 0; i < nf1; i++) {
            if (i % d_plan->opts.gpu_binsizex == 0 && i != 0)
                printf(" |");
            binx = floor(i / bin_size_x);
            binidx = binx + biny * numbins[0];
            if (i != 0)
                std::cout << " ";
            std::cout << setw(2)
                      << h_fgstartpts[binidx * bin_size_x * bin_size_y + (i - binx * bin_size_x) +
                                      (j - bin_size_y * biny) * bin_size_x];
        }
        std::cout << std::endl;
    }
    free(h_fgstartpts);
    std::cout << "[debug ] -----------------------------------------------" << std::endl;
#endif
    cudaEventRecord(start);
    CalcInvertofGlobalSortIdx_Paul<<<(M + 1024 - 1) / 1024, 1024>>>(nf1, nf2, M, bin_size_x, bin_size_y, numbins[0],
                                                                    numbins[1], ns, d_kx, d_ky, d_fgstartpts, d_sortidx,
                                                                    d_idxnupts, pirange);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tCalcInvertofGlobalSortIdx_Paul\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_idxnupts;
    h_idxnupts = (int *)malloc(M * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_idxnupts, d_idxnupts, M * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; i++) {
        std::cout << "idx=" << h_idxnupts[i] << " ";
    }
    std::cout << std::endl;
    free(h_idxnupts);
#endif
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
    cudaEventRecord(start);
    int blocksize = bin_size_x * bin_size_y;
    cudaEventRecord(start);
    CalcSubProb_2d_Paul<<<numbins[0] * numbins[1], blocksize>>>(d_finegridsize, d_numsubprob, maxsubprobsize,
                                                                bin_size_x, bin_size_y);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tCalcSubProb_2d_Paul\t\t%.3g ms\n", milliseconds);
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
    int *d_subprobstartpts = d_plan->subprobstartpts;
    n = numbins[0] * numbins[1];
    cudaEventRecord(start);
    d_ptr = thrust::device_pointer_cast(d_numsubprob);
    d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemset(d_subprobstartpts, 0, sizeof(int)));
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tScan subproblem size array\t%.3g ms\n", milliseconds);
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
    int *d_subprob_to_bin;
    int totalnumsubprob;
    cudaEventRecord(start);
    checkCudaErrors(cudaMemcpy(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int), cudaMemcpyDeviceToHost));
    // TODO: Warning! This gets malloc'ed but not freed
    checkCudaErrors(cudaMalloc(&d_subprob_to_bin, totalnumsubprob * sizeof(int)));
    MapBintoSubProb_2d<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024>>>(d_subprob_to_bin, d_subprobstartpts,
                                                                              d_numsubprob, numbins[0] * numbins[1]);
    assert(d_subprob_to_bin != NULL);
    d_plan->subprob_to_bin = d_subprob_to_bin;
    assert(d_plan->subprob_to_bin != NULL);
    d_plan->totalnumsubprob = totalnumsubprob;
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tMap Subproblem to Bins\t\t%.3g ms\n", milliseconds);
#endif
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
#endif
    cudaFree(d_temp_storage);
    return 0;
}

template <typename T>
int cuspread2d_paul(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan, int blksize) {
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
    int *d_fgstartpts = d_plan->fgstartpts;
    int *d_finegridsize = d_plan->finegridsize;

    int totalnumsubprob = d_plan->totalnumsubprob;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;

    int pirange = d_plan->spopts.pirange;
    T sigma = d_plan->opts.upsampfac;
    cudaEventRecord(start);
    size_t sharedplanorysize = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) * sizeof(cuda_complex<T>);
    if (sharedplanorysize > 49152) {
        std::cout << "error: not enough shared memory" << std::endl;
        return 1;
    }
    for (int t = 0; t < blksize; t++) {
        Spread_2d_Subprob_Paul<<<totalnumsubprob, 1024, sharedplanorysize>>>(
            d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, ns, nf1, nf2, es_c, es_beta, sigma, d_binstartpts,
            d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
            numbins[0], numbins[1], d_idxnupts, d_fgstartpts, d_finegridsize, pirange);
    }
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Spread_2d_Subprob_Paul \t%.3g ms\n", milliseconds);
#endif
    return 0;
}

} // namespace spreadinterp
} // namespace cufinufft
