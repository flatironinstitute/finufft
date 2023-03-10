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
using namespace cufinufft::memtransfer;

namespace cufinufft {
namespace spreadinterp {

template <typename T>
int cufinufft_spread3d(int nf1, int nf2, int nf3, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, T *d_kz,
                       cuda_complex<T> *d_c, cufinufft_plan_template<T> *d_plan)
/*
    This c function is written for only doing 3D spreading. See
    test/spread3d_test.cu for usage.

    Melody Shih 07/25/19
    not allocate,transfer and free memories on gpu. Shih 09/24/20
*/
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int ier;
    d_plan->kx = d_kx;
    d_plan->ky = d_ky;
    d_plan->kz = d_kz;
    d_plan->c = d_c;
    d_plan->fw = d_fw;
    // ier = setup_spreader_for_nufft(d_plan->spopts, eps, d_plan->opts);
    d_plan->nf1 = nf1;
    d_plan->nf2 = nf2;
    d_plan->nf3 = nf3;
    d_plan->M = M;
    d_plan->maxbatchsize = 1;

    cudaEventRecord(start);
    ier = ALLOCGPUMEM3D_PLAN(d_plan);
    ier = ALLOCGPUMEM3D_NUPTS(d_plan);

    cudaEventRecord(start);
    if (d_plan->opts.gpu_method == 1) {
        ier = CUSPREAD3D_NUPTSDRIVEN_PROP(nf1, nf2, nf3, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n", d_plan->opts.gpu_method);
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
    if (d_plan->opts.gpu_method == 4) {
        ier = CUSPREAD3D_BLOCKGATHER_PROP(nf1, nf2, nf3, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread3d_blockgather_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }
#ifdef TIME
    float milliseconds;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] Obtain Spread Prop\t %.3g ms\n", milliseconds);
#endif

    cudaEventRecord(start);
    ier = CUSPREAD3D(d_plan, 1);
#ifdef TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] Spread (%d)\t\t %.3g ms\n", d_plan->opts.gpu_method, milliseconds);
#endif

    cudaEventRecord(start);
    FREEGPUMEMORY3D(d_plan);
#ifdef TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
    return ier;
}

template <typename T>
int cuspread3d(cufinufft_plan_template<T> *d_plan, int blksize)
/*
    A wrapper for different spreading methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem
    (4) Block gather

    Melody Shih 07/25/19
*/
{
    int nf1 = d_plan->nf1;
    int nf2 = d_plan->nf2;
    int nf3 = d_plan->nf3;
    int M = d_plan->M;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int ier = 0;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        cudaEventRecord(start);
        ier = CUSPREAD3D_NUPTSDRIVEN(nf1, nf2, nf3, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread3d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    case 2: {
        cudaEventRecord(start);
        ier = CUSPREAD3D_SUBPROB(nf1, nf2, nf3, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread3d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    case 4: {
        cudaEventRecord(start);
        ier = CUSPREAD3D_BLOCKGATHER(nf1, nf2, nf3, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread3d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    default:
        std::cerr << "error: incorrect method, should be 1,2,4" << std::endl;
        return 2;
    }
    return ier;
}

template <typename T>
int cuspread3d_nuptsdriven_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (d_plan->opts.gpu_sort) {
        int bin_size_x = d_plan->opts.gpu_binsizex;
        int bin_size_y = d_plan->opts.gpu_binsizey;
        int bin_size_z = d_plan->opts.gpu_binsizez;
        if (bin_size_x < 0 || bin_size_y < 0 || bin_size_z < 0) {
            std::cout << "error: invalid binsize (binsizex, binsizey, binsizez) = (";
            std::cout << bin_size_x << "," << bin_size_y << "," << bin_size_z << ")" << std::endl;
            return 1;
        }

        int numbins[3];
        numbins[0] = ceil((T)nf1 / bin_size_x);
        numbins[1] = ceil((T)nf2 / bin_size_y);
        numbins[2] = ceil((T)nf3 / bin_size_z);

#ifdef DEBUG
        std::cout << "[debug ] Dividing the uniform grids to bin size[" << d_plan->opts.gpu_binsizex << "x"
                  << d_plan->opts.gpu_binsizey << "x" << d_plan->opts.gpu_binsizez << "]" << std::endl;
        std::cout << "[debug ] numbins = [" << numbins[0] << "x" << numbins[1] << "x" << numbins[2] << "]" << std::endl;
#endif

        T *d_kx = d_plan->kx;
        T *d_ky = d_plan->ky;
        T *d_kz = d_plan->kz;
#ifdef DEBUG
        T *h_kx;
        T *h_ky;
        T *h_kz;
        h_kx = (T *)malloc(M * sizeof(T));
        h_ky = (T *)malloc(M * sizeof(T));
        h_kz = (T *)malloc(M * sizeof(T));

        checkCudaErrors(cudaMemcpy(h_kx, d_kx, M * sizeof(T), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_ky, d_ky, M * sizeof(T), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_kz, d_kz, M * sizeof(T), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 10; i++) {
            std::cout << "[debug ] ";
            std::cout << "(" << setw(3) << h_kx[i] << "," << setw(3) << h_ky[i] << "," << setw(3) << h_kz[i] << ")"
                      << std::endl;
        }
#endif

        int *d_binsize = d_plan->binsize;
        int *d_binstartpts = d_plan->binstartpts;
        int *d_sortidx = d_plan->sortidx;
        int *d_idxnupts = d_plan->idxnupts;

        int pirange = d_plan->spopts.pirange;

        cudaEventRecord(start);
        checkCudaErrors(cudaMemset(d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
        CalcBinSize_noghost_3d<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, nf2, nf3, bin_size_x, bin_size_y, bin_size_z,
                                                                numbins[0], numbins[1], numbins[2], d_binsize, d_kx,
                                                                d_ky, d_kz, d_sortidx, pirange);
#ifdef SPREADTIME
        float milliseconds = 0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] \tKernel CalcBinSize_noghost_3d \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
        int *h_binsize; // For debug
        h_binsize = (int *)malloc(numbins[0] * numbins[1] * numbins[2] * sizeof(int));
        checkCudaErrors(cudaMemcpy(h_binsize, d_binsize, numbins[0] * numbins[1] * numbins[2] * sizeof(int),
                                   cudaMemcpyDeviceToHost));
        std::cout << "[debug ] bin size:" << std::endl;
        for (int k = 0; k < numbins[2]; k++) {
            for (int j = 0; j < numbins[1]; j++) {
                std::cout << "[debug ] ";
                for (int i = 0; i < numbins[0]; i++) {
                    if (i != 0)
                        std::cout << " ";
                    std::cout << " bin[" << setw(1) << i << "," << setw(1) << j << "," << setw(1) << k
                              << "]=" << h_binsize[i + j * numbins[0] + k * numbins[0] * numbins[1]];
                }
                std::cout << std::endl;
            }
        }
        free(h_binsize);
        std::cout << "[debug ] ------------------------------------------------" << std::endl;
#endif
#ifdef DEBUG
        int *h_sortidx;
        h_sortidx = (int *)malloc(M * sizeof(int));

        checkCudaErrors(cudaMemcpy(h_sortidx, d_sortidx, M * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < M; i++) {
            std::cout << "[debug ] ";
            std::cout << "point[" << setw(3) << i << "]=" << setw(3) << h_sortidx[i] << std::endl;
        }
#endif

        cudaEventRecord(start);
        int n = numbins[0] * numbins[1] * numbins[2];
        thrust::device_ptr<int> d_ptr(d_binsize);
        thrust::device_ptr<int> d_result(d_binstartpts);
        thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
#ifdef SPREADTIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] \tKernel BinStartPts_3d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
        int *h_binstartpts;
        h_binstartpts = (int *)malloc((numbins[0] * numbins[1] * numbins[2]) * sizeof(int));
        checkCudaErrors(cudaMemcpy(h_binstartpts, d_binstartpts, (numbins[0] * numbins[1] * numbins[2]) * sizeof(int),
                                   cudaMemcpyDeviceToHost));
        std::cout << "[debug ] Result of scan bin_size array:" << std::endl;
        for (int k = 0; k < numbins[2]; k++) {
            for (int j = 0; j < numbins[1]; j++) {
                std::cout << "[debug ] ";
                for (int i = 0; i < numbins[0]; i++) {
                    if (i != 0)
                        std::cout << " ";
                    std::cout << " bin[" << setw(1) << i << "," << setw(1) << j << "," << setw(1) << k
                              << "]=" << h_binstartpts[i + j * numbins[0] + k * numbins[0] * numbins[1]];
                }
                std::cout << std::endl;
            }
        }
        free(h_binstartpts);
        std::cout << "[debug ] ------------------------------------------------" << std::endl;
#endif
        cudaEventRecord(start);
        CalcInvertofGlobalSortIdx_3d<<<(M + 1024 - 1) / 1024, 1024>>>(
            M, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2], d_binstartpts, d_sortidx, d_kx,
            d_ky, d_kz, d_idxnupts, pirange, nf1, nf2, nf3);
#ifdef SPREADTIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] \tKernel CalcInvertofGlobalSortIdx_3d \t%.3g ms\n", milliseconds);
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
        TrivialGlobalSortIdx_3d<<<(M + 1024 - 1) / 1024, 1024>>>(M, d_idxnupts);
#ifdef SPREADTIME
        float milliseconds = 0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] \tKernel TrivialGlobalSortIDx_3d \t\t%.3g ms\n", milliseconds);
#endif
    }
    return 0;
}

template <typename T>
int cuspread3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan, int blksize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock;
    dim3 blocks;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    T sigma = d_plan->spopts.upsampfac;
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    int pirange = d_plan->spopts.pirange;

    int *d_idxnupts = d_plan->idxnupts;
    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    threadsPerBlock.x = 16;
    threadsPerBlock.y = 1;
    blocks.x = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = 1;
    cudaEventRecord(start);
    if (d_plan->opts.gpu_kerevalmeth == 1) {
        for (int t = 0; t < blksize; t++) {
            Spread_3d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_kz, d_c + t * M,
                                                                      d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3,
                                                                      sigma, d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            Spread_3d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_kz, d_c + t * M,
                                                               d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c,
                                                               es_beta, d_idxnupts, pirange);
        }
    }
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Spread_3d_NUptsdriven (%d)\t%.3g ms\n", milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
    return 0;
}

template <typename T>
int cuspread3d_blockgather_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock;
    dim3 blocks;

    int pirange = d_plan->spopts.pirange;

    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
    int o_bin_size_x = d_plan->opts.gpu_obinsizex;
    int o_bin_size_y = d_plan->opts.gpu_obinsizey;
    int o_bin_size_z = d_plan->opts.gpu_obinsizez;

    int numobins[3];
    if (nf1 % o_bin_size_x != 0 || nf2 % o_bin_size_y != 0 || nf3 % o_bin_size_z != 0) {
        std::cout << "error: mod(nf1, opts.gpu_obinsizex) != 0" << std::endl;
        std::cout << "       mod(nf2, opts.gpu_obinsizey) != 0" << std::endl;
        std::cout << "       mod(nf3, opts.gpu_obinsizez) != 0" << std::endl;
        std::cout << "error: (nf1, nf2, nf3) = (" << nf1 << ", " << nf2 << ", " << nf3 << ")" << std::endl;
        std::cout << "error: (obinsizex, obinsizey, obinsizez) = (" << o_bin_size_x << ", " << o_bin_size_y << ", "
                  << o_bin_size_z << ")" << std::endl;
        return 1;
    }

    numobins[0] = ceil((T)nf1 / o_bin_size_x);
    numobins[1] = ceil((T)nf2 / o_bin_size_y);
    numobins[2] = ceil((T)nf3 / o_bin_size_z);

    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int bin_size_z = d_plan->opts.gpu_binsizez;
    if (o_bin_size_x % bin_size_x != 0 || o_bin_size_y % bin_size_y != 0 || o_bin_size_z % bin_size_z != 0) {
        std::cout << "error: mod(ops.gpu_obinsizex, opts.gpu_binsizex) != 0" << std::endl;
        std::cout << "       mod(ops.gpu_obinsizey, opts.gpu_binsizey) != 0" << std::endl;
        std::cout << "       mod(ops.gpu_obinsizez, opts.gpu_binsizez) != 0" << std::endl;
        std::cout << "error: (binsizex, binsizey, binsizez) = (" << bin_size_x << ", " << bin_size_y << ", "
                  << bin_size_z << ")" << std::endl;
        std::cout << "error: (obinsizex, obinsizey, obinsizez) = (" << o_bin_size_x << ", " << o_bin_size_y << ", "
                  << o_bin_size_z << ")" << std::endl;
        return 1;
    }

    int binsperobinx, binsperobiny, binsperobinz;
    int numbins[3];
    binsperobinx = o_bin_size_x / bin_size_x + 2;
    binsperobiny = o_bin_size_y / bin_size_y + 2;
    binsperobinz = o_bin_size_z / bin_size_z + 2;
    numbins[0] = numobins[0] * (binsperobinx);
    numbins[1] = numobins[1] * (binsperobiny);
    numbins[2] = numobins[2] * (binsperobinz);
#ifdef DEBUG
    std::cout << "[debug ] Dividing the uniform grids to bin size[" << d_plan->opts.gpu_binsizex << "x"
              << d_plan->opts.gpu_binsizey << "x" << d_plan->opts.gpu_binsizez << "]" << std::endl;
    std::cout << "[debug ] numobins = [" << numobins[0] << "x" << numobins[1] << "x" << numobins[2] << "]" << std::endl;
    std::cout << "[debug ] numbins = [" << numbins[0] << "x" << numbins[1] << "x" << numbins[2] << "]" << std::endl;
#endif

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;

#ifdef DEBUG
    T *h_kx, *h_ky, *h_kz;
    h_kx = (T *)malloc(M * sizeof(T));
    h_ky = (T *)malloc(M * sizeof(T));
    h_kz = (T *)malloc(M * sizeof(T));

    checkCudaErrors(cudaMemcpy(h_kx, d_kx, M * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_ky, d_ky, M * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_kz, d_kz, M * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; i++) {
        std::cout << "[debug ] ";
        std::cout << "(" << setw(3) << h_kx[i] << "," << setw(3) << h_ky[i] << "," << h_kz[i] << ")" << std::endl;
    }
#endif
    int *d_binsize = d_plan->binsize;
    int *d_sortidx = d_plan->sortidx;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_numsubprob = d_plan->numsubprob;
    void *d_temp_storage = NULL;
    int *d_idxnupts = NULL;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_subprob_to_bin = NULL;

    cudaEventRecord(start);
    checkCudaErrors(cudaMemset(d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
    LocateNUptstoBins_ghost<<<(M + 1024 - 1) / 1024, 1024>>>(
        M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2], binsperobinx, binsperobiny,
        binsperobinz, d_binsize, d_kx, d_ky, d_kz, d_sortidx, pirange, nf1, nf2, nf3);

#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel LocateNUptstoBins_ghost \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_binsize; // For debug
    h_binsize = (int *)malloc(numbins[0] * numbins[1] * numbins[2] * sizeof(int));
    checkCudaErrors(
        cudaMemcpy(h_binsize, d_binsize, numbins[0] * numbins[1] * numbins[2] * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "[debug ] bin size:" << std::endl;
    for (int k = 0; k < numbins[2]; k++) {
        std::cout << "[debug ]" << std::endl;
        for (int j = 0; j < numbins[1]; j++) {
            if (j % binsperobinx == 0 && j != 0)
                std::cout << "[debug ] -----------------" << std::endl;
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i % binsperobinx == 0 && i != 0)
                    std::cout << "|";
                if (i != 0)
                    std::cout << " ";
                int binidx = CalcGlobalIdx(i, j, k, numobins[0], numobins[1], numobins[2], binsperobinx, binsperobiny,
                                           binsperobinz);
                std::cout << h_binsize[binidx];
            }
            std::cout << std::endl;
        }
    }
    std::cout << "[debug ] ---------------------------------------------------" << std::endl;
#endif
#ifdef DEBUG
    int *h_sortidx;
    h_sortidx = (int *)malloc(M * sizeof(int));

    checkCudaErrors(cudaMemcpy(h_sortidx, d_sortidx, M * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; i++) {
        std::cout << "[debug ] point[" << setw(3) << i << "]=" << setw(3) << h_sortidx[i] << std::endl;
    }
#endif
    cudaEventRecord(start);
    threadsPerBlock.x = 8;
    threadsPerBlock.y = 8;
    threadsPerBlock.z = 8;

    blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
    blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
    blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

    FillGhostBins<<<blocks, threadsPerBlock>>>(binsperobinx, binsperobiny, binsperobinz, numobins[0], numobins[1],
                                               numobins[2], d_binsize);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel FillGhostBins \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    checkCudaErrors(
        cudaMemcpy(h_binsize, d_binsize, numbins[0] * numbins[1] * numbins[2] * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "[debug ] Filled ghost bins:" << std::endl;
    for (int k = 0; k < numbins[2]; k++) {
        std::cout << "[debug ] " << std::endl;
        std::cout << "[debug ] " << std::endl;
        for (int j = 0; j < numbins[1]; j++) {
            if (j % binsperobinx == 0 && j != 0)
                std::cout << "[debug ] -----------------" << std::endl;
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i % binsperobinx == 0 && i != 0)
                    std::cout << "|";
                int binidx = CalcGlobalIdx(i, j, k, numobins[0], numobins[1], numobins[2], binsperobinx, binsperobiny,
                                           binsperobinz);
                if (i != 0)
                    std::cout << " ";
                std::cout << h_binsize[binidx];
            }
            std::cout << std::endl;
        }
    }
    std::cout << "[debug ] ---------------------------------------------------" << std::endl;
#endif
    cudaEventRecord(start);
    int n = numbins[0] * numbins[1] * numbins[2];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemset(d_binstartpts, 0, sizeof(int)));
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel BinStartPts_3d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_binstartpts;
    h_binstartpts = (int *)malloc((numbins[0] * numbins[1] * numbins[2]) * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_binstartpts, d_binstartpts, (numbins[0] * numbins[1] * numbins[2]) * sizeof(int),
                               cudaMemcpyDeviceToHost));
    std::cout << "[debug ] Result of scan bin_size array:" << std::endl;
    for (int k = 0; k < numbins[2]; k++) {
        std::cout << "[debug ] " << std::endl;
        for (int j = 0; j < numbins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                int binidx = CalcGlobalIdx(i, j, k, numobins[0], numobins[1], numobins[2], binsperobinx, binsperobiny,
                                           binsperobinz);
                std::cout << h_binstartpts[binidx];
            }
            std::cout << std::endl;
        }
    }
    std::cout << "[debug ] ----------------------------------------------------" << std::endl;
#endif
    cudaEventRecord(start);
    int totalNUpts;
    checkCudaErrors(cudaMemcpy(&totalNUpts, &d_binstartpts[n], sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMalloc(&d_idxnupts, totalNUpts * sizeof(int)));
#ifdef DEBUG
    checkCudaErrors(cudaMemset(d_idxnupts, -1, totalNUpts * sizeof(int)));
#endif
    cudaEventRecord(start);
    CalcInvertofGlobalSortIdx_ghost<<<(M + 1024 - 1) / 1024, 1024>>>(
        M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2], binsperobinx, binsperobiny,
        binsperobinz, d_binstartpts, d_sortidx, d_kx, d_ky, d_kz, d_idxnupts, pirange, nf1, nf2, nf3);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel CalcInvertofGlobalIdx_ghost \t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_idxnupts;
    h_idxnupts = (int *)malloc(totalNUpts * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_idxnupts, d_idxnupts, totalNUpts * sizeof(int), cudaMemcpyDeviceToHost));
    int pts = 0;
    for (int b = 0; b < numbins[0] * numbins[1] * numbins[1]; b++) {
        if (h_binsize[b] > 0)
            std::cout << "[debug ] Bin " << b << std::endl;
        for (int i = h_binstartpts[b]; i < h_binstartpts[b] + h_binsize[b]; i++) {
            std::cout << "[debug ] NUpts-index= " << h_idxnupts[i] << std::endl;
            pts++;
        }
    }
    std::cout << "[debug ] totalpts = " << pts << std::endl;
#endif
    cudaEventRecord(start);
    threadsPerBlock.x = 2;
    threadsPerBlock.y = 2;
    threadsPerBlock.z = 2;

    blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
    blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
    blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

    GhostBinPtsIdx<<<blocks, threadsPerBlock>>>(binsperobinx, binsperobiny, binsperobinz, numobins[0], numobins[1],
                                                numobins[2], d_binsize, d_idxnupts, d_binstartpts, M);
    if (d_plan->idxnupts != NULL)
        cudaFree(d_plan->idxnupts);
    d_plan->idxnupts = d_idxnupts;
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel GhostBinPtsIdx \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    checkCudaErrors(cudaMemcpy(h_idxnupts, d_idxnupts, totalNUpts * sizeof(int), cudaMemcpyDeviceToHost));
    pts = 0;
    for (int b = 0; b < numbins[0] * numbins[1] * numbins[1]; b++) {
        if (h_binsize[b] > 0)
            std::cout << "[debug ] Bin " << b << std::endl;
        for (int i = h_binstartpts[b]; i < h_binstartpts[b] + h_binsize[b]; i++) {
            std::cout << "[debug ] NUpts-index= " << h_idxnupts[i] << std::endl;
            pts++;
        }
    }
    std::cout << "[debug ] totalpts = " << pts << std::endl;
    free(h_idxnupts);
    free(h_binstartpts);
    free(h_binsize);
#endif

    /* --------------------------------------------- */
    //        Determining Subproblem properties      //
    /* --------------------------------------------- */
    cudaEventRecord(start);
    n = numobins[0] * numobins[1] * numobins[2];
    cudaEventRecord(start);
    CalcSubProb_3d_v1<<<(n + 1024 - 1) / 1024, 1024>>>(binsperobinx, binsperobiny, binsperobinz, d_binsize,
                                                       d_numsubprob, maxsubprobsize,
                                                       numobins[0] * numobins[1] * numobins[2]);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel CalcSubProb_3d_v1\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_numsubprob;
    h_numsubprob = (int *)malloc(n * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_numsubprob, d_numsubprob, numobins[0] * numobins[1] * numobins[2] * sizeof(int),
                               cudaMemcpyDeviceToHost));
    for (int k = 0; k < numobins[2]; k++) {
        std::cout << "[debug ] " << std::endl;
        for (int j = 0; j < numobins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numobins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                std::cout << "s[" << setw(1) << i << "," << setw(1) << j << "," << setw(1) << k << "]= " << setw(3)
                          << h_numsubprob[i + j * numobins[0] + k * numobins[1] * numobins[2]];
            }
            std::cout << std::endl;
        }
    }
    free(h_numsubprob);
#endif
    cudaEventRecord(start);
    n = numobins[0] * numobins[1] * numobins[2];
    d_ptr = thrust::device_pointer_cast(d_numsubprob);
    d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemset(d_subprobstartpts, 0, sizeof(int)));
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tScan  numsubprob\t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    printf("[debug ] Subproblem start points\n");
    int *h_subprobstartpts;
    h_subprobstartpts = (int *)malloc((n + 1) * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_subprobstartpts, d_subprobstartpts,
                               (numobins[0] * numobins[1] * numobins[2] + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int k = 0; k < numobins[2]; k++) {
        if (k != 0)
            std::cout << "[debug ] " << std::endl;
        for (int j = 0; j < numobins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numobins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                std::cout << "s[" << setw(1) << i << "," << setw(1) << j << "," << setw(1) << k << "]= " << setw(3)
                          << h_subprobstartpts[i + j * numobins[0] + k * numobins[1] * numobins[2]];
            }
            std::cout << std::endl;
        }
    }
    printf("[debug ] Total number of subproblems (%d) = %d\n", n, h_subprobstartpts[n]);
    free(h_subprobstartpts);
    std::cout << "[debug ] ---------------------------------------------------" << std::endl;
#endif
    cudaEventRecord(start);
    int totalnumsubprob;
    checkCudaErrors(cudaMemcpy(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMalloc(&d_subprob_to_bin, totalnumsubprob * sizeof(int)));
    MapBintoSubProb_3d_v1<<<(n + 1024 - 1) / 1024, 1024>>>(d_subprob_to_bin, d_subprobstartpts, d_numsubprob, n);
    assert(d_subprob_to_bin != NULL);
    if (d_plan->subprob_to_bin != NULL)
        cudaFree(d_plan->subprob_to_bin);
    d_plan->subprob_to_bin = d_subprob_to_bin;
    d_plan->totalnumsubprob = totalnumsubprob;
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Subproblem to Bin map\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    printf("[debug ] Map Subproblem to Bins\n");
    int *h_subprob_to_bin;
    h_subprob_to_bin = (int *)malloc((totalnumsubprob) * sizeof(int));
    checkCudaErrors(
        cudaMemcpy(h_subprob_to_bin, d_subprob_to_bin, (totalnumsubprob) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int j = 0; j < totalnumsubprob; j++) {
        std::cout << "[debug ] ";
        std::cout << "s[" << j << "] = " << setw(2) << "b[" << h_subprob_to_bin[j] << "]";
        std::cout << std::endl;
    }
    free(h_subprob_to_bin);
#endif
    cudaFree(d_temp_storage);

    return 0;
}

template <typename T>
int cuspread3d_blockgather(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan, int blksize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int ns = d_plan->spopts.nspread;
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    T sigma = d_plan->spopts.upsampfac;
    int pirange = d_plan->spopts.pirange;
    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

    int obin_size_x = d_plan->opts.gpu_obinsizex;
    int obin_size_y = d_plan->opts.gpu_obinsizey;
    int obin_size_z = d_plan->opts.gpu_obinsizez;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int bin_size_z = d_plan->opts.gpu_binsizez;
    int numobins[3];
    numobins[0] = ceil((T)nf1 / obin_size_x);
    numobins[1] = ceil((T)nf2 / obin_size_y);
    numobins[2] = ceil((T)nf3 / obin_size_z);

    int binsperobinx, binsperobiny, binsperobinz;
    binsperobinx = obin_size_x / bin_size_x + 2;
    binsperobiny = obin_size_y / bin_size_y + 2;
    binsperobinz = obin_size_z / bin_size_z + 2;
#ifdef INFO
    std::cout << "[info  ] Dividing the uniform grids to bin size[" << obin_size_x << "x" << obin_size_y << "x"
              << obin_size_z << "]" << std::endl;
    std::cout << "[info  ] numbins = [" << numobins[0] << "x" << numobins[1] << "x" << numobins[2] << "]" << std::endl;
    std::cout << "[info  ] ns = " << ns << std::endl;
#endif

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    int *d_binstartpts = d_plan->binstartpts;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int totalnumsubprob = d_plan->totalnumsubprob;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;

    cudaEventRecord(start);
    for (int t = 0; t < blksize; t++) {
        if (d_plan->opts.gpu_kerevalmeth == 1) {
            size_t sharedplanorysize = obin_size_x * obin_size_y * obin_size_z * sizeof(cuda_complex<T>);
            if (sharedplanorysize > 49152) {
                std::cout << "error: not enough shared memory" << std::endl;
                return 1;
            }
            Spread_3d_BlockGather_Horner<<<totalnumsubprob, 64, sharedplanorysize>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c, es_beta, sigma,
                d_binstartpts, obin_size_x, obin_size_y, obin_size_z, binsperobinx * binsperobiny * binsperobinz,
                d_subprob_to_bin, d_subprobstartpts, maxsubprobsize, numobins[0], numobins[1], numobins[2], d_idxnupts,
                pirange);
        } else {
            size_t sharedplanorysize = obin_size_x * obin_size_y * obin_size_z * sizeof(cuda_complex<T>);
            if (sharedplanorysize > 49152) {
                std::cout << "error: not enough shared memory" << std::endl;
                return 1;
            }
            Spread_3d_BlockGather<<<totalnumsubprob, 64, sharedplanorysize>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c, es_beta, sigma,
                d_binstartpts, obin_size_x, obin_size_y, obin_size_z, binsperobinx * binsperobiny * binsperobinz,
                d_subprob_to_bin, d_subprobstartpts, maxsubprobsize, numobins[0], numobins[1], numobins[2], d_idxnupts,
                pirange);
        }
    }
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Spread_3d_BlockGather (%d)\t%.3g ms\n", milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
    return 0;
}

template <typename T>
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int bin_size_z = d_plan->opts.gpu_binsizez;
    if (bin_size_x < 0 || bin_size_y < 0 || bin_size_z < 0) {
        std::cout << "error: invalid binsize (binsizex, binsizey, binsizez) = (";
        std::cout << bin_size_x << "," << bin_size_y << "," << bin_size_z << ")" << std::endl;
        return 1;
    }

    int numbins[3];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);
    numbins[2] = ceil((T)nf3 / bin_size_z);
#ifdef DEBUG
    std::cout << "[debug ] Dividing the uniform grids to bin size[" << d_plan->opts.gpu_binsizex << "x"
              << d_plan->opts.gpu_binsizey << "x" << d_plan->opts.gpu_binsizez << "]" << std::endl;
    std::cout << "[debug ] numbins = [" << numbins[0] << "x" << numbins[1] << "x" << numbins[2] << "]" << std::endl;
#endif

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;

#ifdef DEBUG
    T *h_kx;
    T *h_ky;
    T *h_kz;
    h_kx = (T *)malloc(M * sizeof(T));
    h_ky = (T *)malloc(M * sizeof(T));
    h_kz = (T *)malloc(M * sizeof(T));

    checkCudaErrors(cudaMemcpy(h_kx, d_kx, M * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_ky, d_ky, M * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_kz, d_kz, M * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = M - 10; i < M; i++) {
        std::cout << "[debug ] ";
        std::cout << "(" << setw(3) << h_kx[i] << "," << setw(3) << h_ky[i] << "," << setw(3) << h_kz[i] << ")"
                  << std::endl;
    }
#endif

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx = d_plan->sortidx;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int *d_subprob_to_bin = NULL;
    void *d_temp_storage = NULL;
    int pirange = d_plan->spopts.pirange;

    cudaEventRecord(start);
    checkCudaErrors(cudaMemset(d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int)));
    CalcBinSize_noghost_3d<<<(M + 1024 - 1) / 1024, 1024>>>(M, nf1, nf2, nf3, bin_size_x, bin_size_y, bin_size_z,
                                                            numbins[0], numbins[1], numbins[2], d_binsize, d_kx, d_ky,
                                                            d_kz, d_sortidx, pirange);
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel CalcBinSize_noghost_3d \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_binsize; // For debug
    h_binsize = (int *)malloc(numbins[0] * numbins[1] * numbins[2] * sizeof(int));
    checkCudaErrors(
        cudaMemcpy(h_binsize, d_binsize, numbins[0] * numbins[1] * numbins[2] * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "[debug ] bin size:" << std::endl;
    for (int k = 0; k < numbins[2]; k++) {
        for (int j = 0; j < numbins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                std::cout << h_binsize[i + j * numbins[0] + k * numbins[0] * numbins[1]];
            }
            std::cout << std::endl;
        }
    }
    free(h_binsize);
    std::cout << "[debug ] ----------------------------------------------------" << std::endl;
#endif
#ifdef DEBUG
    int *h_sortidx;
    h_sortidx = (int *)malloc(M * sizeof(int));

    checkCudaErrors(cudaMemcpy(h_sortidx, d_sortidx, M * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 10; i++) {
        std::cout << "[debug ] ";
        std::cout << "point[" << setw(3) << i << "]=" << setw(3) << h_sortidx[i] << std::endl;
    }
#endif

    cudaEventRecord(start);
    int n = numbins[0] * numbins[1] * numbins[2];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel BinStartPts_3d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_binstartpts;
    h_binstartpts = (int *)malloc((numbins[0] * numbins[1] * numbins[2]) * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_binstartpts, d_binstartpts, (numbins[0] * numbins[1] * numbins[2]) * sizeof(int),
                               cudaMemcpyDeviceToHost));
    std::cout << "[debug ] Result of scan bin_size array:" << std::endl;
    for (int k = 0; k < numbins[2]; k++) {
        for (int j = 0; j < numbins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                std::cout << h_binstartpts[i + j * numbins[0] + k * numbins[0] * numbins[1]];
            }
            std::cout << std::endl;
        }
    }
    free(h_binstartpts);
    std::cout << "[debug ] ---------------------------------------------------" << std::endl;
#endif
    cudaEventRecord(start);
    CalcInvertofGlobalSortIdx_3d<<<(M + 1024 - 1) / 1024, 1024>>>(M, bin_size_x, bin_size_y, bin_size_z, numbins[0],
                                                                  numbins[1], numbins[2], d_binstartpts, d_sortidx,
                                                                  d_kx, d_ky, d_kz, d_idxnupts, pirange, nf1, nf2, nf3);
#ifdef DEBUG
    int *h_idxnupts;
    h_idxnupts = (int *)malloc(M * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_idxnupts, d_idxnupts, M * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; i++) {
        std::cout << "[debug ] idx=" << h_idxnupts[i] << std::endl;
    }
    free(h_idxnupts);
#endif
    /* --------------------------------------------- */
    //        Determining Subproblem properties      //
    /* --------------------------------------------- */
    cudaEventRecord(start);
    CalcSubProb_3d_v2<<<(M + 1024 - 1) / 1024, 1024>>>(d_binsize, d_numsubprob, maxsubprobsize,
                                                       numbins[0] * numbins[1] * numbins[2]);
#ifdef SPREADTIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel CalcSubProb_3d_v2\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
    int *h_numsubprob;
    h_numsubprob = (int *)malloc(n * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_numsubprob, d_numsubprob, numbins[0] * numbins[1] * numbins[2] * sizeof(int),
                               cudaMemcpyDeviceToHost));
    for (int k = 0; k < numbins[2]; k++) {
        for (int j = 0; j < numbins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                std::cout << h_numsubprob[i + j * numbins[0] + k * numbins[0] * numbins[1]];
            }
            std::cout << std::endl;
        }
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
    for (int k = 0; k < numbins[2]; k++) {
        for (int j = 0; j < numbins[1]; j++) {
            std::cout << "[debug ] ";
            for (int i = 0; i < numbins[0]; i++) {
                if (i != 0)
                    std::cout << " ";
                std::cout << h_subprobstartpts[i + j * numbins[0] + k * numbins[0] * numbins[1]];
            }
            std::cout << std::endl;
        }
    }
    printf("[debug ] Total number of subproblems = %d\n", h_subprobstartpts[n]);
    free(h_subprobstartpts);
#endif
    int totalnumsubprob;
    checkCudaErrors(cudaMemcpy(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMalloc(&d_subprob_to_bin, totalnumsubprob * sizeof(int)));
    MapBintoSubProb_3d_v2<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024>>>(
        d_subprob_to_bin, d_subprobstartpts, d_numsubprob, numbins[0] * numbins[1] * numbins[2]);
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
    std::cout << totalnumsubprob << std::endl;
    for (int j = 0; j < min(totalnumsubprob, 10); j++) {
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
    cudaFree(d_temp_storage);

    return 0;
}

template <typename T>
int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan, int blksize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
#ifdef INFO
    std::cout << "[info  ] Dividing the uniform grids to bin size[" << d_plan->opts.gpu_binsizex << "x"
              << d_plan->opts.gpu_binsizey << "x" << d_plan->opts.gpu_binsizez << "]" << std::endl;
    std::cout << "[info  ] numbins = [" << numbins[0] << "x" << numbins[1] << "]" << std::endl;
    std::cout << ns << std::endl;
#endif

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

    int totalnumsubprob = d_plan->totalnumsubprob;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;

    T sigma = d_plan->spopts.upsampfac;
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    int pirange = d_plan->spopts.pirange;
    cudaEventRecord(start);
    size_t sharedplanorysize = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) *
                               (bin_size_z + 2 * ceil(ns / 2.0)) * sizeof(cuda_complex<T>);
    if (sharedplanorysize > 49152) {
        std::cout << "error: not enough shared memory (" << sharedplanorysize << ")" << std::endl;
        return 1;
    }

    for (int t = 0; t < blksize; t++) {
        if (d_plan->opts.gpu_kerevalmeth) {
            Spread_3d_Subprob_Horner<<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, sigma, d_binstartpts,
                d_binsize, bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
                maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, pirange);
        } else {
            Spread_3d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c, es_beta,
                d_binstartpts, d_binsize, bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts,
                d_numsubprob, maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, pirange);
        }
    }
#ifdef SPREADTIME
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] \tKernel Spread_3d_Subprob (%d) \t%.3g ms\n", milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
    return 0;
}

} // namespace spreadinterp
} // namespace cufinufft
