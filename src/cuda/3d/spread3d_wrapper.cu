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

#include "spreadinterp3d.cuh"

namespace cufinufft {
namespace spreadinterp {

template <typename T>
int cufinufft_spread3d(int nf1, int nf2, int nf3, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, T *d_kz,
                       cuda_complex<T> *d_c, cufinufft_plan_t<T> *d_plan)
/*
    This c function is written for only doing 3D spreading. See
    test/spread3d_test.cu for usage.

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

    ier = allocgpumem3d_plan<T>(d_plan);
    ier = allocgpumem3d_nupts<T>(d_plan);

    if (d_plan->opts.gpu_method == 1) {
        ier = cuspread3d_nuptsdriven_prop<T>(nf1, nf2, nf3, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }
    if (d_plan->opts.gpu_method == 2) {
        ier = cuspread3d_subprob_prop<T>(nf1, nf2, nf3, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread3d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }
    if (d_plan->opts.gpu_method == 4) {
        ier = cuspread3d_blockgather_prop<T>(nf1, nf2, nf3, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread3d_blockgather_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    ier = cuspread3d<T>(d_plan, 1);
    freegpumemory3d<T>(d_plan);

    return ier;
}

template <typename T>
int cuspread3d(cufinufft_plan_t<T> *d_plan, int blksize)
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

    int ier = 0;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        ier = cuspread3d_nuptsdriven<T>(nf1, nf2, nf3, M, d_plan, blksize);

        if (ier != 0) {
            std::cout << "error: cnufftspread3d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    case 2: {
        ier = cuspread3d_subprob<T>(nf1, nf2, nf3, M, d_plan, blksize);

        if (ier != 0) {
            std::cout << "error: cnufftspread3d_gpu_subprob" << std::endl;
            return 1;
        }
    } break;
    case 4: {
        ier = cuspread3d_blockgather<T>(nf1, nf2, nf3, M, d_plan, blksize);

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
int cuspread3d_nuptsdriven_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan) {
    auto &stream = d_plan->streams[d_plan->curr_stream];

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

        T *d_kx = d_plan->kx;
        T *d_ky = d_plan->ky;
        T *d_kz = d_plan->kz;

        int *d_binsize = d_plan->binsize;
        int *d_binstartpts = d_plan->binstartpts;
        int *d_sortidx = d_plan->sortidx;
        int *d_idxnupts = d_plan->idxnupts;

        int pirange = d_plan->spopts.pirange;

        checkCudaErrors(cudaMemsetAsync(d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));
        calc_bin_size_noghost_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
            M, nf1, nf2, nf3, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2], d_binsize, d_kx,
            d_ky, d_kz, d_sortidx, pirange);

        int n = numbins[0] * numbins[1] * numbins[2];
        thrust::device_ptr<int> d_ptr(d_binsize);
        thrust::device_ptr<int> d_result(d_binstartpts);
        thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

        calc_inverse_of_global_sort_index_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
            M, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2], d_binstartpts, d_sortidx, d_kx,
            d_ky, d_kz, d_idxnupts, pirange, nf1, nf2, nf3);
    } else {
        int *d_idxnupts = d_plan->idxnupts;

        trivial_global_sort_index_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(M, d_idxnupts);
    }

    return 0;
}

template <typename T>
int cuspread3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    auto &stream = d_plan->streams[d_plan->curr_stream];

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

    if (d_plan->opts.gpu_kerevalmeth == 1) {
        for (int t = 0; t < blksize; t++) {
            spread_3d_nupts_driven_horner<<<blocks, threadsPerBlock, 0, stream>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, sigma, d_idxnupts,
                pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            spread_3d_nupts_driven<<<blocks, threadsPerBlock, 0, stream>>>(d_kx, d_ky, d_kz, d_c + t * M,
                                                                           d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2,
                                                                           nf3, es_c, es_beta, d_idxnupts, pirange);
        }
    }

    return 0;
}

template <typename T>
int cuspread3d_blockgather_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan) {
    auto &stream = d_plan->streams[d_plan->curr_stream];

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

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;

    int *d_binsize = d_plan->binsize;
    int *d_sortidx = d_plan->sortidx;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_idxnupts = NULL;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_subprob_to_bin = NULL;

    checkCudaErrors(cudaMemsetAsync(d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));
    locate_nupts_to_bins_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2], binsperobinx, binsperobiny,
        binsperobinz, d_binsize, d_kx, d_ky, d_kz, d_sortidx, pirange, nf1, nf2, nf3);

    threadsPerBlock.x = 8;
    threadsPerBlock.y = 8;
    threadsPerBlock.z = 8;

    blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
    blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
    blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

    fill_ghost_bins<<<blocks, threadsPerBlock, 0, stream>>>(binsperobinx, binsperobiny, binsperobinz, numobins[0],
                                                            numobins[1], numobins[2], d_binsize);

    int n = numbins[0] * numbins[1] * numbins[2];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemsetAsync(d_binstartpts, 0, sizeof(int), stream));

    int totalNUpts;
    checkCudaErrors(cudaMemcpyAsync(&totalNUpts, &d_binstartpts[n], sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMallocAsync(&d_idxnupts, totalNUpts * sizeof(int), stream));

    calc_inverse_of_global_sort_index_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2], binsperobinx, binsperobiny,
        binsperobinz, d_binstartpts, d_sortidx, d_kx, d_ky, d_kz, d_idxnupts, pirange, nf1, nf2, nf3);

    threadsPerBlock.x = 2;
    threadsPerBlock.y = 2;
    threadsPerBlock.z = 2;

    blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
    blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
    blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

    ghost_bin_pts_index<<<blocks, threadsPerBlock, 0, stream>>>(binsperobinx, binsperobiny, binsperobinz, numobins[0],
                                                                numobins[1], numobins[2], d_binsize, d_idxnupts,
                                                                d_binstartpts, M);
    if (d_plan->idxnupts != nullptr)
        cudaFreeAsync(d_plan->idxnupts, stream);
    d_plan->idxnupts = d_idxnupts;

    /* --------------------------------------------- */
    //        Determining Subproblem properties      //
    /* --------------------------------------------- */
    n = numobins[0] * numobins[1] * numobins[2];
    calc_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(binsperobinx, binsperobiny, binsperobinz, d_binsize,
                                                                   d_numsubprob, maxsubprobsize,
                                                                   numobins[0] * numobins[1] * numobins[2]);

    n = numobins[0] * numobins[1] * numobins[2];
    d_ptr = thrust::device_pointer_cast(d_numsubprob);
    d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream));

    int totalnumsubprob;
    checkCudaErrors(
        cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMallocAsync(&d_subprob_to_bin, totalnumsubprob * sizeof(int), stream));
    map_b_into_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(d_subprob_to_bin, d_subprobstartpts,
                                                                         d_numsubprob, n);
    assert(d_subprob_to_bin != nullptr);
    if (d_plan->subprob_to_bin != nullptr)
        cudaFreeAsync(d_plan->subprob_to_bin, stream);
    d_plan->subprob_to_bin = d_subprob_to_bin;
    d_plan->totalnumsubprob = totalnumsubprob;

    return 0;
}

template <typename T>
int cuspread3d_blockgather(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    auto &stream = d_plan->streams[d_plan->curr_stream];

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

    for (int t = 0; t < blksize; t++) {
        if (d_plan->opts.gpu_kerevalmeth == 1) {
            size_t sharedplanorysize = obin_size_x * obin_size_y * obin_size_z * sizeof(cuda_complex<T>);
            if (sharedplanorysize > 49152) {
                std::cout << "error: not enough shared memory" << std::endl;
                return 1;
            }
            spread_3d_block_gather_horner<<<totalnumsubprob, 64, sharedplanorysize, stream>>>(
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
            spread_3d_block_gather<<<totalnumsubprob, 64, sharedplanorysize, stream>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c, es_beta, sigma,
                d_binstartpts, obin_size_x, obin_size_y, obin_size_z, binsperobinx * binsperobiny * binsperobinz,
                d_subprob_to_bin, d_subprobstartpts, maxsubprobsize, numobins[0], numobins[1], numobins[2], d_idxnupts,
                pirange);
        }
    }

    return 0;
}

template <typename T>
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan) {
    auto &stream = d_plan->streams[d_plan->curr_stream];

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

    T *d_kx = d_plan->kx;
    T *d_ky = d_plan->ky;
    T *d_kz = d_plan->kz;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx = d_plan->sortidx;
    int *d_numsubprob = d_plan->numsubprob;
    int *d_subprobstartpts = d_plan->subprobstartpts;
    int *d_idxnupts = d_plan->idxnupts;

    int *d_subprob_to_bin = NULL;
    int pirange = d_plan->spopts.pirange;

    checkCudaErrors(cudaMemsetAsync(d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));
    calc_bin_size_noghost_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, nf1, nf2, nf3, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2], d_binsize, d_kx, d_ky,
        d_kz, d_sortidx, pirange);

    int n = numbins[0] * numbins[1] * numbins[2];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);

    calc_inverse_of_global_sort_index_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2], d_binstartpts, d_sortidx, d_kx, d_ky,
        d_kz, d_idxnupts, pirange, nf1, nf2, nf3);
    /* --------------------------------------------- */
    //        Determining Subproblem properties      //
    /* --------------------------------------------- */
    calc_subprob_3d_v2<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(d_binsize, d_numsubprob, maxsubprobsize,
                                                                   numbins[0] * numbins[1] * numbins[2]);

    d_ptr = thrust::device_pointer_cast(d_numsubprob);
    d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    checkCudaErrors(cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream));

    int totalnumsubprob;
    checkCudaErrors(
        cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMallocAsync(&d_subprob_to_bin, totalnumsubprob * sizeof(int), stream));
    map_b_into_subprob_3d_v2<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024, 0, stream>>>(
        d_subprob_to_bin, d_subprobstartpts, d_numsubprob, numbins[0] * numbins[1] * numbins[2]);
    assert(d_subprob_to_bin != nullptr);
    if (d_plan->subprob_to_bin != nullptr)
        cudaFreeAsync(d_plan->subprob_to_bin, stream);
    d_plan->subprob_to_bin = d_subprob_to_bin;
    assert(d_plan->subprob_to_bin != nullptr);
    d_plan->totalnumsubprob = totalnumsubprob;

    return 0;
}

template <typename T>
int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    auto &stream = d_plan->streams[d_plan->curr_stream];

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

    int totalnumsubprob = d_plan->totalnumsubprob;
    int *d_subprob_to_bin = d_plan->subprob_to_bin;

    T sigma = d_plan->spopts.upsampfac;
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    int pirange = d_plan->spopts.pirange;
    size_t sharedplanorysize = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) *
                               (bin_size_z + 2 * ceil(ns / 2.0)) * sizeof(cuda_complex<T>);
    if (sharedplanorysize > 49152) {
        std::cout << "error: not enough shared memory (" << sharedplanorysize << ")" << std::endl;
        return 1;
    }

    for (int t = 0; t < blksize; t++) {
        if (d_plan->opts.gpu_kerevalmeth) {
            spread_3d_subprob_horner<<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, sigma, d_binstartpts,
                d_binsize, bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
                maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, pirange);
        } else {
            spread_3d_subprob<<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
                d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, ns, nf1, nf2, nf3, es_c, es_beta,
                d_binstartpts, d_binsize, bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts,
                d_numsubprob, maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, pirange);
        }
    }

    return 0;
}

template int cuspread3d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuspread3d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

template int cufinufft_spread3d<float>(int nf1, int nf2, int nf3, cuda_complex<float> *d_fw, int M, float *d_kx,
                                       float *d_ky, float *d_kz, cuda_complex<float> *d_c,
                                       cufinufft_plan_t<float> *d_plan);
template int cufinufft_spread3d<double>(int nf1, int nf2, int nf3, cuda_complex<double> *d_fw, int M, double *d_kx,
                                        double *d_ky, double *d_kz, cuda_complex<double> *d_c,
                                        cufinufft_plan_t<double> *d_plan);
} // namespace spreadinterp
} // namespace cufinufft
