#include <cuComplex.h>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>

#include <cufinufft/memtransfer.h>
#include <cufinufft/profile.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>

using namespace cufinufft::memtransfer;

#include "spreadinterp1d.cuh"

namespace cufinufft {
namespace spreadinterp {

template <typename T>
inline int cufinufft_interp1d(int nf1, cuda_complex<T> *d_fw, int M, T *d_kx, cuda_complex<T> *d_c,
                              cufinufft_plan_t<T> *d_plan)
/*
    This c function is written for only doing 1D interpolation. See
    test/interp1d_test.cu for usage.

    note: not allocate,transfer and free memories on gpu.
    Melody Shih 11/21/21
*/
{
    d_plan->nf1 = nf1;
    d_plan->M = M;
    d_plan->maxbatchsize = 1;

    d_plan->kx = d_kx;
    d_plan->c = d_c;
    d_plan->fw = d_fw;

    int ier;

    ier = allocgpumem1d_plan<T>(d_plan);
    ier = allocgpumem1d_nupts<T>(d_plan);

    if (d_plan->opts.gpu_method == 1) {
        ier = cuspread1d_nuptsdriven_prop<T>(nf1, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread1d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }
    if (d_plan->opts.gpu_method == 2) {
        ier = cuspread1d_subprob_prop<T>(nf1, M, d_plan);
        if (ier != 0) {
            printf("error: cuspread1d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
            return ier;
        }
    }

    ier = cuinterp1d<T>(d_plan, 1);
    freegpumemory1d<T>(d_plan);

    return ier;
}

template int cufinufft_interp1d(int nf1, cuda_complex<float> *d_fw, int M, float *d_kx, cuda_complex<float> *d_c,
                                cufinufft_plan_t<float> *d_plan);
template int cufinufft_interp1d(int nf1, cuda_complex<double> *d_fw, int M, double *d_kx, cuda_complex<double> *d_c,
                                cufinufft_plan_t<double> *d_plan);

template <typename T>
int cuinterp1d(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    A wrapper for different interpolation methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem

    Melody Shih 11/21/21
*/
{
    int nf1 = d_plan->nf1;
    int M = d_plan->M;

    int ier;
    switch (d_plan->opts.gpu_method) {
    case 1: {
        ier = cuinterp1d_nuptsdriven<T>(nf1, M, d_plan, blksize);
        if (ier != 0) {
            std::cout << "error: cnufftspread1d_gpu_nuptsdriven" << std::endl;
            return 1;
        }
    } break;
    default:
        std::cout << "error: incorrect method, should be 1" << std::endl;
        return 2;
    }

    return ier;
}

template <typename T>
int cuinterp1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
    dim3 threadsPerBlock;
    dim3 blocks;

    int ns = d_plan->spopts.nspread; // psi's support in terms of number of cells
    T es_c = d_plan->spopts.ES_c;
    T es_beta = d_plan->spopts.ES_beta;
    T sigma = d_plan->opts.upsampfac;
    int pirange = d_plan->spopts.pirange;
    int *d_idxnupts = d_plan->idxnupts;

    T *d_kx = d_plan->kx;
    cuda_complex<T> *d_c = d_plan->c;
    cuda_complex<T> *d_fw = d_plan->fw;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    blocks.x = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = 1;

    if (d_plan->opts.gpu_kerevalmeth) {
        for (int t = 0; t < blksize; t++) {
            interp_1d_nuptsdriven_horner<<<blocks, threadsPerBlock>>>(d_kx, d_c + t * M, d_fw + t * nf1, M, ns, nf1,
                                                                      sigma, d_idxnupts, pirange);
        }
    } else {
        for (int t = 0; t < blksize; t++) {
            interp_1d_nuptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_c + t * M, d_fw + t * nf1, M, ns, nf1, es_c,
                                                               es_beta, d_idxnupts, pirange);
        }
    }

    return 0;
}

template int cuinterp1d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp1d<double>(cufinufft_plan_t<double> *d_plan, int blksize);
} // namespace spreadinterp
} // namespace cufinufft
