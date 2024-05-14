#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <thrust/extrema.h>

#include <cufinufft/defs.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>
using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {
/* ------------------------ 1d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

template <typename T, int KEREVALMETH>
__global__ void spread_1d_nuptsdriven(const T *x, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, int ns, int nf1,
                                      T es_c, T es_beta, T sigma, const int *idxnupts) {
    int xx, ix;
    T ker1[MAX_NSPREAD];

    T x_rescaled;
    cuda_complex<T> cnow;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        x_rescaled = fold_rescale(x[idxnupts[i]], nf1);
        cnow = c[idxnupts[i]];
        int xstart = ceil(x_rescaled - ns / 2.0);
        int xend = floor(x_rescaled + ns / 2.0);

        T x1 = (T)xstart - x_rescaled;
        if constexpr (KEREVALMETH == 1)
            eval_kernel_vec_horner(ker1, x1, ns, sigma);
        else
            eval_kernel_vec(ker1, x1, ns, es_c, es_beta);

        for (xx = xstart; xx <= xend; xx++) {
            ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
            T kervalue = ker1[xx - xstart];
            atomicAdd(&fw[ix].x, cnow.x * kervalue);
            atomicAdd(&fw[ix].y, cnow.y * kervalue);
        }
    }
}

/* Kernels for SubProb Method */
// SubProb properties
template <typename T>
__global__ void calc_bin_size_noghost_1d(int M, int nf1, int bin_size_x, int nbinx, int *bin_size, const T *x,
                                         int *sortidx) {
    int binx;
    int oldidx;
    T x_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        x_rescaled = fold_rescale(x[i], nf1);
        binx = floor(x_rescaled / bin_size_x);
        binx = binx >= nbinx ? binx - 1 : binx;
        binx = binx < 0 ? 0 : binx;
        oldidx = atomicAdd(&bin_size[binx], 1);
        sortidx[i] = oldidx;
        if (binx >= nbinx) {
            sortidx[i] = -binx;
        }
    }
}

template <typename T>
__global__ void calc_inverse_of_global_sort_idx_1d(int M, int bin_size_x, int nbinx, const int *bin_startpts,
                                                   const int *sortidx, const T *x, int *index, int nf1) {
    int binx;
    T x_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        x_rescaled = fold_rescale(x[i], nf1);
        binx = floor(x_rescaled / bin_size_x);
        binx = binx >= nbinx ? binx - 1 : binx;
        binx = binx < 0 ? 0 : binx;

        index[bin_startpts[binx] + sortidx[i]] = i;
    }
}

template <typename T, int KEREVALMETH>
__global__ void spread_1d_subprob(const T *x, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, int ns, int nf1,
                                  T es_c, T es_beta, T sigma, const int *binstartpts, const int *bin_size,
                                  int bin_size_x, const int *subprob_to_bin, const int *subprobstartpts,
                                  const int *numsubprob, int maxsubprobsize, int nbinx, const int *idxnupts) {
    extern __shared__ char sharedbuf[];
    cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;

    int xstart, xend;
    int subpidx = blockIdx.x;
    int bidx = subprob_to_bin[subpidx];
    int binsubp_idx = subpidx - subprobstartpts[bidx];
    int ix;
    int ptstart = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
    int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

    int xoffset = (bidx % nbinx) * bin_size_x;

    int N = (bin_size_x + 2 * ceil(ns / 2.0));
    T ker1[MAX_NSPREAD];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        fwshared[i].x = 0.0;
        fwshared[i].y = 0.0;
    }
    __syncthreads();

    T x_rescaled;
    cuda_complex<T> cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int idx = ptstart + i;
        x_rescaled = fold_rescale(x[idxnupts[idx]], nf1);
        cnow = c[idxnupts[idx]];

        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        xend = floor(x_rescaled + ns / 2.0) - xoffset;

        T x1 = (T)xstart + xoffset - x_rescaled;
        if constexpr (KEREVALMETH == 1)
            eval_kernel_vec_horner(ker1, x1, ns, sigma);
        else
            eval_kernel_vec(ker1, x1, ns, es_c, es_beta);

        for (int xx = xstart; xx <= xend; xx++) {
            ix = xx + ceil(ns / 2.0);
            if (ix >= (bin_size_x + (int)ceil(ns / 2.0) * 2) || ix < 0)
                break;
            atomicAdd(&fwshared[ix].x, cnow.x * ker1[xx - xstart]);
            atomicAdd(&fwshared[ix].y, cnow.y * ker1[xx - xstart]);
        }
    }
    __syncthreads();
    /* write to global memory */
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        ix = xoffset - ceil(ns / 2.0) + k;
        if (ix < (nf1 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            atomicAdd(&fw[ix].x, fwshared[k].x);
            atomicAdd(&fw[ix].y, fwshared[k].y);
        }
    }
}

/* --------------------- 1d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
template <typename T, int KEREVALMETH>
__global__ void interp_1d_nuptsdriven(const T *x, cuda_complex<T> *c, const cuda_complex<T> *fw, int M, int ns, int nf1,
                                      T es_c, T es_beta, T sigma, const int *idxnupts) {
    T ker1[MAX_NSPREAD];
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        T x_rescaled = fold_rescale(x[idxnupts[i]], nf1);

        int xstart = ceil(x_rescaled - ns / 2.0);
        int xend = floor(x_rescaled + ns / 2.0);
        cuda_complex<T> cnow;
        cnow.x = 0.0;
        cnow.y = 0.0;

        T x1 = (T)xstart - x_rescaled;
        if constexpr (KEREVALMETH == 1)
            eval_kernel_vec_horner(ker1, x1, ns, sigma);
        else
            eval_kernel_vec(ker1, x1, ns, es_c, es_beta);

        for (int xx = xstart; xx <= xend; xx++) {
            int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
            T kervalue1 = ker1[xx - xstart];
            cnow.x += fw[ix].x * kervalue1;
            cnow.y += fw[ix].y * kervalue1;
        }
        c[idxnupts[i]].x = cnow.x;
        c[idxnupts[i]].y = cnow.y;
    }
}

} // namespace spreadinterp
} // namespace cufinufft
