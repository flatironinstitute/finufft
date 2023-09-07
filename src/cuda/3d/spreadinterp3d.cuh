#include <cmath>
#include <iostream>

#include <cuda.h>
#include <helper_cuda.h>

#include <cufinufft/defs.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

namespace cufinufft {
namespace spreadinterp {
/* ---------------------- 3d Spreading Kernels -------------------------------*/
/* Kernels for bin sort NUpts */

template <typename T>
__global__ void calc_bin_size_noghost_3d(int M, int nf1, int nf2, int nf3, int bin_size_x, int bin_size_y,
                                         int bin_size_z, int nbinx, int nbiny, int nbinz, int *bin_size, const T *x,
                                         const T *y, const T *z, int *sortidx, int pirange) {
    int binidx, binx, biny, binz;
    int oldidx;
    T x_rescaled, y_rescaled, z_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        x_rescaled = RESCALE(x[i], nf1, pirange);
        y_rescaled = RESCALE(y[i], nf2, pirange);
        z_rescaled = RESCALE(z[i], nf3, pirange);
        binx = floor(x_rescaled / bin_size_x);
        binx = binx >= nbinx ? binx - 1 : binx;
        binx = binx < 0 ? 0 : binx;

        biny = floor(y_rescaled / bin_size_y);
        biny = biny >= nbiny ? biny - 1 : biny;
        biny = biny < 0 ? 0 : biny;

        binz = floor(z_rescaled / bin_size_z);
        binz = binz >= nbinz ? binz - 1 : binz;
        binz = binz < 0 ? 0 : binz;
        binidx = binx + biny * nbinx + binz * nbinx * nbiny;
        oldidx = atomicAdd(&bin_size[binidx], 1);
        sortidx[i] = oldidx;
    }
}

template <typename T>
__global__ void calc_inverse_of_global_sort_index_3d(int M, int bin_size_x, int bin_size_y, int bin_size_z, int nbinx,
                                                     int nbiny, int nbinz, const int *bin_startpts, const int *sortidx,
                                                     const T *x, const T *y, const T *z, int *index, int pirange,
                                                     int nf1, int nf2, int nf3) {
    int binx, biny, binz;
    int binidx;
    T x_rescaled, y_rescaled, z_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        x_rescaled = RESCALE(x[i], nf1, pirange);
        y_rescaled = RESCALE(y[i], nf2, pirange);
        z_rescaled = RESCALE(z[i], nf3, pirange);
        binx = floor(x_rescaled / bin_size_x);
        binx = binx >= nbinx ? binx - 1 : binx;
        binx = binx < 0 ? 0 : binx;
        biny = floor(y_rescaled / bin_size_y);
        biny = biny >= nbiny ? biny - 1 : biny;
        biny = biny < 0 ? 0 : biny;
        binz = floor(z_rescaled / bin_size_z);
        binz = binz >= nbinz ? binz - 1 : binz;
        binz = binz < 0 ? 0 : binz;
        binidx = common::calc_global_index_v2(binx, biny, binz, nbinx, nbiny, nbinz);

        index[bin_startpts[binidx] + sortidx[i]] = i;
    }
}

/* Kernels for NUptsdriven method */

template <typename T>
__global__ void spread_3d_nupts_driven_horner(const T *x, const T *y, const T *z, const cuda_complex<T> *c,
                                              cuda_complex<T> *fw, int M, int ns, int nf1, int nf2, int nf3, T sigma,
                                              const int *idxnupts, int pirange) {
    int xx, yy, zz, ix, iy, iz;
    int outidx;
    T ker1[MAX_NSPREAD];
    T ker2[MAX_NSPREAD];
    T ker3[MAX_NSPREAD];

    T ker1val, ker2val, ker3val;

    T x_rescaled, y_rescaled, z_rescaled;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
        z_rescaled = RESCALE(z[idxnupts[i]], nf3, pirange);

        int xstart = ceil(x_rescaled - ns / 2.0);
        int ystart = ceil(y_rescaled - ns / 2.0);
        int zstart = ceil(z_rescaled - ns / 2.0);
        int xend = floor(x_rescaled + ns / 2.0);
        int yend = floor(y_rescaled + ns / 2.0);
        int zend = floor(z_rescaled + ns / 2.0);

        T x1 = (T)xstart - x_rescaled;
        T y1 = (T)ystart - y_rescaled;
        T z1 = (T)zstart - z_rescaled;

        eval_kernel_vec_horner(ker1, x1, ns, sigma);
        eval_kernel_vec_horner(ker2, y1, ns, sigma);
        eval_kernel_vec_horner(ker3, z1, ns, sigma);
        for (zz = zstart; zz <= zend; zz++) {
            ker3val = ker3[zz - zstart];
            for (yy = ystart; yy <= yend; yy++) {
                ker2val = ker2[yy - ystart];
                for (xx = xstart; xx <= xend; xx++) {
                    ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
                    iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
                    iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
                    outidx = ix + iy * nf1 + iz * nf1 * nf2;
                    ker1val = ker1[xx - xstart];
                    T kervalue = ker1val * ker2val * ker3val;
                    atomicAdd(&fw[outidx].x, c[idxnupts[i]].x * kervalue);
                    atomicAdd(&fw[outidx].y, c[idxnupts[i]].y * kervalue);
                }
            }
        }
    }
}

template <typename T>
__global__ void spread_3d_nupts_driven(const T *x, const T *y, const T *z, const cuda_complex<T> *c,
                                       cuda_complex<T> *fw, int M, int ns, int nf1, int nf2, int nf3, T es_c, T es_beta,
                                       const int *idxnupts, int pirange) {
    int xx, yy, zz, ix, iy, iz;
    int outidx;
    T ker1[MAX_NSPREAD];
    T ker2[MAX_NSPREAD];
    T ker3[MAX_NSPREAD];

    T x_rescaled, y_rescaled, z_rescaled;
    T ker1val, ker2val, ker3val;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
        z_rescaled = RESCALE(z[idxnupts[i]], nf3, pirange);

        int xstart = ceil(x_rescaled - ns / 2.0);
        int ystart = ceil(y_rescaled - ns / 2.0);
        int zstart = ceil(z_rescaled - ns / 2.0);
        int xend = floor(x_rescaled + ns / 2.0);
        int yend = floor(y_rescaled + ns / 2.0);
        int zend = floor(z_rescaled + ns / 2.0);

        T x1 = (T)xstart - x_rescaled;
        T y1 = (T)ystart - y_rescaled;
        T z1 = (T)zstart - z_rescaled;

        eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
        eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
        eval_kernel_vec(ker3, z1, ns, es_c, es_beta);
        for (zz = zstart; zz <= zend; zz++) {
            ker3val = ker3[zz - zstart];
            for (yy = ystart; yy <= yend; yy++) {
                ker2val = ker2[yy - ystart];
                for (xx = xstart; xx <= xend; xx++) {
                    ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
                    iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
                    iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
                    outidx = ix + iy * nf1 + iz * nf1 * nf2;

                    ker1val = ker1[xx - xstart];
                    T kervalue = ker1val * ker2val * ker3val;

                    atomicAdd(&fw[outidx].x, c[idxnupts[i]].x * kervalue);
                    atomicAdd(&fw[outidx].y, c[idxnupts[i]].y * kervalue);
                }
            }
        }
    }
}

/* Kernels for Subprob method */

template <typename T, int KEREVALMETH>
__global__ void spread_3d_subprob(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M, int ns, int nf1,
                                  int nf2, int nf3, T sigma, T es_c, T es_beta, int *binstartpts, int *bin_size,
                                  int bin_size_x, int bin_size_y, int bin_size_z, int *subprob_to_bin,
                                  int *subprobstartpts, int *numsubprob, int maxsubprobsize, int nbinx, int nbiny,
                                  int nbinz, int *idxnupts, int pirange) {
    extern __shared__ char sharedbuf[];
    cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;

    const int bidx = subprob_to_bin[blockIdx.x];
    const int binsubp_idx = blockIdx.x - subprobstartpts[bidx];
    const int ptstart = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
    const int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

    const int xoffset = (bidx % nbinx) * bin_size_x;
    const int yoffset = ((bidx / nbinx) % nbiny) * bin_size_y;
    const int zoffset = (bidx / (nbinx * nbiny)) * bin_size_z;

    int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) * (bin_size_z + 2 * ceil(ns / 2.0));

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        fwshared[i].x = 0.0;
        fwshared[i].y = 0.0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        T ker1[MAX_NSPREAD];
        T ker2[MAX_NSPREAD];
        T ker3[MAX_NSPREAD];

        const int nuptsidx = idxnupts[ptstart + i];
        const T x_rescaled = RESCALE(x[nuptsidx], nf1, pirange);
        const T y_rescaled = RESCALE(y[nuptsidx], nf2, pirange);
        const T z_rescaled = RESCALE(z[nuptsidx], nf3, pirange);
        cuda_complex<T> cnow = c[nuptsidx];

        const int xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        const int ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        const int zstart = ceil(z_rescaled - ns / 2.0) - zoffset;

        const int xend = floor(x_rescaled + ns / 2.0) - xoffset;
        const int yend = floor(y_rescaled + ns / 2.0) - yoffset;
        const int zend = floor(z_rescaled + ns / 2.0) - zoffset;

        if constexpr (KEREVALMETH == 1) {
            eval_kernel_vec_horner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
            eval_kernel_vec_horner(ker2, ystart + yoffset - y_rescaled, ns, sigma);
            eval_kernel_vec_horner(ker3, zstart + zoffset - z_rescaled, ns, sigma);
        } else {
            eval_kernel_vec(ker1, xstart + xoffset - x_rescaled, ns, es_c, es_beta);
            eval_kernel_vec(ker2, ystart + yoffset - y_rescaled, ns, es_c, es_beta);
            eval_kernel_vec(ker3, zstart + zoffset - z_rescaled, ns, es_c, es_beta);
        }

        for (int zz = zstart; zz <= zend; zz++) {
            const T kervalue3 = ker3[zz - zstart];
            const int iz = zz + ceil(ns / 2.0);
            if (iz >= (bin_size_z + (int)ceil(ns / 2.0) * 2) || iz < 0)
                break;
            for (int yy = ystart; yy <= yend; yy++) {
                const T kervalue2 = ker2[yy - ystart];
                const int iy = yy + ceil(ns / 2.0);
                if (iy >= (bin_size_y + (int)ceil(ns / 2.0) * 2) || iy < 0)
                    break;
                for (int xx = xstart; xx <= xend; xx++) {
                    const int ix = xx + ceil(ns / 2.0);
                    if (ix >= (bin_size_x + (int)ceil(ns / 2.0) * 2) || ix < 0)
                        break;
                    const int outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2) +
                                       iz * (bin_size_x + ceil(ns / 2.0) * 2) * (bin_size_y + ceil(ns / 2.0) * 2);
                    const T kervalue1 = ker1[xx - xstart];
                    atomicAdd(&fwshared[outidx].x, cnow.x * kervalue1 * kervalue2 * kervalue3);
                    atomicAdd(&fwshared[outidx].y, cnow.y * kervalue1 * kervalue2 * kervalue3);
                }
            }
        }
    }
    __syncthreads();

    /* write to global memory */
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        const int i = n % (int)(bin_size_x + 2 * ceil(ns / 2.0));
        const int j = (int)(n / (bin_size_x + 2 * ceil(ns / 2.0))) % (int)(bin_size_y + 2 * ceil(ns / 2.0));
        const int k = n / ((bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)));

        int ix = xoffset - ceil(ns / 2.0) + i;
        int iy = yoffset - ceil(ns / 2.0) + j;
        int iz = zoffset - ceil(ns / 2.0) + k;

        if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0)) && iz < (nf3 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
            iz = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
            const int outidx = ix + iy * nf1 + iz * nf1 * nf2;
            const int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2) +
                                  k * (bin_size_x + ceil(ns / 2.0) * 2) * (bin_size_y + ceil(ns / 2.0) * 2);
            atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
            atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
        }
    }
}

/* Kernels for BlockGather Method */
template <typename T>
__global__ void locate_nupts_to_bins_ghost(int M, int bin_size_x, int bin_size_y, int bin_size_z, int nobinx,
                                           int nobiny, int nobinz, int binsperobinx, int binsperobiny, int binsperobinz,
                                           int *bin_size, const T *x, const T *y, const T *z, int *sortidx, int pirange,
                                           int nf1, int nf2, int nf3) {
    int binidx, binx, biny, binz;
    int oldidx;
    T x_rescaled, y_rescaled, z_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        x_rescaled = RESCALE(x[i], nf1, pirange);
        y_rescaled = RESCALE(y[i], nf2, pirange);
        z_rescaled = RESCALE(z[i], nf3, pirange);
        binx = floor(x_rescaled / bin_size_x);
        biny = floor(y_rescaled / bin_size_y);
        binz = floor(z_rescaled / bin_size_z);
        binx = binx / (binsperobinx - 2) * binsperobinx + (binx % (binsperobinx - 2) + 1);
        biny = biny / (binsperobiny - 2) * binsperobiny + (biny % (binsperobiny - 2) + 1);
        binz = binz / (binsperobinz - 2) * binsperobinz + (binz % (binsperobinz - 2) + 1);

        binidx = common::calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz, binsperobinx, binsperobiny,
                                           binsperobinz);
        oldidx = atomicAdd(&bin_size[binidx], 1);
        sortidx[i] = oldidx;
    }
}

template <typename T>
__global__ void calc_inverse_of_global_sort_index_ghost(int M, int bin_size_x, int bin_size_y, int bin_size_z,
                                                        int nobinx, int nobiny, int nobinz, int binsperobinx,
                                                        int binsperobiny, int binsperobinz, int *bin_startpts,
                                                        const int *sortidx, const T *x, const T *y, const T *z,
                                                        int *index, int pirange, int nf1, int nf2, int nf3) {
    int binx, biny, binz;
    int binidx;
    T x_rescaled, y_rescaled, z_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        x_rescaled = RESCALE(x[i], nf1, pirange);
        y_rescaled = RESCALE(y[i], nf2, pirange);
        z_rescaled = RESCALE(z[i], nf3, pirange);
        binx = floor(x_rescaled / bin_size_x);
        biny = floor(y_rescaled / bin_size_y);
        binz = floor(z_rescaled / bin_size_z);
        binx = binx / (binsperobinx - 2) * binsperobinx + (binx % (binsperobinx - 2) + 1);
        biny = biny / (binsperobiny - 2) * binsperobiny + (biny % (binsperobiny - 2) + 1);
        binz = binz / (binsperobinz - 2) * binsperobinz + (binz % (binsperobinz - 2) + 1);

        binidx = common::calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz, binsperobinx, binsperobiny,
                                           binsperobinz);

        index[bin_startpts[binidx] + sortidx[i]] = i;
    }
}

template <typename T>
__global__ void spread_3d_block_gather(const T *x, const T *y, const T *z, const cuda_complex<T> *c,
                                       cuda_complex<T> *fw, int M, int ns, int nf1, int nf2, int nf3, T es_c, T es_beta,
                                       T sigma, const int *binstartpts, int obin_size_x, int obin_size_y,
                                       int obin_size_z, int binsperobin, const int *subprob_to_bin,
                                       const int *subprobstartpts, int maxsubprobsize, int nobinx, int nobiny,
                                       int nobinz, const int *idxnupts, int pirange) {
    extern __shared__ char sharedbuf[];
    cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;

    int xstart, ystart, zstart, xend, yend, zend;
    int subpidx = blockIdx.x;
    int obidx = subprob_to_bin[subpidx];
    int bidx = obidx * binsperobin;

    int obinsubp_idx = subpidx - subprobstartpts[obidx];
    int ix, iy, iz;
    int outidx;
    int ptstart = binstartpts[bidx] + obinsubp_idx * maxsubprobsize;
    int nupts =
        min(maxsubprobsize, binstartpts[bidx + binsperobin] - binstartpts[bidx] - obinsubp_idx * maxsubprobsize);

    int xoffset = (obidx % nobinx) * obin_size_x;
    int yoffset = (obidx / nobinx) % nobiny * obin_size_y;
    int zoffset = (obidx / (nobinx * nobiny)) * obin_size_z;

    int N = obin_size_x * obin_size_y * obin_size_z;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        fwshared[i].x = 0.0;
        fwshared[i].y = 0.0;
    }
    __syncthreads();
    T x_rescaled, y_rescaled, z_rescaled;
    cuda_complex<T> cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int idx = ptstart + i;
        int b = idxnupts[idx] / M;
        int box[3];
        for (int d = 0; d < 3; d++) {
            box[d] = b % 3;
            if (box[d] == 1)
                box[d] = -1;
            if (box[d] == 2)
                box[d] = 1;
            b = b / 3;
        }
        int ii = idxnupts[idx] % M;
        x_rescaled = RESCALE(x[ii], nf1, pirange) + box[0] * nf1;
        y_rescaled = RESCALE(y[ii], nf2, pirange) + box[1] * nf2;
        z_rescaled = RESCALE(z[ii], nf3, pirange) + box[2] * nf3;
        cnow = c[ii];

        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        xstart = xstart < 0 ? 0 : xstart;
        ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        ystart = ystart < 0 ? 0 : ystart;
        zstart = ceil(z_rescaled - ns / 2.0) - zoffset;
        zstart = zstart < 0 ? 0 : zstart;
        xend = floor(x_rescaled + ns / 2.0) - xoffset;
        xend = xend >= obin_size_x ? obin_size_x - 1 : xend;
        yend = floor(y_rescaled + ns / 2.0) - yoffset;
        yend = yend >= obin_size_y ? obin_size_y - 1 : yend;
        zend = floor(z_rescaled + ns / 2.0) - zoffset;
        zend = zend >= obin_size_z ? obin_size_z - 1 : zend;

        for (int zz = zstart; zz <= zend; zz++) {
            T disz = abs(z_rescaled - (zz + zoffset));
            T kervalue3 = evaluate_kernel(disz, es_c, es_beta, ns);
            for (int yy = ystart; yy <= yend; yy++) {
                T disy = abs(y_rescaled - (yy + yoffset));
                T kervalue2 = evaluate_kernel(disy, es_c, es_beta, ns);
                for (int xx = xstart; xx <= xend; xx++) {
                    outidx = xx + yy * obin_size_x + zz * obin_size_y * obin_size_x;
                    T disx = abs(x_rescaled - (xx + xoffset));
                    T kervalue1 = evaluate_kernel(disx, es_c, es_beta, ns);
                    atomicAdd(&fwshared[outidx].x, cnow.x * kervalue1 * kervalue2 * kervalue3);
                    atomicAdd(&fwshared[outidx].y, cnow.y * kervalue1 * kervalue2 * kervalue3);
                }
            }
        }
    }
    __syncthreads();
    /* write to global memory */
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        int i = n % obin_size_x;
        int j = (n / obin_size_x) % obin_size_y;
        int k = n / (obin_size_x * obin_size_y);

        ix = xoffset + i;
        iy = yoffset + j;
        iz = zoffset + k;
        outidx = ix + iy * nf1 + iz * nf1 * nf2;
        atomicAdd(&fw[outidx].x, fwshared[n].x);
        atomicAdd(&fw[outidx].y, fwshared[n].y);
    }
}

template <typename T>
__global__ void spread_3d_block_gather_horner(const T *x, const T *y, const T *z, const cuda_complex<T> *c,
                                              cuda_complex<T> *fw, int M, int ns, int nf1, int nf2, int nf3, T es_c,
                                              T es_beta, T sigma, const int *binstartpts, int obin_size_x,
                                              int obin_size_y, int obin_size_z, int binsperobin, int *subprob_to_bin,
                                              const int *subprobstartpts, int maxsubprobsize, int nobinx, int nobiny,
                                              int nobinz, const int *idxnupts, int pirange) {
    extern __shared__ char sharedbuf[];
    cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;

    int xstart, ystart, zstart, xend, yend, zend;
    int xstartnew, ystartnew, zstartnew, xendnew, yendnew, zendnew;
    int subpidx = blockIdx.x;
    int obidx = subprob_to_bin[subpidx];
    int bidx = obidx * binsperobin;

    int obinsubp_idx = subpidx - subprobstartpts[obidx];
    int ix, iy, iz;
    int outidx;
    int ptstart = binstartpts[bidx] + obinsubp_idx * maxsubprobsize;
    int nupts =
        min(maxsubprobsize, binstartpts[bidx + binsperobin] - binstartpts[bidx] - obinsubp_idx * maxsubprobsize);

    int xoffset = (obidx % nobinx) * obin_size_x;
    int yoffset = (obidx / nobinx) % nobiny * obin_size_y;
    int zoffset = (obidx / (nobinx * nobiny)) * obin_size_z;

    int N = obin_size_x * obin_size_y * obin_size_z;

    T ker1[MAX_NSPREAD];
    T ker2[MAX_NSPREAD];
    T ker3[MAX_NSPREAD];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        fwshared[i].x = 0.0;
        fwshared[i].y = 0.0;
    }
    __syncthreads();

    T x_rescaled, y_rescaled, z_rescaled;
    cuda_complex<T> cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int nidx = idxnupts[ptstart + i];
        int b = nidx / M;
        int box[3];
        for (int d = 0; d < 3; d++) {
            box[d] = b % 3;
            if (box[d] == 1)
                box[d] = -1;
            if (box[d] == 2)
                box[d] = 1;
            b = b / 3;
        }
        int ii = nidx % M;
        x_rescaled = RESCALE(x[ii], nf1, pirange) + box[0] * nf1;
        y_rescaled = RESCALE(y[ii], nf2, pirange) + box[1] * nf2;
        z_rescaled = RESCALE(z[ii], nf3, pirange) + box[2] * nf3;
        cnow = c[ii];

        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        zstart = ceil(z_rescaled - ns / 2.0) - zoffset;
        xend = floor(x_rescaled + ns / 2.0) - xoffset;
        yend = floor(y_rescaled + ns / 2.0) - yoffset;
        zend = floor(z_rescaled + ns / 2.0) - zoffset;

        eval_kernel_vec_horner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
        eval_kernel_vec_horner(ker2, ystart + yoffset - y_rescaled, ns, sigma);
        eval_kernel_vec_horner(ker3, zstart + zoffset - z_rescaled, ns, sigma);

        xstartnew = xstart < 0 ? 0 : xstart;
        ystartnew = ystart < 0 ? 0 : ystart;
        zstartnew = zstart < 0 ? 0 : zstart;
        xendnew = xend >= obin_size_x ? obin_size_x - 1 : xend;
        yendnew = yend >= obin_size_y ? obin_size_y - 1 : yend;
        zendnew = zend >= obin_size_z ? obin_size_z - 1 : zend;

        for (int zz = zstartnew; zz <= zendnew; zz++) {
            T kervalue3 = ker3[zz - zstart];
            for (int yy = ystartnew; yy <= yendnew; yy++) {
                T kervalue2 = ker2[yy - ystart];
                for (int xx = xstartnew; xx <= xendnew; xx++) {
                    outidx = xx + yy * obin_size_x + zz * obin_size_y * obin_size_x;
                    T kervalue1 = ker1[xx - xstart];
                    atomicAdd(&fwshared[outidx].x, cnow.x * kervalue1 * kervalue2 * kervalue3);
                    atomicAdd(&fwshared[outidx].y, cnow.y * kervalue1 * kervalue2 * kervalue3);
                }
            }
        }
    }
    __syncthreads();
    /* write to global memory */
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        int i = n % obin_size_x;
        int j = (n / obin_size_x) % obin_size_y;
        int k = n / (obin_size_x * obin_size_y);

        ix = xoffset + i;
        iy = yoffset + j;
        iz = zoffset + k;
        outidx = ix + iy * nf1 + iz * nf1 * nf2;
        atomicAdd(&fw[outidx].x, fwshared[n].x);
        atomicAdd(&fw[outidx].y, fwshared[n].y);
    }
}

/* ---------------------- 3d Interpolation Kernels ---------------------------*/
/* Kernels for NUptsdriven Method */
template <typename T>
__global__ void interp_3d_nupts_driven(const T *x, const T *y, const T *z, cuda_complex<T> *c,
                                       const cuda_complex<T> *fw, int M, int ns, int nf1, int nf2, int nf3, T es_c,
                                       T es_beta, const int *idxnupts, int pirange) {
    T ker1[MAX_NSPREAD];
    T ker2[MAX_NSPREAD];
    T ker3[MAX_NSPREAD];

    T ker1val, ker2val, ker3val;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        T x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
        T y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
        T z_rescaled = RESCALE(z[idxnupts[i]], nf3, pirange);
        int xstart = ceil(x_rescaled - ns / 2.0);
        int ystart = ceil(y_rescaled - ns / 2.0);
        int zstart = ceil(z_rescaled - ns / 2.0);
        int xend = floor(x_rescaled + ns / 2.0);
        int yend = floor(y_rescaled + ns / 2.0);
        int zend = floor(z_rescaled + ns / 2.0);
        cuda_complex<T> cnow;
        cnow.x = 0.0;
        cnow.y = 0.0;
        T x1 = (T)xstart - x_rescaled;
        T y1 = (T)ystart - y_rescaled;
        T z1 = (T)zstart - z_rescaled;

        eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
        eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
        eval_kernel_vec(ker3, z1, ns, es_c, es_beta);
        for (int zz = zstart; zz <= zend; zz++) {
            ker3val = ker3[zz - zstart];
            for (int yy = ystart; yy <= yend; yy++) {
                ker2val = ker2[yy - ystart];
                for (int xx = xstart; xx <= xend; xx++) {
                    int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
                    int iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
                    int iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);

                    int inidx = ix + iy * nf1 + iz * nf2 * nf1;

                    ker1val = ker1[xx - xstart];
                    cnow.x += fw[inidx].x * ker1val * ker2val * ker3val;
                    cnow.y += fw[inidx].y * ker1val * ker2val * ker3val;
                }
            }
        }
        c[idxnupts[i]].x = cnow.x;
        c[idxnupts[i]].y = cnow.y;
    }
}

template <typename T>
__global__ void interp_3d_nupts_driven_horner(const T *x, const T *y, const T *z, cuda_complex<T> *c,
                                              const cuda_complex<T> *fw, int M, int ns, int nf1, int nf2, int nf3,
                                              T sigma, int *idxnupts, int pirange) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        T x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
        T y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
        T z_rescaled = RESCALE(z[idxnupts[i]], nf3, pirange);

        int xstart = ceil(x_rescaled - ns / 2.0);
        int ystart = ceil(y_rescaled - ns / 2.0);
        int zstart = ceil(z_rescaled - ns / 2.0);

        int xend = floor(x_rescaled + ns / 2.0);
        int yend = floor(y_rescaled + ns / 2.0);
        int zend = floor(z_rescaled + ns / 2.0);

        cuda_complex<T> cnow;
        cnow.x = 0.0;
        cnow.y = 0.0;

        T ker1[MAX_NSPREAD];
        T ker2[MAX_NSPREAD];
        T ker3[MAX_NSPREAD];

        eval_kernel_vec_horner(ker1, xstart - x_rescaled, ns, sigma);
        eval_kernel_vec_horner(ker2, ystart - y_rescaled, ns, sigma);
        eval_kernel_vec_horner(ker3, zstart - z_rescaled, ns, sigma);

        for (int zz = zstart; zz <= zend; zz++) {
            T kervalue3 = ker3[zz - zstart];
            int iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
            for (int yy = ystart; yy <= yend; yy++) {
                T kervalue2 = ker2[yy - ystart];
                int iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
                for (int xx = xstart; xx <= xend; xx++) {
                    int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
                    int inidx = ix + iy * nf1 + iz * nf2 * nf1;
                    T kervalue1 = ker1[xx - xstart];
                    cnow.x += fw[inidx].x * kervalue1 * kervalue2 * kervalue3;
                    cnow.y += fw[inidx].y * kervalue1 * kervalue2 * kervalue3;
                }
            }
        }
        c[idxnupts[i]].x = cnow.x;
        c[idxnupts[i]].y = cnow.y;
    }
}

/* Kernels for SubProb Method */
template <typename T>
__global__ void interp_3d_subprob(const T *x, const T *y, const T *z, cuda_complex<T> *c, const cuda_complex<T> *fw,
                                  int M, int ns, int nf1, int nf2, int nf3, T es_c, T es_beta, const int *binstartpts,
                                  const int *bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
                                  const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
                                  int maxsubprobsize, int nbinx, int nbiny, int nbinz, const int *idxnupts,
                                  int pirange) {
    extern __shared__ char sharedbuf[];
    cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;

    int xstart, ystart, xend, yend, zstart, zend;
    int subpidx = blockIdx.x;
    int bidx = subprob_to_bin[subpidx];
    int binsubp_idx = subpidx - subprobstartpts[bidx];
    int ix, iy, iz;
    int outidx;
    int ptstart = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
    int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

    int xoffset = (bidx % nbinx) * bin_size_x;
    int yoffset = ((bidx / nbinx) % nbiny) * bin_size_y;
    int zoffset = (bidx / (nbinx * nbiny)) * bin_size_z;

    int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) * (bin_size_z + 2 * ceil(ns / 2.0));

#if 1
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        int i = n % (int)(bin_size_x + 2 * ceil(ns / 2.0));
        int j = (int)(n / (bin_size_x + 2 * ceil(ns / 2.0))) % (int)(bin_size_y + 2 * ceil(ns / 2.0));
        int k = n / ((bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)));

        ix = xoffset - ceil(ns / 2.0) + i;
        iy = yoffset - ceil(ns / 2.0) + j;
        iz = zoffset - ceil(ns / 2.0) + k;
        if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0)) && iz < (nf3 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
            iz = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
            outidx = ix + iy * nf1 + iz * nf1 * nf2;
            int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2) +
                            k * (bin_size_x + ceil(ns / 2.0) * 2) * (bin_size_y + ceil(ns / 2.0) * 2);
            fwshared[sharedidx].x = fw[outidx].x;
            fwshared[sharedidx].y = fw[outidx].y;
        }
    }
#endif
    __syncthreads();
    T ker1[MAX_NSPREAD];
    T ker2[MAX_NSPREAD];
    T ker3[MAX_NSPREAD];

    T x_rescaled, y_rescaled, z_rescaled;
    cuda_complex<T> cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int idx = ptstart + i;
        x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
        z_rescaled = RESCALE(z[idxnupts[idx]], nf3, pirange);
        cnow.x = 0.0;
        cnow.y = 0.0;
        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        zstart = ceil(z_rescaled - ns / 2.0) - zoffset;

        xend = floor(x_rescaled + ns / 2.0) - xoffset;
        yend = floor(y_rescaled + ns / 2.0) - yoffset;
        zend = floor(z_rescaled + ns / 2.0) - zoffset;

        T x1 = (T)xstart + xoffset - x_rescaled;
        T y1 = (T)ystart + yoffset - y_rescaled;
        T z1 = (T)zstart + zoffset - z_rescaled;

        eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
        eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
        eval_kernel_vec(ker3, z1, ns, es_c, es_beta);

        for (int zz = zstart; zz <= zend; zz++) {
            T kervalue3 = ker3[zz - zstart];
            iz = zz + ceil(ns / 2.0);
            for (int yy = ystart; yy <= yend; yy++) {
                T kervalue2 = ker2[yy - ystart];
                iy = yy + ceil(ns / 2.0);
                for (int xx = xstart; xx <= xend; xx++) {
                    ix = xx + ceil(ns / 2.0);
                    outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2) +
                             iz * (bin_size_x + ceil(ns / 2.0) * 2) * (bin_size_y + ceil(ns / 2.0) * 2);

                    T kervalue1 = ker1[xx - xstart];
                    cnow.x += fwshared[outidx].x * kervalue1 * kervalue2 * kervalue3;
                    cnow.y += fwshared[outidx].y * kervalue1 * kervalue2 * kervalue3;
                }
            }
        }
        c[idxnupts[idx]].x = cnow.x;
        c[idxnupts[idx]].y = cnow.y;
    }
}

template <typename T>
__global__ void interp_3d_subprob_horner(const T *x, const T *y, const T *z, cuda_complex<T> *c,
                                         const cuda_complex<T> *fw, int M, int ns, int nf1, int nf2, int nf3, T sigma,
                                         const int *binstartpts, const int *bin_size, int bin_size_x, int bin_size_y,
                                         int bin_size_z, const int *subprob_to_bin, const int *subprobstartpts,
                                         const int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, int nbinz,
                                         const int *idxnupts, int pirange) {
    extern __shared__ char sharedbuf[];
    cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;

    int xstart, ystart, xend, yend, zstart, zend;
    int subpidx = blockIdx.x;
    int bidx = subprob_to_bin[subpidx];
    int binsubp_idx = subpidx - subprobstartpts[bidx];
    int ix, iy, iz;
    int outidx;
    int ptstart = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
    int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

    int xoffset = (bidx % nbinx) * bin_size_x;
    int yoffset = ((bidx / nbinx) % nbiny) * bin_size_y;
    int zoffset = (bidx / (nbinx * nbiny)) * bin_size_z;

    int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)) * (bin_size_z + 2 * ceil(ns / 2.0));

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        int i = n % (int)(bin_size_x + 2 * ceil(ns / 2.0));
        int j = (int)(n / (bin_size_x + 2 * ceil(ns / 2.0))) % (int)(bin_size_y + 2 * ceil(ns / 2.0));
        int k = n / ((bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0)));

        ix = xoffset - ceil(ns / 2.0) + i;
        iy = yoffset - ceil(ns / 2.0) + j;
        iz = zoffset - ceil(ns / 2.0) + k;
        if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0)) && iz < (nf3 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
            iz = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
            outidx = ix + iy * nf1 + iz * nf1 * nf2;
            int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2) +
                            k * (bin_size_x + ceil(ns / 2.0) * 2) * (bin_size_y + ceil(ns / 2.0) * 2);
            fwshared[sharedidx].x = fw[outidx].x;
            fwshared[sharedidx].y = fw[outidx].y;
        }
    }
    __syncthreads();
    T ker1[MAX_NSPREAD];
    T ker2[MAX_NSPREAD];
    T ker3[MAX_NSPREAD];
    T x_rescaled, y_rescaled, z_rescaled;
    cuda_complex<T> cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int idx = ptstart + i;
        x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
        z_rescaled = RESCALE(z[idxnupts[idx]], nf3, pirange);
        cnow.x = 0.0;
        cnow.y = 0.0;

        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        zstart = ceil(z_rescaled - ns / 2.0) - zoffset;

        xend = floor(x_rescaled + ns / 2.0) - xoffset;
        yend = floor(y_rescaled + ns / 2.0) - yoffset;
        zend = floor(z_rescaled + ns / 2.0) - zoffset;

        eval_kernel_vec_horner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
        eval_kernel_vec_horner(ker2, ystart + yoffset - y_rescaled, ns, sigma);
        eval_kernel_vec_horner(ker3, zstart + zoffset - z_rescaled, ns, sigma);
        for (int zz = zstart; zz <= zend; zz++) {
            T kervalue3 = ker3[zz - zstart];
            iz = zz + ceil(ns / 2.0);
            for (int yy = ystart; yy <= yend; yy++) {
                T kervalue2 = ker2[yy - ystart];
                iy = yy + ceil(ns / 2.0);
                for (int xx = xstart; xx <= xend; xx++) {
                    ix = xx + ceil(ns / 2.0);
                    outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2) +
                             iz * (bin_size_x + ceil(ns / 2.0) * 2) * (bin_size_y + ceil(ns / 2.0) * 2);
                    T kervalue1 = ker1[xx - xstart];
                    cnow.x += fwshared[outidx].x * kervalue1 * kervalue2 * kervalue3;
                    cnow.y += fwshared[outidx].y * kervalue1 * kervalue2 * kervalue3;
                }
            }
        }
        c[idxnupts[idx]].x = cnow.x;
        c[idxnupts[idx]].y = cnow.y;
    }
}

} // namespace spreadinterp
} // namespace cufinufft
