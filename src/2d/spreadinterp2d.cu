#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>
#include <math.h>
#include <thrust/extrema.h>

#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

/* ------------------------ 2d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

__global__ void Spread_2d_NUptsdriven(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                                      int nf1, int nf2, CUFINUFFT_FLT es_c, CUFINUFFT_FLT es_beta, int *idxnupts,
                                      int pirange) {
    int xstart, ystart, xend, yend;
    int xx, yy, ix, iy;
    int outidx;
    CUFINUFFT_FLT ker1[MAX_NSPREAD];
    CUFINUFFT_FLT ker2[MAX_NSPREAD];

    CUFINUFFT_FLT x_rescaled, y_rescaled;
    CUFINUFFT_FLT kervalue1, kervalue2;
    CUCPX cnow;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
        cnow = c[idxnupts[i]];

        xstart = ceil(x_rescaled - ns / 2.0);
        ystart = ceil(y_rescaled - ns / 2.0);
        xend = floor(x_rescaled + ns / 2.0);
        yend = floor(y_rescaled + ns / 2.0);

        CUFINUFFT_FLT x1 = (CUFINUFFT_FLT)xstart - x_rescaled;
        CUFINUFFT_FLT y1 = (CUFINUFFT_FLT)ystart - y_rescaled;
        eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
        eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
        for (yy = ystart; yy <= yend; yy++) {
            for (xx = xstart; xx <= xend; xx++) {
                ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
                iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
                outidx = ix + iy * nf1;
                kervalue1 = ker1[xx - xstart];
                kervalue2 = ker2[yy - ystart];
                atomicAdd(&fw[outidx].x, cnow.x * kervalue1 * kervalue2);
                atomicAdd(&fw[outidx].y, cnow.y * kervalue1 * kervalue2);
            }
        }
    }
}

__global__ void Spread_2d_NUptsdriven_Horner(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M,
                                             const int ns, int nf1, int nf2, CUFINUFFT_FLT sigma, int *idxnupts,
                                             int pirange) {
    int xx, yy, ix, iy;
    int outidx;
    CUFINUFFT_FLT ker1[MAX_NSPREAD];
    CUFINUFFT_FLT ker2[MAX_NSPREAD];
    CUFINUFFT_FLT ker1val, ker2val;

    CUFINUFFT_FLT x_rescaled, y_rescaled;
    CUCPX cnow;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);
        cnow = c[idxnupts[i]];
        int xstart = ceil(x_rescaled - ns / 2.0);
        int ystart = ceil(y_rescaled - ns / 2.0);
        int xend = floor(x_rescaled + ns / 2.0);
        int yend = floor(y_rescaled + ns / 2.0);

        CUFINUFFT_FLT x1 = (CUFINUFFT_FLT)xstart - x_rescaled;
        CUFINUFFT_FLT y1 = (CUFINUFFT_FLT)ystart - y_rescaled;
        eval_kernel_vec_Horner(ker1, x1, ns, sigma);
        eval_kernel_vec_Horner(ker2, y1, ns, sigma);
        for (yy = ystart; yy <= yend; yy++) {
            for (xx = xstart; xx <= xend; xx++) {
                ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
                iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
                outidx = ix + iy * nf1;
                ker1val = ker1[xx - xstart];
                ker2val = ker2[yy - ystart];
                CUFINUFFT_FLT kervalue = ker1val * ker2val;
                atomicAdd(&fw[outidx].x, cnow.x * kervalue);
                atomicAdd(&fw[outidx].y, cnow.y * kervalue);
            }
        }
    }
}

/* Kernels for SubProb Method */
// SubProb properties
__global__ void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                                       int *bin_size, CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, int *sortidx, int pirange) {
    int binidx, binx, biny;
    int oldidx;
    CUFINUFFT_FLT x_rescaled, y_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        x_rescaled = RESCALE(x[i], nf1, pirange);
        y_rescaled = RESCALE(y[i], nf2, pirange);
        binx = floor(x_rescaled / bin_size_x);
        binx = binx >= nbinx ? binx - 1 : binx;
        binx = binx < 0 ? 0 : binx;
        biny = floor(y_rescaled / bin_size_y);
        biny = biny >= nbiny ? biny - 1 : biny;
        biny = biny < 0 ? 0 : biny;
        binidx = binx + biny * nbinx;
        oldidx = atomicAdd(&bin_size[binidx], 1);
        sortidx[i] = oldidx;
        if (binx >= nbinx || biny >= nbiny) {
            sortidx[i] = -biny;
        }
    }
}

__global__ void CalcInvertofGlobalSortIdx_2d(int M, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                                             int *bin_startpts, int *sortidx, CUFINUFFT_FLT *x, CUFINUFFT_FLT *y,
                                             int *index, int pirange, int nf1, int nf2) {
    int binx, biny;
    int binidx;
    CUFINUFFT_FLT x_rescaled, y_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        x_rescaled = RESCALE(x[i], nf1, pirange);
        y_rescaled = RESCALE(y[i], nf2, pirange);
        binx = floor(x_rescaled / bin_size_x);
        binx = binx >= nbinx ? binx - 1 : binx;
        binx = binx < 0 ? 0 : binx;
        biny = floor(y_rescaled / bin_size_y);
        biny = biny >= nbiny ? biny - 1 : biny;
        biny = biny < 0 ? 0 : biny;
        binidx = binx + biny * nbinx;

        index[bin_startpts[binidx] + sortidx[i]] = i;
    }
}

__global__ void Spread_2d_Subprob(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns, int nf1,
                                  int nf2, CUFINUFFT_FLT es_c, CUFINUFFT_FLT es_beta, CUFINUFFT_FLT sigma,
                                  int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y, int *subprob_to_bin,
                                  int *subprobstartpts, int *numsubprob, int maxsubprobsize, int nbinx, int nbiny,
                                  int *idxnupts, int pirange) {
    extern __shared__ CUCPX fwshared[];

    int xstart, ystart, xend, yend;
    int subpidx = blockIdx.x;
    int bidx = subprob_to_bin[subpidx];
    int binsubp_idx = subpidx - subprobstartpts[bidx];
    int ix, iy;
    int outidx;
    int ptstart = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
    int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

    int xoffset = (bidx % nbinx) * bin_size_x;
    int yoffset = (bidx / nbinx) * bin_size_y;

    int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));
    CUFINUFFT_FLT ker1[MAX_NSPREAD];
    CUFINUFFT_FLT ker2[MAX_NSPREAD];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        fwshared[i].x = 0.0;
        fwshared[i].y = 0.0;
    }
    __syncthreads();

    CUFINUFFT_FLT x_rescaled, y_rescaled;
    CUCPX cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int idx = ptstart + i;
        x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
        cnow = c[idxnupts[idx]];

        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        xend = floor(x_rescaled + ns / 2.0) - xoffset;
        yend = floor(y_rescaled + ns / 2.0) - yoffset;

        CUFINUFFT_FLT x1 = (CUFINUFFT_FLT)xstart + xoffset - x_rescaled;
        CUFINUFFT_FLT y1 = (CUFINUFFT_FLT)ystart + yoffset - y_rescaled;
        eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
        eval_kernel_vec(ker2, y1, ns, es_c, es_beta);

        for (int yy = ystart; yy <= yend; yy++) {
            iy = yy + ceil(ns / 2.0);
            if (iy >= (bin_size_y + (int)ceil(ns / 2.0) * 2) || iy < 0)
                break;
            for (int xx = xstart; xx <= xend; xx++) {
                ix = xx + ceil(ns / 2.0);
                if (ix >= (bin_size_x + (int)ceil(ns / 2.0) * 2) || ix < 0)
                    break;
                outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2);
                CUFINUFFT_FLT kervalue1 = ker1[xx - xstart];
                CUFINUFFT_FLT kervalue2 = ker2[yy - ystart];
                atomicAdd(&fwshared[outidx].x, cnow.x * kervalue1 * kervalue2);
                atomicAdd(&fwshared[outidx].y, cnow.y * kervalue1 * kervalue2);
            }
        }
    }
    __syncthreads();
    /* write to global memory */
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        int i = k % (int)(bin_size_x + 2 * ceil(ns / 2.0));
        int j = k / (bin_size_x + 2 * ceil(ns / 2.0));
        ix = xoffset - ceil(ns / 2.0) + i;
        iy = yoffset - ceil(ns / 2.0) + j;
        if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
            outidx = ix + iy * nf1;
            int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
            atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
            atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
        }
    }
}

__global__ void Spread_2d_Subprob_Horner(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                                         int nf1, int nf2, CUFINUFFT_FLT sigma, int *binstartpts, int *bin_size,
                                         int bin_size_x, int bin_size_y, int *subprob_to_bin, int *subprobstartpts,
                                         int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, int *idxnupts,
                                         int pirange) {
    extern __shared__ CUCPX fwshared[];

    int xstart, ystart, xend, yend;
    int subpidx = blockIdx.x;
    int bidx = subprob_to_bin[subpidx];
    int binsubp_idx = subpidx - subprobstartpts[bidx];
    int ix, iy, outidx;
    int ptstart = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
    int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

    int xoffset = (bidx % nbinx) * bin_size_x;
    int yoffset = (bidx / nbinx) * bin_size_y;

    int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));

    CUFINUFFT_FLT ker1[MAX_NSPREAD];
    CUFINUFFT_FLT ker2[MAX_NSPREAD];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        fwshared[i].x = 0.0;
        fwshared[i].y = 0.0;
    }
    __syncthreads();

    CUFINUFFT_FLT x_rescaled, y_rescaled;
    CUCPX cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int idx = ptstart + i;
        x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
        cnow = c[idxnupts[idx]];

        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        xend = floor(x_rescaled + ns / 2.0) - xoffset;
        yend = floor(y_rescaled + ns / 2.0) - yoffset;

        eval_kernel_vec_Horner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
        eval_kernel_vec_Horner(ker2, ystart + yoffset - y_rescaled, ns, sigma);

        for (int yy = ystart; yy <= yend; yy++) {
            iy = yy + ceil(ns / 2.0);
            if (iy >= (bin_size_y + (int)ceil(ns / 2.0) * 2) || iy < 0)
                break;
            CUFINUFFT_FLT kervalue2 = ker2[yy - ystart];
            for (int xx = xstart; xx <= xend; xx++) {
                ix = xx + ceil(ns / 2.0);
                if (ix >= (bin_size_x + (int)ceil(ns / 2.0) * 2) || ix < 0)
                    break;
                outidx = ix + iy * (bin_size_x + (int)ceil(ns / 2.0) * 2);
                CUFINUFFT_FLT kervalue1 = ker1[xx - xstart];
                atomicAdd(&fwshared[outidx].x, cnow.x * kervalue1 * kervalue2);
                atomicAdd(&fwshared[outidx].y, cnow.y * kervalue1 * kervalue2);
            }
        }
    }
    __syncthreads();

    /* write to global memory */
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        int i = k % (int)(bin_size_x + 2 * ceil(ns / 2.0));
        int j = k / (bin_size_x + 2 * ceil(ns / 2.0));
        ix = xoffset - ceil(ns / 2.0) + i;
        iy = yoffset - ceil(ns / 2.0) + j;
        if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
            outidx = ix + iy * nf1;
            int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
            atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
            atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
        }
    }
}

/* Kernels for Paul's Method */
__global__ void LocateFineGridPos_Paul(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                                       int *bin_size, int ns, CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, int *sortidx,
                                       int *finegridsize, int pirange) {
    int binidx, binx, biny;
    int oldidx;
    int xidx, yidx, finegrididx;
    CUFINUFFT_FLT x_rescaled, y_rescaled;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        if (ns % 2 == 0) {
            x_rescaled = RESCALE(x[i], nf1, pirange);
            y_rescaled = RESCALE(y[i], nf2, pirange);
            binx = floor(floor(x_rescaled) / bin_size_x);
            biny = floor(floor(y_rescaled) / bin_size_y);
            binidx = binx + biny * nbinx;
            xidx = floor(x_rescaled) - binx * bin_size_x;
            yidx = floor(y_rescaled) - biny * bin_size_y;
            finegrididx = binidx * bin_size_x * bin_size_y + xidx + yidx * bin_size_x;
        } else {
            x_rescaled = RESCALE(x[i], nf1, pirange);
            y_rescaled = RESCALE(y[i], nf2, pirange);
            xidx = ceil(x_rescaled - 0.5);
            yidx = ceil(y_rescaled - 0.5);

            // xidx = (xidx == nf1) ? (xidx-nf1) : xidx;
            // yidx = (yidx == nf2) ? (yidx-nf2) : yidx;

            binx = floor(xidx / (float)bin_size_x);
            biny = floor(yidx / (float)bin_size_y);
            binidx = binx + biny * nbinx;

            xidx = xidx - binx * bin_size_x;
            yidx = yidx - biny * bin_size_y;
            finegrididx = binidx * bin_size_x * bin_size_y + xidx + yidx * bin_size_x;
        }
        oldidx = atomicAdd(&finegridsize[finegrididx], 1);
        sortidx[i] = oldidx;
    }
}

__global__ void CalcInvertofGlobalSortIdx_Paul(int nf1, int nf2, int M, int bin_size_x, int bin_size_y, int nbinx,
                                               int nbiny, int ns, CUFINUFFT_FLT *x, CUFINUFFT_FLT *y,
                                               int *finegridstartpts, int *sortidx, int *index, int pirange) {
    CUFINUFFT_FLT x_rescaled, y_rescaled;
    int binx, biny, binidx, xidx, yidx, finegrididx;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i += gridDim.x * blockDim.x) {
        if (ns % 2 == 0) {
            x_rescaled = RESCALE(x[i], nf1, pirange);
            y_rescaled = RESCALE(y[i], nf2, pirange);
            binx = floor(floor(x_rescaled) / bin_size_x);
            biny = floor(floor(y_rescaled) / bin_size_y);
            binidx = binx + biny * nbinx;
            xidx = floor(x_rescaled) - binx * bin_size_x;
            yidx = floor(y_rescaled) - biny * bin_size_y;
            finegrididx = binidx * bin_size_x * bin_size_y + xidx + yidx * bin_size_x;
        } else {
            x_rescaled = RESCALE(x[i], nf1, pirange);
            y_rescaled = RESCALE(y[i], nf2, pirange);
            xidx = ceil(x_rescaled - 0.5);
            yidx = ceil(y_rescaled - 0.5);

            xidx = (xidx == nf1) ? xidx - nf1 : xidx;
            yidx = (yidx == nf2) ? yidx - nf2 : yidx;

            binx = floor(xidx / (float)bin_size_x);
            biny = floor(yidx / (float)bin_size_y);
            binidx = binx + biny * nbinx;

            xidx = xidx - binx * bin_size_x;
            yidx = yidx - biny * bin_size_y;
            finegrididx = binidx * bin_size_x * bin_size_y + xidx + yidx * bin_size_x;
        }
        index[finegridstartpts[finegrididx] + sortidx[i]] = i;
    }
}

__global__ void Spread_2d_Subprob_Paul(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                                       int nf1, int nf2, CUFINUFFT_FLT es_c, CUFINUFFT_FLT es_beta, CUFINUFFT_FLT sigma,
                                       int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y,
                                       int *subprob_to_bin, int *subprobstartpts, int *numsubprob, int maxsubprobsize,
                                       int nbinx, int nbiny, int *idxnupts, int *fgstartpts, int *finegridsize,
                                       int pirange) {
    extern __shared__ CUCPX fwshared[];

    int xstart, ystart, xend, yend;
    int subpidx = blockIdx.x;
    int bidx = subprob_to_bin[subpidx];
    int binsubp_idx = subpidx - subprobstartpts[bidx];

    int ix, iy, outidx;

    int xoffset = (bidx % nbinx) * bin_size_x;
    int yoffset = (bidx / nbinx) * bin_size_y;

    int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));
#if 0
	CUFINUFFT_FLT ker1[MAX_NSPREAD*10];
    CUFINUFFT_FLT ker2[MAX_NSPREAD*10];
#endif
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        fwshared[i].x = 0.0;
        fwshared[i].y = 0.0;
    }
    __syncthreads();

    CUFINUFFT_FLT x_rescaled, y_rescaled;
    for (int i = threadIdx.x; i < bin_size_x * bin_size_y; i += blockDim.x) {
        int fineidx = bidx * bin_size_x * bin_size_y + i;
        int idxstart = fgstartpts[fineidx] + binsubp_idx * maxsubprobsize;
        int nupts = min(maxsubprobsize, finegridsize[fineidx] - binsubp_idx * maxsubprobsize);
        if (nupts > 0) {
            x_rescaled = x[idxnupts[idxstart]];
            y_rescaled = y[idxnupts[idxstart]];

            xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
            ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
            xend = floor(x_rescaled + ns / 2.0) - xoffset;
            yend = floor(y_rescaled + ns / 2.0) - yoffset;
#if 0
			for(int m=0; m<nupts; m++){
				int idx = idxstart+m;
				x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
				y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);

				eval_kernel_vec_Horner(ker1+m*MAX_NSPREAD,xstart+xoffset-
					x_rescaled,ns,sigma);
				eval_kernel_vec_Horner(ker2+m*MAX_NSPREAD,ystart+yoffset-
					y_rescaled,ns,sigma);
			}
#endif
            for (int yy = ystart; yy <= yend; yy++) {
                CUFINUFFT_FLT kervalue2[10];
                for (int m = 0; m < nupts; m++) {
                    int idx = idxstart + m;
#if 1
                    y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
                    CUFINUFFT_FLT disy = abs(y_rescaled - (yy + yoffset));
                    kervalue2[m] = evaluate_kernel(disy, es_c, es_beta, ns);
#else
                    kervalue2[m] = ker2[m * MAX_NSPREAD + yy - ystart];
#endif
                }
                for (int xx = xstart; xx <= xend; xx++) {
                    ix = xx + ceil(ns / 2.0);
                    iy = yy + ceil(ns / 2.0);
                    outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2);
                    CUCPX updatevalue;
                    updatevalue.x = 0.0;
                    updatevalue.y = 0.0;
                    for (int m = 0; m < nupts; m++) {
                        int idx = idxstart + m;
#if 1
                        x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
                        CUFINUFFT_FLT disx = abs(x_rescaled - (xx + xoffset));
                        CUFINUFFT_FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta, ns);

                        updatevalue.x += kervalue2[m] * kervalue1 * c[idxnupts[idx]].x;
                        updatevalue.y += kervalue2[m] * kervalue1 * c[idxnupts[idx]].y;
#else
                        CUFINUFFT_FLT kervalue1 = ker1[m * MAX_NSPREAD + xx - xstart];
                        updatevalue.x += kervalue1 * kervalue2[m] * c[idxnupts[idx]].x;
                        updatevalue.y += kervalue1 * kervalue2[m] * c[idxnupts[idx]].y;
#endif
                    }
                    atomicAdd(&fwshared[outidx].x, updatevalue.x);
                    atomicAdd(&fwshared[outidx].y, updatevalue.y);
                }
            }
        }
    }
    __syncthreads();

    /* write to global memory */
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        int i = k % (int)(bin_size_x + 2 * ceil(ns / 2.0));
        int j = k / (bin_size_x + 2 * ceil(ns / 2.0));
        ix = xoffset - ceil(ns / 2.0) + i;
        iy = yoffset - ceil(ns / 2.0) + j;
        if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
            outidx = ix + iy * nf1;
            int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
            atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
            atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
        }
    }
}
/* --------------------- 2d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
__global__ void Interp_2d_NUptsdriven(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                                      int nf1, int nf2, CUFINUFFT_FLT es_c, CUFINUFFT_FLT es_beta, int *idxnupts,
                                      int pirange) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        CUFINUFFT_FLT x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
        CUFINUFFT_FLT y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);

        int xstart = ceil(x_rescaled - ns / 2.0);
        int ystart = ceil(y_rescaled - ns / 2.0);
        int xend = floor(x_rescaled + ns / 2.0);
        int yend = floor(y_rescaled + ns / 2.0);
        CUCPX cnow;
        cnow.x = 0.0;
        cnow.y = 0.0;
        CUFINUFFT_FLT ker1[MAX_NSPREAD];
        CUFINUFFT_FLT ker2[MAX_NSPREAD];

        CUFINUFFT_FLT x1 = (CUFINUFFT_FLT)xstart - x_rescaled;
        CUFINUFFT_FLT y1 = (CUFINUFFT_FLT)ystart - y_rescaled;
        eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
        eval_kernel_vec(ker2, y1, ns, es_c, es_beta);

        for (int yy = ystart; yy <= yend; yy++) {
            CUFINUFFT_FLT kervalue2 = ker2[yy - ystart];
            for (int xx = xstart; xx <= xend; xx++) {
                int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
                int iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
                int inidx = ix + iy * nf1;
                CUFINUFFT_FLT kervalue1 = ker1[xx - xstart];
                cnow.x += fw[inidx].x * kervalue1 * kervalue2;
                cnow.y += fw[inidx].y * kervalue1 * kervalue2;
            }
        }
        c[idxnupts[i]].x = cnow.x;
        c[idxnupts[i]].y = cnow.y;
    }
}

__global__ void Interp_2d_NUptsdriven_Horner(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M,
                                             const int ns, int nf1, int nf2, CUFINUFFT_FLT sigma, int *idxnupts,
                                             int pirange) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
        CUFINUFFT_FLT x_rescaled = RESCALE(x[idxnupts[i]], nf1, pirange);
        CUFINUFFT_FLT y_rescaled = RESCALE(y[idxnupts[i]], nf2, pirange);

        int xstart = ceil(x_rescaled - ns / 2.0);
        int ystart = ceil(y_rescaled - ns / 2.0);
        int xend = floor(x_rescaled + ns / 2.0);
        int yend = floor(y_rescaled + ns / 2.0);

        CUCPX cnow;
        cnow.x = 0.0;
        cnow.y = 0.0;
        CUFINUFFT_FLT ker1[MAX_NSPREAD];
        CUFINUFFT_FLT ker2[MAX_NSPREAD];

        eval_kernel_vec_Horner(ker1, xstart - x_rescaled, ns, sigma);
        eval_kernel_vec_Horner(ker2, ystart - y_rescaled, ns, sigma);

        for (int yy = ystart; yy <= yend; yy++) {
            CUFINUFFT_FLT kervalue2 = ker2[yy - ystart];
            for (int xx = xstart; xx <= xend; xx++) {
                int ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
                int iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
                int inidx = ix + iy * nf1;
                CUFINUFFT_FLT kervalue1 = ker1[xx - xstart];
                cnow.x += fw[inidx].x * kervalue1 * kervalue2;
                cnow.y += fw[inidx].y * kervalue1 * kervalue2;
            }
        }
        c[idxnupts[i]].x = cnow.x;
        c[idxnupts[i]].y = cnow.y;
    }
}

/* Kernels for Subprob Method */
__global__ void Interp_2d_Subprob(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns, int nf1,
                                  int nf2, CUFINUFFT_FLT es_c, CUFINUFFT_FLT es_beta, CUFINUFFT_FLT sigma,
                                  int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y, int *subprob_to_bin,
                                  int *subprobstartpts, int *numsubprob, int maxsubprobsize, int nbinx, int nbiny,
                                  int *idxnupts, int pirange) {
    extern __shared__ CUCPX fwshared[];

    int xstart, ystart, xend, yend;
    int subpidx = blockIdx.x;
    int bidx = subprob_to_bin[subpidx];
    int binsubp_idx = subpidx - subprobstartpts[bidx];
    int ix, iy;
    int outidx;
    int ptstart = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
    int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

    int xoffset = (bidx % nbinx) * bin_size_x;
    int yoffset = (bidx / nbinx) * bin_size_y;
    int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));

    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        int i = k % (int)(bin_size_x + 2 * ceil(ns / 2.0));
        int j = k / (bin_size_x + 2 * ceil(ns / 2.0));
        ix = xoffset - ceil(ns / 2.0) + i;
        iy = yoffset - ceil(ns / 2.0) + j;
        if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
            outidx = ix + iy * nf1;
            int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
            fwshared[sharedidx].x = fw[outidx].x;
            fwshared[sharedidx].y = fw[outidx].y;
        }
    }
    __syncthreads();

    CUFINUFFT_FLT ker1[MAX_NSPREAD];
    CUFINUFFT_FLT ker2[MAX_NSPREAD];

    CUFINUFFT_FLT x_rescaled, y_rescaled;
    CUCPX cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int idx = ptstart + i;
        x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
        cnow.x = 0.0;
        cnow.y = 0.0;

        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        xend = floor(x_rescaled + ns / 2.0) - xoffset;
        yend = floor(y_rescaled + ns / 2.0) - yoffset;

        CUFINUFFT_FLT x1 = (CUFINUFFT_FLT)xstart + xoffset - x_rescaled;
        CUFINUFFT_FLT y1 = (CUFINUFFT_FLT)ystart + yoffset - y_rescaled;

        eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
        eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
        for (int yy = ystart; yy <= yend; yy++) {
            CUFINUFFT_FLT kervalue2 = ker2[yy - ystart];
            for (int xx = xstart; xx <= xend; xx++) {
                ix = xx + ceil(ns / 2.0);
                iy = yy + ceil(ns / 2.0);
                outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2);
                CUFINUFFT_FLT kervalue1 = ker1[xx - xstart];
                cnow.x += fwshared[outidx].x * kervalue1 * kervalue2;
                cnow.y += fwshared[outidx].y * kervalue1 * kervalue2;
            }
        }
        c[idxnupts[idx]] = cnow;
    }
}

__global__ void Interp_2d_Subprob_Horner(CUFINUFFT_FLT *x, CUFINUFFT_FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                                         int nf1, int nf2, CUFINUFFT_FLT sigma, int *binstartpts, int *bin_size,
                                         int bin_size_x, int bin_size_y, int *subprob_to_bin, int *subprobstartpts,
                                         int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, int *idxnupts,
                                         int pirange) {
    extern __shared__ CUCPX fwshared[];

    int xstart, ystart, xend, yend;
    int subpidx = blockIdx.x;
    int bidx = subprob_to_bin[subpidx];
    int binsubp_idx = subpidx - subprobstartpts[bidx];
    int ix, iy;
    int outidx;
    int ptstart = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
    int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

    int xoffset = (bidx % nbinx) * bin_size_x;
    int yoffset = (bidx / nbinx) * bin_size_y;

    int N = (bin_size_x + 2 * ceil(ns / 2.0)) * (bin_size_y + 2 * ceil(ns / 2.0));

    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        int i = k % (int)(bin_size_x + 2 * ceil(ns / 2.0));
        int j = k / (bin_size_x + 2 * ceil(ns / 2.0));
        ix = xoffset - ceil(ns / 2.0) + i;
        iy = yoffset - ceil(ns / 2.0) + j;
        if (ix < (nf1 + ceil(ns / 2.0)) && iy < (nf2 + ceil(ns / 2.0))) {
            ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
            iy = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
            outidx = ix + iy * nf1;
            int sharedidx = i + j * (bin_size_x + ceil(ns / 2.0) * 2);
            fwshared[sharedidx].x = fw[outidx].x;
            fwshared[sharedidx].y = fw[outidx].y;
        }
    }
    __syncthreads();

    CUFINUFFT_FLT ker1[MAX_NSPREAD];
    CUFINUFFT_FLT ker2[MAX_NSPREAD];

    CUFINUFFT_FLT x_rescaled, y_rescaled;
    CUCPX cnow;
    for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
        int idx = ptstart + i;
        x_rescaled = RESCALE(x[idxnupts[idx]], nf1, pirange);
        y_rescaled = RESCALE(y[idxnupts[idx]], nf2, pirange);
        cnow.x = 0.0;
        cnow.y = 0.0;

        xstart = ceil(x_rescaled - ns / 2.0) - xoffset;
        ystart = ceil(y_rescaled - ns / 2.0) - yoffset;
        xend = floor(x_rescaled + ns / 2.0) - xoffset;
        yend = floor(y_rescaled + ns / 2.0) - yoffset;

        eval_kernel_vec_Horner(ker1, xstart + xoffset - x_rescaled, ns, sigma);
        eval_kernel_vec_Horner(ker2, ystart + yoffset - y_rescaled, ns, sigma);

        for (int yy = ystart; yy <= yend; yy++) {
            CUFINUFFT_FLT kervalue2 = ker2[yy - ystart];
            for (int xx = xstart; xx <= xend; xx++) {
                ix = xx + ceil(ns / 2.0);
                iy = yy + ceil(ns / 2.0);
                outidx = ix + iy * (bin_size_x + ceil(ns / 2.0) * 2);

                CUFINUFFT_FLT kervalue1 = ker1[xx - xstart];
                cnow.x += fwshared[outidx].x * kervalue1 * kervalue2;
                cnow.y += fwshared[outidx].y * kervalue1 * kervalue2;
            }
        }
        c[idxnupts[idx]] = cnow;
    }
}
