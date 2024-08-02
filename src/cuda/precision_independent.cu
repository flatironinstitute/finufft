/* These are functions that do not rely on CUFINUFFT_FLT.
   They are organized by originating file.
*/
// TODO: remove kernels that do not depend on dimension

#include <cuComplex.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>

#include <cufinufft/precision_independent.h>

namespace cufinufft {
namespace common {

/* Auxiliary func to compute power of complex number */
__device__ RT carg(const CT &z) { return (RT)atan2(ipart(z), rpart(z)); } // polar angle
__device__ RT cabs(const CT &z) { return (RT)cuCabs(z); }

/* Common Kernels from spreadinterp3d */
__host__ __device__ int calc_global_index(int xidx, int yidx, int zidx, int onx, int ony,
                                          int onz, int bnx, int bny, int bnz) {
  int oix, oiy, oiz;
  oix = xidx / bnx;
  oiy = yidx / bny;
  oiz = zidx / bnz;
  return (oix + oiy * onx + oiz * ony * onx) * (bnx * bny * bnz) +
         (xidx % bnx + yidx % bny * bnx + zidx % bnz * bny * bnx);
}

__device__ int calc_global_index_v2(int xidx, int yidx, int zidx, int nbinx, int nbiny,
                                    int nbinz) {
  return xidx + yidx * nbinx + zidx * nbinx * nbiny;
}

/* spreadinterp 1d */
__global__ void calc_subprob_1d(int *bin_size, int *num_subprob, int maxsubprobsize,
                                int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    num_subprob[i] = ceil(bin_size[i] / (float)maxsubprobsize);
  }
}

__global__ void map_b_into_subprob_1d(int *d_subprob_to_bin, int *d_subprobstartpts,
                                      int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < d_numsubprob[i]; j++) {
      d_subprob_to_bin[d_subprobstartpts[i] + j] = i;
    }
  }
}

/* spreadinterp 2d */
__global__ void calc_subprob_2d(int *bin_size, int *num_subprob, int maxsubprobsize,
                                int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    num_subprob[i] = ceil(bin_size[i] / (float)maxsubprobsize);
  }
}

__global__ void map_b_into_subprob_2d(int *d_subprob_to_bin, int *d_subprobstartpts,
                                      int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < d_numsubprob[i]; j++) {
      d_subprob_to_bin[d_subprobstartpts[i] + j] = i;
    }
  }
}

__global__ void trivial_global_sort_index_2d(int M, int *index) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    index[i] = i;
  }
}

/* spreadinterp3d */
__global__ void calc_subprob_3d_v2(int *bin_size, int *num_subprob, int maxsubprobsize,
                                   int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    num_subprob[i] = ceil(bin_size[i] / (float)maxsubprobsize);
  }
}

__global__ void map_b_into_subprob_3d_v2(int *d_subprob_to_bin, int *d_subprobstartpts,
                                         int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < d_numsubprob[i]; j++) {
      d_subprob_to_bin[d_subprobstartpts[i] + j] = i;
    }
  }
}

__global__ void calc_subprob_3d_v1(int binsperobinx, int binsperobiny, int binsperobinz,
                                   int *bin_size, int *num_subprob, int maxsubprobsize,
                                   int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    int numnupts    = 0;
    int binsperobin = binsperobinx * binsperobiny * binsperobinz;
    for (int b = 0; b < binsperobin; b++) {
      numnupts += bin_size[binsperobin * i + b];
    }
    num_subprob[i] = ceil(numnupts / (float)maxsubprobsize);
  }
}

__global__ void map_b_into_subprob_3d_v1(int *d_subprob_to_obin, int *d_subprobstartpts,
                                         int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < d_numsubprob[i]; j++) {
      d_subprob_to_obin[d_subprobstartpts[i] + j] = i;
    }
  }
}

__global__ void trivial_global_sort_index_3d(int M, int *index) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    index[i] = i;
  }
}

__global__ void fill_ghost_bins(int binsperobinx, int binsperobiny, int binsperobinz,
                                int nobinx, int nobiny, int nobinz, int *binsize) {
  int binx = threadIdx.x + blockIdx.x * blockDim.x;
  int biny = threadIdx.y + blockIdx.y * blockDim.y;
  int binz = threadIdx.z + blockIdx.z * blockDim.z;

  int nbinx = nobinx * binsperobinx;
  int nbiny = nobiny * binsperobiny;
  int nbinz = nobinz * binsperobinz;

  if (binx < nbinx && biny < nbiny && binz < nbinz) {
    int binidx = calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz, binsperobinx,
                                   binsperobiny, binsperobinz);
    int i, j, k;
    i = binx;
    j = biny;
    k = binz;
    if (binx % binsperobinx == 0) {
      i = binx - 2;
      i = i < 0 ? i + nbinx : i;
    }
    if (binx % binsperobinx == binsperobinx - 1) {
      i = binx + 2;
      i = (i >= nbinx) ? i - nbinx : i;
    }
    if (biny % binsperobiny == 0) {
      j = biny - 2;
      j = j < 0 ? j + nbiny : j;
    }
    if (biny % binsperobiny == binsperobiny - 1) {
      j = biny + 2;
      j = (j >= nbiny) ? j - nbiny : j;
    }
    if (binz % binsperobinz == 0) {
      k = binz - 2;
      k = k < 0 ? k + nbinz : k;
    }
    if (binz % binsperobinz == binsperobinz - 1) {
      k = binz + 2;
      k = (k >= nbinz) ? k - nbinz : k;
    }
    int idxtoupdate = calc_global_index(i, j, k, nobinx, nobiny, nobinz, binsperobinx,
                                        binsperobiny, binsperobinz);
    if (idxtoupdate != binidx) {
      binsize[binidx] = binsize[idxtoupdate];
    }
  }
}

__global__ void ghost_bin_pts_index(int binsperobinx, int binsperobiny, int binsperobinz,
                                    int nobinx, int nobiny, int nobinz, int *binsize,
                                    int *index, int *binstartpts, int M) {
  int binx  = threadIdx.x + blockIdx.x * blockDim.x;
  int biny  = threadIdx.y + blockIdx.y * blockDim.y;
  int binz  = threadIdx.z + blockIdx.z * blockDim.z;
  int nbinx = nobinx * binsperobinx;
  int nbiny = nobiny * binsperobiny;
  int nbinz = nobinz * binsperobinz;

  int i, j, k;
  int w = 0;
  int box[3];
  if (binx < nbinx && biny < nbiny && binz < nbinz) {
    box[0] = box[1] = box[2] = 0;
    i                        = binx;
    j                        = biny;
    k                        = binz;
    int binidx = calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz, binsperobinx,
                                   binsperobiny, binsperobinz);
    if (binx % binsperobinx == 0) {
      i      = binx - 2;
      box[0] = (i < 0);
      i      = i < 0 ? i + nbinx : i;
      w      = 1;
    }
    if (binx % binsperobinx == binsperobinx - 1) {
      i      = binx + 2;
      box[0] = (i > nbinx) * 2;
      i      = (i > nbinx) ? i - nbinx : i;
      w      = 1;
    }
    if (biny % binsperobiny == 0) {
      j      = biny - 2;
      box[1] = (j < 0);
      j      = j < 0 ? j + nbiny : j;
      w      = 1;
    }
    if (biny % binsperobiny == binsperobiny - 1) {
      j      = biny + 2;
      box[1] = (j > nbiny) * 2;
      j      = (j > nbiny) ? j - nbiny : j;
      w      = 1;
    }
    if (binz % binsperobinz == 0) {
      k      = binz - 2;
      box[2] = (k < 0);
      k      = k < 0 ? k + nbinz : k;
      w      = 1;
    }
    if (binz % binsperobinz == binsperobinz - 1) {
      k      = binz + 2;
      box[2] = (k > nbinz) * 2;
      k      = (k > nbinz) ? k - nbinz : k;
      w      = 1;
    }
    int corbinidx = calc_global_index(i, j, k, nobinx, nobiny, nobinz, binsperobinx,
                                      binsperobiny, binsperobinz);
    if (w == 1) {
      for (int n = 0; n < binsize[binidx]; n++) {
        index[binstartpts[binidx] + n] =
            M * (box[0] + box[1] * 3 + box[2] * 9) + index[binstartpts[corbinidx] + n];
      }
    }
  }
}

} // namespace common
} // namespace cufinufft
