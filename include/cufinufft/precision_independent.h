/* These are functions that do not rely on CUFINUFFT_FLT.
   They are organized by originating file.
*/

#ifndef PRECISION_INDEPENDENT_H
#define PRECISION_INDEPENDENT_H

#include <cuComplex.h>
#define rpart(x)    (cuCreal(x))
#define ipart(x)    (cuCimag(x))
#define cmplx(x, y) (make_cuDoubleComplex(x, y))
namespace cufinufft {
namespace common {
/* Auxiliary var/func to compute power of complex number */
typedef double RT;
typedef cuDoubleComplex CT;

__device__ RT carg(const CT &z); // polar angle
__device__ RT cabs(const CT &z);
__device__ CT cpow(const CT &z, const int &n);

/* Common Kernels from spreadinterp3d */
__host__ __device__ int calc_global_index(int xidx, int yidx, int zidx, int onx, int ony,
                                          int onz, int bnx, int bny, int bnz);
__device__ int calc_global_index_v2(int xidx, int yidx, int zidx, int nbinx, int nbiny,
                                    int nbinz);

/* spreadinterp 1d */
__global__ void calc_subprob_1d(int *bin_size, int *num_subprob, int maxsubprobsize,
                                int numbins);

__global__ void map_b_into_subprob_1d(int *d_subprob_to_bin, int *d_subprobstartpts,
                                      int *d_numsubprob, int numbins);

__global__ void trivial_global_sort_index_1d(int M, int *index);

/* spreadinterp 2d */
__global__ void calc_subprob_2d(int *bin_size, int *num_subprob, int maxsubprobsize,
                                int numbins);

__global__ void map_b_into_subprob_2d(int *d_subprob_to_bin, int *d_subprobstartpts,
                                      int *d_numsubprob, int numbins);

__global__ void trivial_global_sort_index_2d(int M, int *index);

/* spreadinterp3d */
__global__ void calc_subprob_3d_v2(int *bin_size, int *num_subprob, int maxsubprobsize,
                                   int numbins);

__global__ void map_b_into_subprob_3d_v2(int *d_subprob_to_bin, int *d_subprobstartpts,
                                         int *d_numsubprob, int numbins);

__global__ void calc_subprob_3d_v1(int binsperobinx, int binsperobiny, int binsperobinz,
                                   int *bin_size, int *num_subprob, int maxsubprobsize,
                                   int numbins);

__global__ void map_b_into_subprob_3d_v1(int *d_subprob_to_obin, int *d_subprobstartpts,
                                         int *d_numsubprob, int numbins);

__global__ void trivial_global_sort_index_3d(int M, int *index);

__global__ void fill_ghost_bins(int binsperobinx, int binsperobiny, int binsperobinz,
                                int nobinx, int nobiny, int nobinz, int *binsize);

__global__ void ghost_bin_pts_index(int binsperobinx, int binsperobiny, int binsperobinz,
                                    int nobinx, int nobiny, int nobinz, int *binsize,
                                    int *index, int *binstartpts, int M);
} // namespace common
} // namespace cufinufft
#endif
