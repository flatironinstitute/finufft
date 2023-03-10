#ifndef __CUSPREADINTERP_H__
#define __CUSPREADINTERP_H__

#include <cufinufft/types.h>
#include <cufinufft_eitherprec.h>
#include <finufft_spread_opts.h>

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
// FIXME: SO MUCH BRANCHING
#define RESCALE(x, N, p)                                                                                               \
    (p ? ((x * M_1_2PI + (x < -PI ? 1.5 : (x >= PI ? -0.5 : 0.5))) * N) : (x < 0 ? x + N : (x >= N ? x - N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.

namespace cufinufft {
namespace spreadinterp {

template <typename T>
inline T evaluate_kernel(T x, const finufft_spread_opts &opts);

template <typename T>
inline int setup_spreader(finufft_spread_opts &opts, T eps, T upsampfac, int kerevalmeth);

template <typename T>
static __forceinline__ __device__ T evaluate_kernel(T x, T es_c, T es_beta, int ns)
/* ES ("exp sqrt") kernel evaluation at single real argument:
   phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_*
    2/17/17 */
{
    return abs(x) < ns / 2.0 ? exp(es_beta * (sqrt(1.0 - es_c * x * x))) : 0.0;
}

template <typename T>
static __inline__ __device__ void eval_kernel_vec_Horner(T *ker, const T x, const int w, const double upsampfac)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
    T z = 2 * x + w - 1.0; // scale so local grid offset z in [-1,1]
    // insert the auto-generated code which expects z, w args, writes to ker...
    if (upsampfac == 2.0) { // floating point equality is fine here
#include "cufinufft/contrib/ker_horner_allw_loop.inc"
    }
}

template <typename T>
static __inline__ __device__ void eval_kernel_vec(T *ker, const T x, const double w, const double es_c,
                                                  const double es_beta) {
    for (int i = 0; i < w; i++) {
        ker[i] = evaluate_kernel(abs(x + i), es_c, es_beta, w);
    }
}

// Kernels for 1D codes
/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for NUptsdriven Method */
template <typename T>
__global__ void Spread_1d_NUptsdriven(T *x, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns, int nf1,
                                      T es_c, T es_beta, int *idxnupts, int pirange);
template <typename T>
__global__ void Spread_1d_NUptsdriven_Horner(T *x, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns,
                                             int nf1, T sigma, int *idxnupts, int pirange);

/* Kernels for SubProb Method */
// SubProb properties
template <typename T>
__global__ void CalcBinSize_noghost_1d(int M, int nf1, int bin_size_x, int nbinx, int *bin_size, T *x,
                                       int *sortidx, int pirange);
template <typename T>
__global__ void CalcInvertofGlobalSortIdx_1d(int M, int bin_size_x, int nbinx, int *bin_startpts, int *sortidx,
                                             T *x, int *index, int pirange, int nf1);

// Main Spreading Kernel
template <typename T>
__global__ void Spread_1d_Subprob(T *x, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns, int nf1,
                                  T es_c, T es_beta, T sigma, int *binstartpts,
                                  int *bin_size, int bin_size_x, int *subprob_to_bin, int *subprobstartpts,
                                  int *numsubprob, int maxsubprobsize, int nbinx, int *idxnupts, int pirange);
template <typename T>
__global__ void Spread_1d_Subprob_Horner(T *x, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns, int nf1,
                                         T sigma, int *binstartpts, int *bin_size, int bin_size_x,
                                         int *subprob_to_bin, int *subprobstartpts, int *numsubprob, int maxsubprobsize,
                                         int nbinx, int *idxnupts, int pirange);
/* ---------------------------Interpolation Kernels---------------------------*/
/* Kernels for NUptsdriven Method */
template <typename T>
__global__ void Interp_1d_NUptsdriven(T *x, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns, int nf2,
                                      T es_c, T es_beta, int *idxnupts, int pirange);
template <typename T>
__global__ void Interp_1d_NUptsdriven_Horner(T *x, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns, int nf2,
                                             T sigma, int *idxnupts, int pirange);

// Kernels for 2D codes
/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for NUptsdriven Method */
template <typename T>
__global__ void Spread_2d_NUptsdriven(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns,
                                      int nf1, int nf2, T es_c, T es_beta, int *idxnupts,
                                      int pirange);
template <typename T>
__global__ void Spread_2d_NUptsdriven_Horner(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M,
                                             const int ns, int nf1, int nf2, T sigma, int *idxnupts,
                                             int pirange);

/* Kernels for SubProb Method */
// SubProb properties
template <typename T>
__global__ void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                                       int *bin_size, T *x, T *y, int *sortidx, int pirange);
template <typename T>
__global__ void CalcInvertofGlobalSortIdx_2d(int M, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                                             int *bin_startpts, int *sortidx, T *x, T *y,
                                             int *index, int pirange, int nf1, int nf2);

// Main Spreading Kernel
template <typename T>
__global__ void Spread_2d_Subprob(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns, int nf1,
                                  int nf2, T es_c, T es_beta, T sigma,
                                  int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y, int *subprob_to_bin,
                                  int *subprobstartpts, int *numsubprob, int maxsubprobsize, int nbinx, int nbiny,
                                  int *idxnupts, int pirange);
template <typename T>
__global__ void Spread_2d_Subprob_Horner(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns,
                                         int nf1, int nf2, T sigma, int *binstartpts, int *bin_size,
                                         int bin_size_x, int bin_size_y, int *subprob_to_bin, int *subprobstartpts,
                                         int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, int *idxnupts,
                                         int pirange);

/* Kernels for Paul's Method */
template <typename T>
__global__ void LocateFineGridPos_Paul(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                                       int *bin_size, int ns, T *x, T *y, int *sortidx,
                                       int *finegridsize, int pirange);
template <typename T>
__global__ void CalcInvertofGlobalSortIdx_Paul(int nf1, int nf2, int M, int bin_size_x, int bin_size_y, int nbinx,
                                               int nbiny, int ns, T *x, T *y,
                                               int *finegridstartpts, int *sortidx, int *index, int pirange);
template <typename T>
__global__ void Spread_2d_Subprob_Paul(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns,
                                       int nf1, int nf2, T es_c, T es_beta, T sigma,
                                       int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y,
                                       int *subprob_to_bin, int *subprobstartpts, int *numsubprob, int maxsubprobsize,
                                       int nbinx, int nbiny, int *idxnupts, int *fgstartpts, int *finegridsize,
                                       int pirange);

/* ---------------------------Interpolation Kernels---------------------------*/
/* Kernels for NUptsdriven Method */
template <typename T>
__global__ void Interp_2d_NUptsdriven(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns,
                                      int nf1, int nf2, T es_c, T es_beta, int *idxnupts,
                                      int pirange);
template <typename T>
__global__ void Interp_2d_NUptsdriven_Horner(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M,
                                             const int ns, int nf1, int nf2, T sigma, int *idxnupts,
                                             int pirange);
/* Kernels for Subprob Method */
template <typename T>
__global__ void Interp_2d_Subprob(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns, int nf1,
                                  int nf2, T es_c, T es_beta, T sigma,
                                  int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y, int *subprob_to_bin,
                                  int *subprobstartpts, int *numsubprob, int maxsubprobsize, int nbinx, int nbiny,
                                  int *idxnupts, int pirange);
template <typename T>
__global__ void Interp_2d_Subprob_Horner(T *x, T *y, cuda_complex<T> *c, cuda_complex<T> *fw, int M, const int ns,
                                         int nf1, int nf2, T sigma, int *binstartpts, int *bin_size,
                                         int bin_size_x, int bin_size_y, int *subprob_to_bin, int *subprobstartpts,
                                         int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, int *idxnupts,
                                         int pirange);

// Kernels for 3D codes
/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for Bin Sort NUpts */
template <typename T>
__global__ void CalcBinSize_noghost_3d(int M, int nf1, int nf2, int nf3, int bin_size_x, int bin_size_y, int bin_size_z,
                                       int nbinx, int nbiny, int nbinz, int *bin_size, T *x,
                                       T *y, T *z, int *sortidx, int pirange);
template <typename T>
__global__ void CalcInvertofGlobalSortIdx_3d(int M, int bin_size_x, int bin_size_y, int bin_size_z, int nbinx,
                                             int nbiny, int nbinz, int *bin_startpts, int *sortidx, T *x,
                                             T *y, T *z, int *index, int pirange, int nf1,
                                             int nf2, int nf3);

/* Kernels for NUptsdriven Method */
template <typename T>
__global__ void Spread_3d_NUptsdriven_Horner(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw,
                                             int M, const int ns, int nf1, int nf2, int nf3, T sigma,
                                             int *idxnupts, int pirange);
template <typename T>
__global__ void Spread_3d_NUptsdriven(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M,
                                      const int ns, int nf1, int nf2, int nf3, T es_c,
                                      T es_beta, int *idxnupts, int pirange);

/* Kernels for Subprob Method */
template <typename T>
__global__ void Spread_3d_Subprob_Horner(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw,
                                         int M, const int ns, int nf1, int nf2, int nf3, T sigma,
                                         int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y,
                                         int bin_size_z, int *subprob_to_bin, int *subprobstartpts, int *numsubprob,
                                         int maxsubprobsize, int nbinx, int nbiny, int nbinz, int *idxnupts,
                                         int pirange);
template <typename T>
__global__ void Spread_3d_Subprob(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M,
                                  const int ns, int nf1, int nf2, int nf3, T es_c, T es_beta,
                                  int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
                                  int *subprob_to_bin, int *subprobstartpts, int *numsubprob, int maxsubprobsize,
                                  int nbinx, int nbiny, int nbinz, int *idxnupts, int pirange);

/* Kernels for Block BlockGather Method */
template <typename T>
__global__ void LocateNUptstoBins_ghost(int M, int bin_size_x, int bin_size_y, int bin_size_z, int nbinx, int nbiny,
                                        int nbinz, int binsperobinx, int binsperobiny, int binsperobinz, int *bin_size,
                                        T *x, T *y, T *z, int *sortidx, int pirange,
                                        int nf1, int nf2, int nf3);
template <typename T>
__global__ void Temp(int binsperobinx, int binsperobiny, int binsperobinz, int nbinx, int nbiny, int nbinz,
                     int *binsize);
template <typename T>
__global__ void CalcInvertofGlobalSortIdx_ghost(int M, int bin_size_x, int bin_size_y, int bin_size_z, int nbinx,
                                                int nbiny, int nbinz, int binsperobinx, int binsperobiny,
                                                int binsperobinz, int *bin_startpts, int *sortidx, T *x,
                                                T *y, T *z, int *index, int pirange, int nf1,
                                                int nf2, int nf3);
template <typename T>
__global__ void Spread_3d_BlockGather(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M,
                                      const int ns, int nf1, int nf2, int nf3, T es_c,
                                      T es_beta, T sigma, int *binstartpts, int obin_size_x,
                                      int obin_size_y, int obin_size_z, int binsperobin, int *subprob_to_bin,
                                      int *subprobstartpts, int maxsubprobsize, int nobinx, int nobiny, int nobinz,
                                      int *idxnupts, int pirange);
template <typename T>
__global__ void Spread_3d_BlockGather_Horner(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw,
                                             int M, const int ns, int nf1, int nf2, int nf3, T es_c,
                                             T es_beta, T sigma, int *binstartpts,
                                             int obin_size_x, int obin_size_y, int obin_size_z, int binsperobin,
                                             int *subprob_to_bin, int *subprobstartpts, int maxsubprobsize, int nobinx,
                                             int nobiny, int nobinz, int *idxnupts, int pirange);

/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for NUptsdriven Method */
template <typename T>
__global__ void Interp_3d_NUptsdriven_Horner(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw,
                                             int M, const int ns, int nf1, int nf2, int nf3, T sigma,
                                             int *idxnupts, int pirange);
template <typename T>
__global__ void Interp_3d_NUptsdriven(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M,
                                      const int ns, int nf1, int nf2, int nf3, T es_c,
                                      T es_beta, int *idxnupts, int pirange);

/* Kernels for Subprob Method */
template <typename T>
__global__ void Interp_3d_Subprob_Horner(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw,
                                         int M, const int ns, int nf1, int nf2, int nf3, T sigma,
                                         int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y,
                                         int bin_size_z, int *subprob_to_bin, int *subprobstartpts, int *numsubprob,
                                         int maxsubprobsize, int nbinx, int nbiny, int nbinz, int *idxnupts,
                                         int pirange);
template <typename T>
__global__ void Interp_3d_Subprob(T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M,
                                  const int ns, int nf1, int nf2, int nf3, T es_c, T es_beta,
                                  int *binstartpts, int *bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
                                  int *subprob_to_bin, int *subprobstartpts, int *numsubprob, int maxsubprobsize,
                                  int nbinx, int nbiny, int nbinz, int *idxnupts, int pirange);

/* C wrapper for calling CUDA kernels */
// Wrapper for testing spread, interpolation only
template <typename T>
int cufinufft_spread1d(int nf1, cuda_complex<T> *d_fw, int M, T *d_kx, cuda_complex<T> *d_c,
                       cufinufft_plan_template<T> *d_plan);
template <typename T>
int cufinufft_interp1d(int nf1, cuda_complex<T> *d_fw, int M, T *d_kx, cuda_complex<T> *d_c,
                       cufinufft_plan_template<T> *d_plan);
template <typename T>
int cufinufft_spread2d(int nf1, int nf2, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, cuda_complex<T> *d_c,
                       cufinufft_plan_template<T> *d_plan);
template <typename T>
int cufinufft_interp2d(int nf1, int nf2, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, cuda_complex<T> *d_c,
                       cufinufft_plan_template<T> *d_plan);
template <typename T>
int cufinufft_spread3d(int nf1, int nf2, int nf3, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, T *d_kz,
                       cuda_complex<T> *d_c, cufinufft_plan_template<T> *dplan);
template <typename T>
int cufinufft_interp3d(int nf1, int nf2, int nf3, cuda_complex<T> *d_fw, int M, T *d_kx, T *d_ky, T *d_kz,
                       cuda_complex<T> *d_c, cufinufft_plan_template<T> *dplan);

// Functions for calling different methods of spreading & interpolation
template <typename T>
int cuspread1d(cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuinterp1d(cufinufft_plan_template<T> *d_plan, int blksize);

template <typename T>
int cuspread2d(cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuinterp2d(cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuspread3d(cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuinterp3d(cufinufft_plan_template<T> *d_plan, int blksize);

// Wrappers for methods of spreading
template <typename T>
int cuspread1d_nuptsdriven_prop(int nf1, int M, cufinufft_plan_template<T> *d_plan);
template <typename T>
int cuspread1d_nuptsdriven(int nf1, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuspread1d_subprob_prop(int nf1, int M, cufinufft_plan_template<T> *d_plan);
template <typename T>
int cuspread1d_subprob(int nf1, int M, cufinufft_plan_template<T> *d_plan, int blksize);

template <typename T>
int cuspread2d_nuptsdriven_prop(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan);
template <typename T>
int cuspread2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan);
template <typename T>
int cuspread2d_paul_prop(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan);
template <typename T>
int cuspread2d_subprob(int nf1, int nf2, int m, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuspread2d_paul(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuspread3d_nuptsdriven_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan);
template <typename T>
int cuspread3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuspread3d_blockgather_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan);
template <typename T>
int cuspread3d_blockgather(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan);
template <typename T>
int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan, int blksize);

// Wrappers for methods of interpolation
template <typename T>
int cuinterp1d_nuptsdriven(int nf1, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuinterp2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuinterp2d_subprob(int nf1, int nf2, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuinterp3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan, int blksize);
template <typename T>
int cuinterp3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_template<T> *d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft
#endif
