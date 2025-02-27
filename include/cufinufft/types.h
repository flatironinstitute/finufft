#ifndef CUFINUFFT_TYPES_H
#define CUFINUFFT_TYPES_H

#include <cufft.h>

#include <cufinufft/defs.h>
#include <cufinufft_opts.h>
#include <finufft_spread_opts.h>
#include <type_traits>

#include <cuComplex.h>

#define CUFINUFFT_BIGINT int

// Marco Barbone 8/5/2924, replaced the ugly trick with std::conditional
// to define cuda_complex
// by using std::conditional and std::is_same, we can define cuda_complex
// if T is float, cuda_complex<T> is cuFloatComplex
// if T is double, cuda_complex<T> is cuDoubleComplex
// where cuFloatComplex and cuDoubleComplex are defined in cuComplex.h
// TODO: migrate to cuda/std/complex and remove this
//       Issue: cufft seems not to support cuda::std::complex
//       A reinterpret_cast should be enough
template<typename T>
using cuda_complex = typename std::conditional<
    std::is_same<T, float>::value, cuFloatComplex,
    typename std::conditional<std::is_same<T, double>::value, cuDoubleComplex,
                              void>::type>::type;

template<typename T> struct cufinufft_plan_t {
  cufinufft_opts opts;
  finufft_spread_opts spopts;

  int type;
  int dim;
  CUFINUFFT_BIGINT M;
  CUFINUFFT_BIGINT nf1;
  CUFINUFFT_BIGINT nf2;
  CUFINUFFT_BIGINT nf3;
  CUFINUFFT_BIGINT ms;
  CUFINUFFT_BIGINT mt;
  CUFINUFFT_BIGINT mu;
  int ntransf;
  int batchsize;
  int iflag;
  int supports_pools;

  int totalnumsubprob;
  T *fwkerhalf1;
  T *fwkerhalf2;
  T *fwkerhalf3;

  // for type 1,2 it is a pointer to kx, ky, kz (no new allocs), for type 3 it
  // for t3: allocated as "primed" (scaled) src pts x'_j, etc
  T *kx;
  T *ky;
  T *kz;
  cuda_complex<T> *CpBatch; // working array of prephased strengths
  cuda_complex<T> *fwbatch;

  // no allocs here
  cuda_complex<T> *c;
  cuda_complex<T> *fw;
  cuda_complex<T> *fk;

  // Type 3 specific
  struct {
    T X1, C1, S1, D1, h1, gam1; // x dim: X=halfwid C=center D=freqcen h,gam=rescale,
                                // s=interval
    T X2, C2, S2, D2, h2, gam2; // y
    T X3, C3, S3, D3, h3, gam3; // z
  } type3_params;
  int N;                        // number of NU freq pts (type 3 only)
  CUFINUFFT_BIGINT nf;
  T *d_Sp;
  T *d_Tp;
  T *d_Up;
  T tol;
  // inner type 2 plan for type 3
  cufinufft_plan_t<T> *t2_plan;
  // new allocs.
  // FIXME: convert to device vectors to use resize
  cuda_complex<T> *prephase; // pre-phase, for all input NU pts
  cuda_complex<T> *deconv;   // reciprocal of kernel FT, phase, all output NU pts

  // Arrays that used in subprob method
  int *idxnupts;        // length: #nupts, index of the nupts in the bin-sorted order
  int *sortidx;         // length: #nupts, order inside the bin the nupt belongs to
  int *numsubprob;      // length: #bins,  number of subproblems in each bin
  int *binsize;         // length: #bins, number of nonuniform ponits in each bin
  int *binstartpts;     // length: #bins, exclusive scan of array binsize
  int *subprob_to_bin;  // length: #subproblems, the bin the subproblem works on
  int *subprobstartpts; // length: #bins, exclusive scan of array numsubprob

  // Arrays for 3d (need to sort out)
  int *numnupts;
  int *subprob_to_nupts;

  cufftHandle fftplan;
  cudaStream_t stream;
};

template<typename T> static inline constexpr cufftType_t cufft_type();
template<> inline constexpr cufftType_t cufft_type<float>() { return CUFFT_C2C; }

template<> inline constexpr cufftType_t cufft_type<double>() { return CUFFT_Z2Z; }

static inline cufftResult cufft_ex(cufftHandle plan, cufftComplex *idata,
                                   cufftComplex *odata, int direction) {
  return cufftExecC2C(plan, idata, odata, direction);
}
static inline cufftResult cufft_ex(cufftHandle plan, cufftDoubleComplex *idata,
                                   cufftDoubleComplex *odata, int direction) {
  return cufftExecZ2Z(plan, idata, odata, direction);
}

#endif
