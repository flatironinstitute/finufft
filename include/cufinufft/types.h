#ifndef CUFINUFFT_TYPES_H
#define CUFINUFFT_TYPES_H

#include <cufft.h>

#include <cufinufft/defs.h>
#include <cufinufft_opts.h>
#include <finufft_spread_opts.h>
#include <type_traits>

#include <complex.h>

#define CUFINUFFT_BIGINT int

// Ugly trick to map a template to a fixed type, here cuda_complex<T>
template<typename T> struct cuda_complex_impl;
template<> struct cuda_complex_impl<float> {
  using type = cuFloatComplex;
};
template<> struct cuda_complex_impl<double> {
  using type = cuDoubleComplex;
};

template<typename T> using cuda_complex = typename cuda_complex_impl<T>::type;

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
  int maxbatchsize;
  int iflag;
  int supports_pools;

  int totalnumsubprob;
  T *fwkerhalf1;
  T *fwkerhalf2;
  T *fwkerhalf3;

  T *kx;
  T *ky;
  T *kz;
  cuda_complex<T> *c;
  cuda_complex<T> *fw;
  cuda_complex<T> *fk;

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

  // Temporary variables to do fseries precomputation
  std::complex<double> fseries_precomp_a[3 * MAX_NQUAD];
  T fseries_precomp_f[3 * MAX_NQUAD];

  cufftHandle fftplan;
  cudaStream_t stream;
};

template<typename T> static cufftType_t cufft_type();
template<> inline cufftType_t cufft_type<float>() { return CUFFT_C2C; }

template<> inline cufftType_t cufft_type<double>() { return CUFFT_Z2Z; }

static inline cufftResult cufft_ex(cufftHandle plan, cufftComplex *idata,
                                   cufftComplex *odata, int direction) {
  return cufftExecC2C(plan, idata, odata, direction);
}
static inline cufftResult cufft_ex(cufftHandle plan, cufftDoubleComplex *idata,
                                   cufftDoubleComplex *odata, int direction) {
  return cufftExecZ2Z(plan, idata, odata, direction);
}

#endif
