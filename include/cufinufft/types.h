#ifndef CUFINUFFT_TYPES_H
#define CUFINUFFT_TYPES_H

#include <cufft.h>

#include <cufinufft_opts.h>
#include <finufft_spread_opts.h>
#include <type_traits>

#include <complex.h>

#define CUFINUFFT_BIGINT int

template <typename T>
struct cufinufft_plan_t;

template <>
struct cufinufft_plan_t<float> {
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

    int totalnumsubprob;
    int byte_now;
    float *fwkerhalf1;
    float *fwkerhalf2;
    float *fwkerhalf3;

    float *kx;
    float *ky;
    float *kz;
    cuFloatComplex *c;
    cuFloatComplex *fw;
    cuFloatComplex *fk;

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
    cudaStream_t *streams;
};

template <>
struct cufinufft_plan_t<double> {
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

    int totalnumsubprob;
    int byte_now;
    double *fwkerhalf1;
    double *fwkerhalf2;
    double *fwkerhalf3;

    double *kx;
    double *ky;
    double *kz;
    cuDoubleComplex *c;
    cuDoubleComplex *fw;
    cuDoubleComplex *fk;

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
    cudaStream_t *streams;
};

template <typename T>
struct cuda_complex_impl;
template <>
struct cuda_complex_impl<float> {
    using type = cuFloatComplex;
};
template <>
struct cuda_complex_impl<double> {
    using type = cuDoubleComplex;
};

template <typename T>
using cuda_complex = typename cuda_complex_impl<T>::type;

template <typename T>
static cufftType_t cufft_type();
template <>
inline cufftType_t cufft_type<float>() {
    return CUFFT_C2C;
}

template <>
inline cufftType_t cufft_type<double>() {
    return CUFFT_Z2Z;
}

static inline cufftResult cufft_ex(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) {
    return cufftExecC2C(plan, idata, odata, direction);
}
static inline cufftResult cufft_ex(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata,
                                   int direction) {
    return cufftExecZ2Z(plan, idata, odata, direction);
}

#endif
