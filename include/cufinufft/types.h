#ifndef TYPES_H
#define TYPES_H

#include <type_traits>
#include <cufft.h>

#include <finufft_spread_opts.h>
#include <cufinufft_types.h>
#include <cufinufft_opts.h>

template <typename T> struct cuda_complex_impl;
template <> struct cuda_complex_impl<float> { using type = cuFloatComplex; };
template <> struct cuda_complex_impl<double> { using type = cuDoubleComplex; };

template <typename T>
using cuda_complex = typename cuda_complex_impl<std::remove_reference_t<T>>::type;

template <typename T>
struct cufinufft_plan_template {
    cufinufft_opts opts;
    finufft_spread_opts spopts;

    int type;
    int dim;
    int M;
    int nf1;
    int nf2;
    int nf3;
    int ms;
    int mt;
    int mu;
    int ntransf;
    int maxbatchsize;
    int iflag;

    int totalnumsubprob;
    int byte_now;
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

    // Extra arrays for Paul's method
    int *finegridsize;
    int *fgstartpts;

    // Arrays for 3d (need to sort out)
    int *numnupts;
    int *subprob_to_nupts;

    cufftHandle fftplan;
    cudaStream_t *streams;
};

#endif
