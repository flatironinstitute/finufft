#ifndef __CUDECONVOLVE_H__
#define __CUDECONVOLVE_H__

#include <cufinufft_eitherprec.h>
#include <cufinufft/types.h>

namespace cufinufft {
namespace deconvolve {

template <typename T>
__global__ void Deconvolve_1d(int ms, int nf1, int fw_width, cuda_complex<T> *fw, cuda_complex<T> *fk, T *fwkerhalf1);
template <typename T>
__global__ void Amplify_1d(int ms, int nf1, int fw_width, cuda_complex<T> *fw, cuda_complex<T> *fk, T *fwkerhalf2);
template <typename T>
__global__ void Deconvolve_2d(int ms, int mt, int nf1, int nf2, int fw_width, cuda_complex<T> *fw, cuda_complex<T> *fk,
                              T *fwkerhalf1, T *fwkerhalf2);
template <typename T>
__global__ void Amplify_2d(int ms, int mt, int nf1, int nf2, int fw_width, cuda_complex<T> *fw, cuda_complex<T> *fk,
                           T *fwkerhalf1, T *fwkerhalf2);

template <typename T>
__global__ void Deconvolve_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, int fw_width, cuda_complex<T> *fw, cuda_complex<T> *fk,
                              T *fwkerhalf1, T *fwkerhalf2, T *fwkerhalf3);
template <typename T>
__global__ void Amplify_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, int fw_width, cuda_complex<T> *fw, cuda_complex<T> *fk,
                           T *fwkerhalf1, T *fwkerhalf2, T *fwkerhalf3);

template <typename T>
int cudeconvolve1d(cufinufft_plan_template<T> *d_mem, int blksize);
template <typename T>
int cudeconvolve2d(cufinufft_plan_template<T> *d_mem, int blksize);
template <typename T>
int cudeconvolve3d(cufinufft_plan_template<T> *d_mem, int blksize);
} // namespace convolve
} // namespace cufinufft
#endif
