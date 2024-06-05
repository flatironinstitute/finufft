#ifndef __CUDECONVOLVE_H__
#define __CUDECONVOLVE_H__

#include <cufinufft/types.h>

namespace cufinufft {
namespace deconvolve {
template<typename T, int modeord>
__global__ void deconvolve_1d(int ms, int nf1, int fw_width, cuda_complex<T> *fw,
                              cuda_complex<T> *fk, T *fwkerhalf1);
template<typename T, int modeord>
__global__ void amplify_1d(int ms, int nf1, int fw_width, cuda_complex<T> *fw,
                           cuda_complex<T> *fk, T *fwkerhalf2);
template<typename T, int modeord>
__global__ void deconvolve_2d(int ms, int mt, int nf1, int nf2, int fw_width,
                              cuda_complex<T> *fw, cuda_complex<T> *fk, T *fwkerhalf1,
                              T *fwkerhalf2);
template<typename T, int modeord>
__global__ void amplify_2d(int ms, int mt, int nf1, int nf2, int fw_width,
                           cuda_complex<T> *fw, cuda_complex<T> *fk, T *fwkerhalf1,
                           T *fwkerhalf2);

template<typename T, int modeord>
__global__ void deconvolve_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3,
                              int fw_width, cuda_complex<T> *fw, cuda_complex<T> *fk,
                              T *fwkerhalf1, T *fwkerhalf2, T *fwkerhalf3);
template<typename T, int modeord>
__global__ void amplify_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3,
                           int fw_width, cuda_complex<T> *fw, cuda_complex<T> *fk,
                           T *fwkerhalf1, T *fwkerhalf2, T *fwkerhalf3);

template<typename T, int modeord>
int cudeconvolve1d(cufinufft_plan_t<T> *d_mem, int blksize);
template<typename T, int modeord>
int cudeconvolve2d(cufinufft_plan_t<T> *d_mem, int blksize);
template<typename T, int modeord>
int cudeconvolve3d(cufinufft_plan_t<T> *d_mem, int blksize);
} // namespace deconvolve
} // namespace cufinufft
#endif
