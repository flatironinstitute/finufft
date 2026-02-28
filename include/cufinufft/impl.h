#ifndef CUFINUFFT_IMPL_H
#define CUFINUFFT_IMPL_H

#include <cufinufft/types.h>


// 1d
template<typename T>
void cufinufft1d_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
// 2d
template<typename T>
void cufinufft2d_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);
// 3d
template<typename T>
void cufinufft3d_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                      cufinufft_plan_t<T> *d_plan);

template<typename T>
int cufinufft_makeplan_impl(int type, int dim, const int *nmodes, int iflag, int ntransf, T tol,
                            cufinufft_plan_t<T> **d_plan_ptr, const cufinufft_opts *opts);

template<typename T>
void cufinufft_setpts_impl(int M, const T *d_kx, const T *d_ky, const T *d_kz, int N, const T *d_s, const T *d_t, const T *d_u,
                          cufinufft_plan_t<T> *d_plan);
template<typename T>
void cufinufft_execute_impl(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                           cufinufft_plan_t<T> *d_plan);

template<typename T>
void cufinufft_destroy_impl(cufinufft_plan_t<T> *d_plan);
#endif
