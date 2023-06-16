#ifndef CUFINUFFT_IMPL_H
#define CUFINUFFT_IMPL_H

#include <cufinufft/types.h>

template <typename T>
int cufinufft_makeplan_impl(int type, int dim, int *nmodes, int iflag, int ntransf, T tol,
                            cufinufft_plan_t<T> **d_plan_ptr, cufinufft_opts *opts);
template <typename T>
int cufinufft_setpts_impl(int M, T *d_kx, T *d_ky, T *d_kz, int N, T *d_s, T *d_t, T *d_u, cufinufft_plan_t<T> *d_plan);
template <typename T>
int cufinufft_execute_impl(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan);

template <typename T>
int cufinufft_destroy_impl(cufinufft_plan_t<T> *d_plan);

// 1d
template <typename T>
int cufinufft1d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan);
template <typename T>
int cufinufft1d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan);

// 2d
template <typename T>
int cufinufft2d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan);
template <typename T>
int cufinufft2d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan);

// 3d
template <typename T>
int cufinufft3d1_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan);
template <typename T>
int cufinufft3d2_exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk, cufinufft_plan_t<T> *d_plan);

#endif
