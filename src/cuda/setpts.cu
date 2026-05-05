// "setpts" stage: cufinufft_plan_t<T>::setpts and the type-1/2 helper.
// Mirrors CPU src/setpts.cpp.

#include <iostream>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

#include <cufinufft.h>
#include <cufinufft/fft.hpp>
#include <cufinufft/spreadinterp.hpp>
#include <cufinufft/types.hpp>
#include <cufinufft/utils.hpp>

#include <finufft_common/constants.h>
#include <finufft_errors.h>
#include <thrust/device_vector.h>

template<typename T>
void cufinufft_plan_t<T>::setpts_type12(int M_, const T *d_kx, const T *d_ky,
                                        const T *d_kz)
/*
    "setNUpts" stage (in single or double precision).

    In this stage, we
        (1) set the number and locations of nonuniform points
        (2) allocate gpu arrays with size determined by number of nupts
        (3) rescale x,y,z coordinates for spread/interp (on gpu, rescaled
            coordinates are stored)
        (4) determine the spread/interp properties that only relates to the
            locations of nupts (see spreadinterp.cu for what has been done in
            function cuspread_<method>_prop() )

        See ../docs/cppdoc.md for main user-facing documentation.
        Here is the old developer docs, which are useful only to translate
        the argument names from the user-facing ones:

    Input:
    M                 number of nonuniform points
    d_kx, d_ky, d_kz  gpu array of x,y,z locations of sources (each a size M
                      T array) in [-pi, pi). set h_kz to "NULL" if dimension
                      is less than 3. same for h_ky for dimension 1.
    N, d_s, d_t, d_u  not used for type1, type2. set to 0 and NULL.

    Input/Output:
    d_plan            pointer to a CUFINUFFT_PLAN_S. Variables and arrays inside
                      the plan are set and allocated.

        Returned value:
        a status flag: 0 if success, otherwise an error occurred

Notes: the type T means either single or double, matching the
    precision of the library version called.

    Melody Shih 07/25/19; Barnett 2/16/21 moved out docs.
*/
{
  M = M_;

  allocate_nupts();

  kxyz = {d_kx, d_ky, d_kz};

  indexSort();

  if (opts.debug) {
    printf("[cufinufft] plan->M=%d\n", M);
  }
}

template<typename T>
void cufinufft_plan_t<T>::setpts(int nj, const T *d_kx, const T *d_ky, const T *d_kz,
                                 int nk, const T *d_s, const T *d_t, const T *d_u) {
  DeviceSwitcher switcher(opts.gpu_device_id);
  // type 1 and type 2 setpts
  if (type == 1 || type == 2) {
    return setpts_type12(nj, d_kx, d_ky, d_kz);
  }
  // type 3 setpts

  // This code follows the same implementation of the CPU code in finufft and uses similar
  // variables names where possible. However, the use of GPU routines and paradigms make
  // it harder to follow. To understand the code, it is recommended to read the CPU code
  // first.

  cuda::std::array<const T *, 3> d_kxyz = {d_kx, d_ky, d_kz};
  cuda::std::array<const T *, 3> d_stu  = {d_s, d_t, d_u};

  M = nj;
  N = nk;
  if (N < 0) {
    fprintf(stderr, "[cufinufft] Invalid N (%d): cannot be negative.\n", N);
    throw int(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  }
  if (N > MAX_NF) {
    fprintf(stderr, "[cufinufft] Invalid N (%d): cannot be greater than %d.\n", N,
            MAX_NF);
    throw int(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  }
  if (dim > 0 && d_s == nullptr) {
    fprintf(stderr, "[%s] Error: d_s is nullptr but dim > 0.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }

  if (dim > 1 && d_t == nullptr) {
    fprintf(stderr, "[%s] Error: d_t is nullptr but dim > 1.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }

  if (dim > 2 && d_u == nullptr) {
    fprintf(stderr, "[%s] Error: d_u is nullptr but dim > 2.\n", __func__);
    throw int(FINUFFT_ERR_INVALID_ARGUMENT);
  }

  using namespace cufinufft::utils;
  for (int idim = 0; idim < dim; ++idim) {
    const auto [xx, cc]           = arraywidcen<T>(M, d_kxyz[idim], stream);
    type3_params.X[idim]          = xx;
    type3_params.C[idim]          = cc;
    const auto [SS, DD]           = arraywidcen<T>(N, d_stu[idim], stream);
    const auto [nfnf, hh, gamgam] = set_nhg_type3(SS, xx);
    nf123[idim]                   = nfnf;
    type3_params.S[idim]          = SS;
    type3_params.D[idim]          = DD;
    type3_params.h[idim]          = hh;
    type3_params.gam[idim]        = gamgam;
  }
  if (opts.debug) {
    printf("[%s]", __func__);
    printf("\tM=%d N=%d\n", M, N);
    printf("\tX1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%d h1=%.3g\t\n",
           type3_params.X[0], type3_params.C[0], type3_params.S[0], type3_params.D[0],
           type3_params.gam[0], nf123[0], type3_params.h[0]);
    if (dim > 1) {
      printf("\tX2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%d h2=%.3g\n",
             type3_params.X[1], type3_params.C[1], type3_params.S[1], type3_params.D[1],
             type3_params.gam[1], nf123[1], type3_params.h[1]);
    }
    if (dim > 2) {
      printf("\tX3=%.3g C3=%.3g S3=%.3g D3=%.3g gam3=%g nf3=%d h3=%.3g\n",
             type3_params.X[2], type3_params.C[2], type3_params.S[2], type3_params.D[2],
             type3_params.gam[2], nf123[2], type3_params.h[2]);
    }
  }
  nf = nf123[0] * nf123[1] * nf123[2];

  // FIXME: MAX_NF might be too small...
  if (nf * opts.gpu_maxbatchsize > MAX_NF) {
    fprintf(stderr,
            "[%s t3] fwBatch would be bigger than MAX_NF, not attempting malloc!\n",
            __func__);
    throw int(FINUFFT_ERR_MAXNALLOC);
  }

  for (int idim = 0; idim < dim; ++idim) {
    kxyzp[idim].resize(M);
    kxyz[idim] = dethrust(kxyzp[idim]);
    STUp[idim].resize(N);
    STU[idim] = dethrust(STUp[idim]);
  }
  prephase.resize(M);
  deconv.resize(N);

  // NOTE: init-captures are not allowed for extended __host__ __device__ lambdas

  for (int idim = 0; idim < dim; ++idim) {
    const auto ig = T(1) / type3_params.gam[idim];
    const auto C  = -type3_params.C[idim];
    thrust::transform(thrust::cuda::par.on(stream), d_kxyz[idim], d_kxyz[idim] + M,
                      dethrust(kxyzp[idim]), [ig, C] __host__ __device__(const T x) -> T {
                        return (x + C) * ig;
                      });
  }
  if (type3_params.D[0] != 0 || type3_params.D[1] != 0 || type3_params.D[2] != 0) {
    // if ky is null, use kx for ky and kz
    // this is not the most efficient implementation, but it is the most compact
    const auto iterator =
        thrust::make_zip_iterator(thrust::make_tuple(d_kx,
                                                     // to avoid out of bounds access, use
                                                     // kx if ky is null
                                                     (dim > 1) ? d_ky : d_kx,
                                                     // same idea as above
                                                     (dim > 2) ? d_kz : d_kx));
    const auto D        = type3_params.D;
    const auto realsign = iflag >= 0 ? T(1) : T(-1);
    thrust::transform(
        thrust::cuda::par.on(stream), iterator, iterator + M, dethrust(prephase),
        [D, realsign] __host__
        __device__(const thrust::tuple<T, T, T> &tuple) -> cuda_complex<T> {
          const auto x = thrust::get<0>(tuple);
          const auto y = thrust::get<1>(tuple);
          const auto z = thrust::get<2>(tuple);
          // no branching because D[1] and D[2] are 0 if dim < 2 and dim < 3
          // this is generally faster on GPU
          const auto phase = D[0] * x + D[1] * y + D[2] * z;
          // TODO: nvcc should have the sincos function
          //       check the cos + i*sin
          //       ref: https://en.wikipedia.org/wiki/Cis_(mathematics)
          return cuda_complex<T>{std::cos(phase), std::sin(phase) * realsign};
        });
  } else {
    thrust::fill(thrust::cuda::par.on(stream), dethrust(prephase), dethrust(prephase) + M,
                 cuda_complex<T>{1, 0});
  }

  for (int idim = 0; idim < dim; ++idim) {
    const auto scale = type3_params.h[idim] * type3_params.gam[idim];
    const auto D     = -type3_params.D[idim];
    thrust::transform(
        thrust::cuda::par.on(stream), d_stu[idim], d_stu[idim] + N, dethrust(STUp[idim]),
        [scale, D] __host__ __device__(const T s) -> T { return scale * (s + D); });
  }
  { // here we declare phi_hat1, phi_hat2, and phi_hat3
    // and the precomputed data for the fseries kernel
    using namespace cufinufft::common;

    std::array<T, 3 * MAX_NQUAD> nuft_precomp_z{};
    std::array<T, 3 * MAX_NQUAD> nuft_precomp_f{};
    thrust::device_vector<T> d_nuft_precomp_z(3 * MAX_NQUAD);
    thrust::device_vector<T> d_nuft_precomp_f(3 * MAX_NQUAD);
    cuda::std::array<gpu_array<T>, 3> phi_hat123(
        {gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}});
    for (int idim = 0; idim < dim; ++idim) phi_hat123[idim].resize(N);
    for (int idim = 0; idim < dim; ++idim)
      onedim_nuft_kernel_precomp<T>(nuft_precomp_f.data() + idim * MAX_NQUAD,
                                    nuft_precomp_z.data() + idim * MAX_NQUAD, spopts);
    // copy the precomputed data to the device using thrust
    thrust::copy(nuft_precomp_z.begin(), nuft_precomp_z.end(), d_nuft_precomp_z.begin());
    thrust::copy(nuft_precomp_f.begin(), nuft_precomp_f.end(), d_nuft_precomp_f.begin());
    // sync the stream before calling the kernel might be needed
    nuft_kernel_compute(dim, {N, N, N}, d_nuft_precomp_f.data().get(),
                        d_nuft_precomp_z.data().get(), STU, phi_hat123, spopts.nspread,
                        stream);

    const auto is_c_finite = std::isfinite(type3_params.C[0]) &&
                             std::isfinite(type3_params.C[1]) &&
                             std::isfinite(type3_params.C[2]);
    const auto is_c_nonzero =
        type3_params.C[0] != 0 || type3_params.C[1] != 0 || type3_params.C[2] != 0;

    const auto phi_hat_iterator = thrust::make_zip_iterator(
        thrust::make_tuple(phi_hat123[0].data(),
                           // to avoid out of bounds access, use phi_hat1 if dim < 2
                           dim > 1 ? phi_hat123[1].data() : phi_hat123[0].data(),
                           // to avoid out of bounds access, use phi_hat1 if dim < 3
                           dim > 2 ? phi_hat123[2].data() : phi_hat123[0].data()));
    auto xdim = dim;
    thrust::transform(thrust::cuda::par.on(stream), phi_hat_iterator,
                      phi_hat_iterator + N, dethrust(deconv),
                      [xdim] __host__
                      __device__(const thrust::tuple<T, T, T> tuple) -> cuda_complex<T> {
                        auto phiHat = thrust::get<0>(tuple);
                        // in case dim < 2 or dim < 3, multiply by 1
                        phiHat *= (xdim > 1) ? thrust::get<1>(tuple) : T(1);
                        phiHat *= (xdim > 2) ? thrust::get<2>(tuple) : T(1);
                        return {T(1) / phiHat, T(0)};
                      });

    if (is_c_finite && is_c_nonzero) {
      const auto c1       = type3_params.C[0];
      const auto c2       = type3_params.C[1];
      const auto c3       = type3_params.C[2];
      const auto d1       = -type3_params.D[0];
      const auto d2       = -type3_params.D[1];
      const auto d3       = -type3_params.D[2];
      const auto realsign = iflag >= 0 ? T(1) : T(-1);
      // passing d_s three times if dim == 1 because d_t and d_u are not allocated
      // passing d_s and d_t if dim == 2 because d_u is not allocated
      const auto phase_iterator = thrust::make_zip_iterator(
          thrust::make_tuple(d_s, dim > 1 ? d_t : d_s, dim > 2 ? d_u : d_s));
      thrust::transform(
          thrust::cuda::par.on(stream), phase_iterator, phase_iterator + N,
          dethrust(deconv), dethrust(deconv),
          [c1, c2, c3, d1, d2, d3, realsign] __host__
          __device__(const thrust::tuple<T, T, T> tuple,
                     cuda_complex<T> deconv) -> cuda_complex<T> {
            // d2 and d3 are 0 if dim < 2 and dim < 3
            const auto phase = c1 * (thrust::get<0>(tuple) + d1) +
                               c2 * (thrust::get<1>(tuple) + d2) +
                               c3 * (thrust::get<2>(tuple) + d3);
            return cuda_complex<T>{std::cos(phase), realsign * std::sin(phase)} * deconv;
          });
    }
    // exiting the block frees the memory allocated for phi_hat1, phi_hat2, and phi_hat3
    // and the precomputed data for the fseries kernel
    // since GPU memory is expensive, we should free it as soon as possible
  }

  allocate_subprob_state();

  setpts_type12(M, kxyz[0], kxyz[1], kxyz[2]);
  {
    int t2modes[]               = {nf123[0], nf123[1], nf123[2]};
    cufinufft_opts t2opts       = opts;
    t2opts.gpu_spreadinterponly = 0;
    t2opts.gpu_method           = 0;
    // Release the old inner plan before allocating the new one to
    // avoid holding both in memory at the same time, which could be wasteful.
    t2_plan.reset();
    t2_plan = std::make_unique<cufinufft_plan_t<T>>(2, dim, t2modes, iflag, batchsize,
                                                    tol, t2opts);
    t2_plan->setpts_type12(N, STU[0], STU[1], STU[2]);
    if (t2_plan->spopts.spread_direction != 2) {
      fprintf(stderr, "[%s] inner t2 plan cufinufft_setpts_12 wrong direction\n",
              __func__);
    }
  }
}
template void cufinufft_plan_t<float>::setpts(
    int nj, const float *d_kx, const float *d_ky, const float *d_kz, int nk,
    const float *d_s, const float *d_t, const float *d_u);
template void cufinufft_plan_t<double>::setpts(
    int nj, const double *d_kx, const double *d_ky, const double *d_kz, int nk,
    const double *d_s, const double *d_t, const double *d_u);
