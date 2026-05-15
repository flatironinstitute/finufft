// "execute" stage: cufinufft_plan_t<T>::execute and the type-1/2/3 drivers,
// plus the deconvolve kernel that only the execute path uses.
// Mirrors CPU src/execute.cpp.

#include <iostream>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

#include <cufinufft.h>
#include <cufinufft/spreadinterp.hpp>
#include <cufinufft/types.hpp>
#include <cufinufft/utils.hpp>

#include <finufft_common/constants.h>
#include <finufft_errors.h>
#include <thrust/device_vector.h>

/* Kernel for copying fw to fk with amplication by prefac/ker */
// Note: assume modeord=0: CMCL-compatible mode ordering in fk (from -N/2 up
// to N/2-1), modeord=1: FFT-compatible mode ordering in fk (from 0 to N/2-1, then -N/2 up
// to -1).
template<typename T, int modeord, int ndim>
static __global__ void deconv_nd(
    cuda::std::array<int, 3> mstu, cuda::std::array<int, 3> nf123, cuda_complex<T> *fw,
    cuda_complex<T> *fk, cuda::std::array<const T *, 3> fwkerhalf, bool fw2fk) {

  cuda::std::array<int, 3> m_acc{1, mstu[0], mstu[0] * mstu[1]};
  int mtotal = m_acc[ndim - 1] * mstu[ndim - 1];
  cuda::std::array<int, 3> nf_acc{1, nf123[0], nf123[0] * nf123[1]};

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < mtotal;
       i += blockDim.x * gridDim.x) {
    cuda::std::array<int, 3> k;
    if constexpr (ndim == 1) k = {i, 0, 0};
    if constexpr (ndim == 2) k = {i % mstu[0], i / mstu[0], 0};
    if constexpr (ndim == 3)
      k = {i % mstu[0], (i / mstu[0]) % mstu[1], (i / mstu[0]) / mstu[1]};
    T kervalue = 1;
    int fkidx = 0, fwidx = 0;
    for (int idim = 0; idim < ndim; ++idim) {
      int wn, fwkerindn;
      if constexpr (modeord == 0) {
        int pivot = k[idim] - mstu[idim] / 2;
        wn        = (pivot >= 0) ? pivot : nf123[idim] + pivot;
        fwkerindn = abs(pivot);
      } else {
        int pivot = k[idim] - mstu[idim] + mstu[idim] / 2;
        wn        = (pivot >= 0) ? nf123[idim] + k[idim] - mstu[idim] : k[idim];
        fwkerindn = (pivot >= 0) ? mstu[idim] - k[idim] : k[idim];
      }
      kervalue *= fwkerhalf[idim][fwkerindn];
      fwidx += wn * nf_acc[idim];
      fkidx += k[idim] * m_acc[idim];
    }

    if (fw2fk) {
      fk[fkidx] = fw[fwidx] / kervalue;
    } else {
      fw[fwidx] = fk[fkidx] / kervalue;
    }
  }
}

template<typename T>
template<int modeord, int ndim>
void cufinufft_plan_t<T>::deconvolve_nd<modeord, ndim>(
    cuda_complex<T> *fw, cuda_complex<T> *fk, int blksize) const
/*
    wrapper for deconvolution & amplification in 1/2/3D.

    Melody Shih 11/21/21
*/
{
  int nmodes = 1, nftot = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    nmodes *= mstu[idim];
    nftot *= nf123[idim];
  }

  bool fw2fk = spopts.spread_direction == 1;
  if (!fw2fk)
    checkCudaErrors(
        cudaMemsetAsync(fw, 0, batchsize * nftot * sizeof(cuda_complex<T>), stream));

  for (int t = 0; t < blksize; t++)
    deconv_nd<T, modeord, ndim><<<(nmodes + 256 - 1) / 256, 256, 0, stream>>>(
        mstu, nf123, fw + t * nftot, fk + t * nmodes, dethrust(fwkerhalf), fw2fk);
}

template<typename T>
void cufinufft_plan_t<T>::deconvolve(cuda_complex<T> *fw, cuda_complex<T> *fk,
                                     int blksize) const {
  if (dim == 1)
    (opts.modeord == 0) ? deconvolve_nd<0, 1>(fw, fk, blksize)
                        : deconvolve_nd<1, 1>(fw, fk, blksize);
  if (dim == 2)
    (opts.modeord == 0) ? deconvolve_nd<0, 2>(fw, fk, blksize)
                        : deconvolve_nd<1, 2>(fw, fk, blksize);
  if (dim == 3)
    (opts.modeord == 0) ? deconvolve_nd<0, 3>(fw, fk, blksize)
                        : deconvolve_nd<1, 3>(fw, fk, blksize);
}

template<typename T>
void cufinufft_plan_t<T>::execute_type1(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const
/*
    1D/2D/3D Type-1 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: spread data to oversampled regular mesh using kernel
        Step 2: compute FFT on uniform mesh
        Step 3: deconvolve by division of each Fourier mode independently by the
                Fourier series coefficient of the kernel.

    Melody Shih 07/25/19
*/
{
  assert(spopts.spread_direction == 1);

  int nmodes = 1;
  for (int idim = 0; idim < dim; ++idim) nmodes *= mstu[idim];
  // We don't need this buffer if we are just spreading; so we set
  // its size to 0 in that case.
  gpu_array<cuda_complex<T>> fwp(opts.gpu_spreadinterponly ? 0 : nf * batchsize, alloc);
  auto *fw = dethrust(fwp);
  for (int i = 0; i * batchsize < ntransf; i++) {
    int blksize   = std::min(ntransf - i * batchsize, batchsize);
    const auto *c = d_c + i * batchsize * M;
    auto *fk      = d_fk + i * batchsize * nmodes; // so deconvolve will write into
                                                   // user output f
    if (opts.gpu_spreadinterponly)
      fw = fk;                                     // spread directly into the appropriate
                                                   // section of the user output f

    checkCudaErrors(
        cudaMemsetAsync(fw, 0, blksize * nf * sizeof(cuda_complex<T>), stream));

    // Step 1: Spread
    spreadSorted(c, fw, blksize);

    if (opts.gpu_spreadinterponly) continue; // skip steps 2 and 3

    // Step 2: FFT
    cufftResult cufft_status = cufft_ex(fftplan.get(), fw, fw, iflag);
    if (cufft_status != CUFFT_SUCCESS) throw finufft::exception(FINUFFT_ERR_CUDA_FAILURE);

    // Step 3: deconvolve and shuffle
    deconvolve(fw, fk, blksize);
  }
}

template<typename T>
void cufinufft_plan_t<T>::execute_type2(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
                                        std::optional<int> ntransf_override) const
/*
    1D/2D/3D Type-2 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
        Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel
                Fourier coeff
        Step 2: compute FFT on uniform mesh
        Step 3: interpolate data to regular mesh

    Melody Shih 07/25/19
*/
{
  assert(spopts.spread_direction == 2);
  // CAUTION: if this particular execute_type2() call is executed as part of
  // a type 3 transform, ntransf will be overridden!
  int ntransf_for_this_run = ntransf;
  if (ntransf_override) ntransf_for_this_run = *ntransf_override;

  int nmodes = 1;
  for (int idim = 0; idim < dim; ++idim) nmodes *= mstu[idim];
  // We don't need this buffer if we are just interpolating; so we set
  // its size to 0 in that case.
  gpu_array<cuda_complex<T>> fwp(opts.gpu_spreadinterponly ? 0 : nf * batchsize, alloc);
  auto *fw = dethrust(fwp);
  for (int i = 0; i * batchsize < ntransf_for_this_run; i++) {
    int blksize = std::min(ntransf_for_this_run - i * batchsize, batchsize);
    auto *c     = d_c + i * batchsize * M;
    auto *fk    = d_fk + i * batchsize * nmodes;

    // Skip steps 1 and 2 if interponly
    if (!opts.gpu_spreadinterponly) {
      // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
      deconvolve(fw, fk, blksize);

      // Step 2: FFT
      THROW_IF_CUDA_ERROR
      cufftResult cufft_status = cufft_ex(fftplan.get(), fw, fw, iflag);
      if (cufft_status != CUFFT_SUCCESS)
        throw finufft::exception(FINUFFT_ERR_CUDA_FAILURE);
    } else
      fw = fk; // interpolate directly from user input f

    // Step 3: Interpolate
    interpSorted(c, fw, blksize);
  }
}

// TODO: in case data is centered, we could save GPU memory
template<typename T>
void cufinufft_plan_t<T>::execute_type3(cuda_complex<T> *d_c,
                                        cuda_complex<T> *d_fk) const {
  /*
    1D/2D/3D Type-3 NUFFT

  This function is called in "exec" stage (See ../cufinufft.cu).
  It includes (copied from doc in finufft library)
    Step 0: pre-phase the input strengths
    Step 1: spread data
    Step 2: Type 2 NUFFT
    Step 3: deconvolve (amplify) each Fourier mode, using kernel Fourier coeff

  Marco Barbone 08/14/2024
  */
  gpu_array<cuda_complex<T>> CpBatch(M * batchsize, alloc);
  gpu_array<cuda_complex<T>> fwp(nf * batchsize, alloc);
  auto *fw = dethrust(fwp);
  for (int i = 0; i * batchsize < ntransf; i++) {
    int blksize                = std::min(ntransf - i * batchsize, batchsize);
    cuda_complex<T> *d_cstart  = d_c + i * batchsize * M;
    cuda_complex<T> *d_fkstart = d_fk + i * batchsize * N;
    // setting input for spreader
    auto *c = dethrust(CpBatch);
    // setting output for spreader
    auto *fk = fw;
    // NOTE: fw might need to be set to 0
    checkCudaErrors(
        cudaMemsetAsync(fw, 0, blksize * nf * sizeof(cuda_complex<T>), stream));
    // Step 0: pre-phase the input strengths
    for (int block = 0; block < blksize; block++) {
      thrust::transform(thrust::cuda::par.on(stream), dethrust(prephase),
                        dethrust(prephase) + M, d_cstart + block * M, c + block * M,
                        thrust::multiplies<cuda_complex<T>>());
    }
    // Step 1: Spread
    spreadSorted(c, fk, blksize);
    // now fk = fw contains the spread values
    // Step 2: Type 2 NUFFT
    // type 2 goes from fk to c
    // saving the results directly in the user output array d_fk
    // it needs to do blksize transforms
    t2_plan->execute_type2(d_fkstart, fw, blksize);
    // Step 3: deconvolve
    // now we need to d_fk = d_fk*deconv
    for (int j = 0; j < blksize; j++) {
      thrust::transform(thrust::cuda::par.on(stream), dethrust(deconv),
                        dethrust(deconv) + N, d_fkstart + j * N, d_fkstart + j * N,
                        thrust::multiplies<cuda_complex<T>>());
    }
  }
}

template<typename T>
void cufinufft_plan_t<T>::execute(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const {
  DeviceSwitcher switcher(opts.gpu_device_id);
  switch (type) {
  case 1:
    return execute_type1(d_c, d_fk);
  case 2:
    return execute_type2(d_c, d_fk);
  case 3:
    return execute_type3(d_c, d_fk);
  }
}
template void cufinufft_plan_t<float>::execute(cuda_complex<float> *d_c,
                                               cuda_complex<float> *d_fk) const;
template void cufinufft_plan_t<double>::execute(cuda_complex<double> *d_c,
                                                cuda_complex<double> *d_fk) const;
