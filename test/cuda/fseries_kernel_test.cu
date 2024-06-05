#include <cmath>
#include <complex>
#include <cufinufft/contrib/helper_cuda.h>
#include <iomanip>
#include <iostream>

#include <cufinufft/common.h>
#include <cufinufft/defs.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

using namespace cufinufft::common;
using namespace cufinufft::spreadinterp;
using namespace cufinufft::utils;

template<typename T> int run_test(int nf1, int dim, T eps, int gpu, int nf2, int nf3) {

  finufft_spread_opts opts;
  T *fwkerhalf1, *fwkerhalf2, *fwkerhalf3;
  T *d_fwkerhalf1, *d_fwkerhalf2, *d_fwkerhalf3;
  checkCudaErrors(cudaMalloc(&d_fwkerhalf1, sizeof(T) * (nf1 / 2 + 1)));
  if (dim > 1) checkCudaErrors(cudaMalloc(&d_fwkerhalf2, sizeof(T) * (nf2 / 2 + 1)));
  if (dim > 2) checkCudaErrors(cudaMalloc(&d_fwkerhalf3, sizeof(T) * (nf3 / 2 + 1)));

  int ier = setup_spreader(opts, (T)eps, (T)2.0, 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float milliseconds = 0;
  float gputime      = 0;
  float cputime      = 0;

  CNTime timer;
  if (!gpu) {
    timer.start();
    fwkerhalf1 = (T *)malloc(sizeof(T) * (nf1 / 2 + 1));
    if (dim > 1) fwkerhalf2 = (T *)malloc(sizeof(T) * (nf2 / 2 + 1));
    if (dim > 2) fwkerhalf3 = (T *)malloc(sizeof(T) * (nf3 / 2 + 1));

    onedim_fseries_kernel(nf1, fwkerhalf1, opts);
    if (dim > 1) onedim_fseries_kernel(nf2, fwkerhalf2, opts);
    if (dim > 2) onedim_fseries_kernel(nf3, fwkerhalf3, opts);
    cputime = timer.elapsedsec();
    cudaEventRecord(start);
    {
      checkCudaErrors(cudaMemcpy(d_fwkerhalf1, fwkerhalf1, sizeof(T) * (nf1 / 2 + 1),
                                 cudaMemcpyHostToDevice));
      if (dim > 1)
        checkCudaErrors(cudaMemcpy(d_fwkerhalf2, fwkerhalf2, sizeof(T) * (nf2 / 2 + 1),
                                   cudaMemcpyHostToDevice));
      if (dim > 2)
        checkCudaErrors(cudaMemcpy(d_fwkerhalf3, fwkerhalf3, sizeof(T) * (nf3 / 2 + 1),
                                   cudaMemcpyHostToDevice));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    gputime = milliseconds;
    printf("[time  ] dim=%d, nf1=%8d, ns=%2d, CPU: %6.2f ms\n", dim, nf1, opts.nspread,
           gputime + cputime * 1000);
    free(fwkerhalf1);
    if (dim > 1) free(fwkerhalf2);
    if (dim > 2) free(fwkerhalf3);
  } else {
    timer.start();
    std::complex<double> a[dim * MAX_NQUAD];
    T f[dim * MAX_NQUAD];
    onedim_fseries_kernel_precomp(nf1, f, a, opts);
    if (dim > 1) onedim_fseries_kernel_precomp(nf2, f + MAX_NQUAD, a + MAX_NQUAD, opts);
    if (dim > 2)
      onedim_fseries_kernel_precomp(nf3, f + 2 * MAX_NQUAD, a + 2 * MAX_NQUAD, opts);
    cputime = timer.elapsedsec();

    cuDoubleComplex *d_a;
    T *d_f;
    cudaEventRecord(start);
    {
      checkCudaErrors(cudaMalloc(&d_a, dim * MAX_NQUAD * sizeof(cuDoubleComplex)));
      checkCudaErrors(cudaMalloc(&d_f, dim * MAX_NQUAD * sizeof(T)));
      checkCudaErrors(cudaMemcpy(d_a, a, dim * MAX_NQUAD * sizeof(cuDoubleComplex),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(
          cudaMemcpy(d_f, f, dim * MAX_NQUAD * sizeof(T), cudaMemcpyHostToDevice));
      ier =
          cufserieskernelcompute(dim, nf1, nf2, nf3, d_f, d_a, d_fwkerhalf1, d_fwkerhalf2,
                                 d_fwkerhalf3, opts.nspread, cudaStreamDefault);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    gputime = milliseconds;
    printf("[time  ] dim=%d, nf1=%8d, ns=%2d, GPU: %6.2f ms\n", dim, nf1, opts.nspread,
           gputime + cputime * 1000);
    cudaFree(d_a);
    cudaFree(d_f);
  }

  fwkerhalf1 = (T *)malloc(sizeof(T) * (nf1 / 2 + 1));
  if (dim > 1) fwkerhalf2 = (T *)malloc(sizeof(T) * (nf2 / 2 + 1));
  if (dim > 2) fwkerhalf3 = (T *)malloc(sizeof(T) * (nf3 / 2 + 1));

  checkCudaErrors(cudaMemcpy(fwkerhalf1, d_fwkerhalf1, sizeof(T) * (nf1 / 2 + 1),
                             cudaMemcpyDeviceToHost));
  if (dim > 1)
    checkCudaErrors(cudaMemcpy(fwkerhalf2, d_fwkerhalf2, sizeof(T) * (nf2 / 2 + 1),
                               cudaMemcpyDeviceToHost));
  if (dim > 2)
    checkCudaErrors(cudaMemcpy(fwkerhalf3, d_fwkerhalf3, sizeof(T) * (nf3 / 2 + 1),
                               cudaMemcpyDeviceToHost));
  for (int i = 0; i < nf1 / 2 + 1; i++) printf("%10.8e ", fwkerhalf1[i]);
  printf("\n");
  if (dim > 1)
    for (int i = 0; i < nf2 / 2 + 1; i++) printf("%10.8e ", fwkerhalf2[i]);
  printf("\n");
  if (dim > 2)
    for (int i = 0; i < nf3 / 2 + 1; i++) printf("%10.8e ", fwkerhalf3[i]);
  printf("\n");

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr,
            "Usage: onedim_fseries_kernel_test prec nf1 [dim [tol [gpuversion [nf2 "
            "[nf3]]]]]\n"
            "Arguments:\n"
            "  prec: 'f' or 'd' (float/double)\n"
            "  nf1: The size of the upsampled fine grid size in x.\n"
            "  dim: Dimension of the nuFFT.\n"
            "  tol: NUFFT tolerance (default 1e-6).\n"
            "  gpuversion: Use gpu version or not (default True).\n"
            "  nf2: The size of the upsampled fine grid size in y. (default nf1)\n"
            "  nf3: The size of the upsampled fine grid size in z. (default nf3)\n");
    return 1;
  }
  char prec  = argv[1][0];
  int nf1    = std::atof(argv[2]);
  int dim    = 1;
  double eps = 1e-6;
  int gpu    = 1;
  int nf2    = nf1;
  int nf3    = nf1;
  if (argc > 3) dim = std::atoi(argv[3]);
  if (argc > 4) eps = std::atof(argv[4]);
  if (argc > 5) gpu = std::atoi(argv[5]);
  if (argc > 6) nf2 = std::atoi(argv[6]);
  if (argc > 7) nf3 = std::atoi(argv[7]);

  if (prec == 'f')
    return run_test<float>(nf1, dim, eps, gpu, nf2, nf3);
  else if (prec == 'd')
    return run_test<double>(nf1, dim, eps, gpu, nf2, nf3);
  else
    return -1;
}
