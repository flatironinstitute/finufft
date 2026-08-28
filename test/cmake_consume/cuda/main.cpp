// Minimal downstream consumer of an installed cufinufft.
//
// Including <cufinufft.h> pulls in <cufft.h> and the finufft_common headers, so
// a CUDA include directory missing from the interface of finufft::cufinufft is
// a hard compile error here. cudaMalloc and cufinufft1d1 then exercise the link
// interface: CUDA::cudart, CUDA::cufft and the library itself.
//
// The program needs a device. CI builds it and does not run it; a run without a
// device exits non-zero at the first CUDA call, so it cannot be mistaken for a
// pass.
#include <cufinufft.h>

#include <cuda_runtime.h>

#include <complex>
#include <cstdio>
#include <vector>

int main() {
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  std::printf("cufinufft_default_opts gpu_method %d\n", opts.gpu_method);

  const int64_t M = 8;
  const int64_t N = 16;
  std::vector<double> x(M, 0.0);
  std::vector<std::complex<double>> c(M, {1.0, 0.0});

  double *d_x               = nullptr;
  cuDoubleComplex *d_c      = nullptr;
  cuDoubleComplex *d_fk     = nullptr;
  const cudaError_t cuerr[] = {
      cudaMalloc(&d_x, M * sizeof(double)), cudaMalloc(&d_c, M * sizeof(cuDoubleComplex)),
      cudaMalloc(&d_fk, N * sizeof(cuDoubleComplex)),
      cudaMemcpy(d_x, x.data(), M * sizeof(double), cudaMemcpyHostToDevice),
      cudaMemcpy(d_c, c.data(), M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice)};
  for (const cudaError_t e : cuerr) {
    if (e != cudaSuccess) {
      std::printf("CUDA error: %s\n", cudaGetErrorString(e));
      return 1;
    }
  }

  const int ier = cufinufft1d1(M, d_x, d_c, +1, 1e-6, N, d_fk, &opts);
  if (ier != 0) {
    std::printf("cufinufft1d1 returned error %d\n", ier);
    return 1;
  }

  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_fk);
  std::printf("cufinufft consume OK\n");
  return 0;
}
