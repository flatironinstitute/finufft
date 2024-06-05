/* This is an example of performing 2d1many
   in single precision.
*/

#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

#include <cufinufft.h>
#include <cufinufft/utils.h>

#include <cuda_runtime.h>
// FIXME: This isn't actually public, though maybe it should be?
using cufinufft::utils::infnorm;

int main(int argc, char *argv[])
/*
 * example code for 2D Type 1 transformation.
 *
 * To compile the code:
 * nvcc example2d1many.cpp -o example2d1many -I/loc/to/cufinufft/include
 * /loc/to/cufinufft/lib-static/libcufinufft.a -lcudart -lcufft -lnvToolsExt
 *
 * or
 * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/loc/to/cufinufft/lib
 * nvcc example2d1many.cpp -o example2d1many -I/loc/to/cufinufft/include
 * -L/loc/to/cufinufft/lib/ -lcufinufft
 *
 *
 */
{
  std::cout << std::scientific << std::setprecision(3);

  int ier;
  int N1      = 256;
  int N2      = 256;
  int M       = 65536;
  int ntransf = 2;
  int iflag   = 1;
  float tol   = 1e-6;

  float *x, *y;
  std::complex<float> *c, *fk;
  cudaMallocHost(&x, M * sizeof(float));
  cudaMallocHost(&y, M * sizeof(float));
  cudaMallocHost(&c, M * ntransf * sizeof(std::complex<float>));
  cudaMallocHost(&fk, N1 * N2 * ntransf * sizeof(std::complex<float>));

  float *d_x, *d_y;
  cuFloatComplex *d_c, *d_fk;
  cudaMalloc(&d_x, M * sizeof(float));
  cudaMalloc(&d_y, M * sizeof(float));
  cudaMalloc(&d_c, M * ntransf * sizeof(cuFloatComplex));
  cudaMalloc(&d_fk, N1 * N2 * ntransf * sizeof(cuFloatComplex));

  std::default_random_engine eng(1);
  std::uniform_real_distribution<float> distr(-1, 1);

  for (int i = 0; i < M; i++) {
    x[i] = M_PI * distr(eng);
    y[i] = M_PI * distr(eng);
  }

  for (int i = 0; i < M * ntransf; i++) {
    c[i].real(distr(eng));
    c[i].imag(distr(eng));
  }
  cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * ntransf * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

  cufinufftf_plan dplan;

  int dim = 2;
  int64_t nmodes[3];
  int type = 1;

  nmodes[0] = N1;
  nmodes[1] = N2;
  nmodes[2] = 1;

  ier = cufinufftf_makeplan(type, dim, nmodes, iflag, ntransf, tol, &dplan, NULL);

  ier = cufinufftf_setpts(dplan, M, d_x, d_y, NULL, 0, NULL, NULL, NULL);

  ier = cufinufftf_execute(dplan, d_c, d_fk);

  ier = cufinufftf_destroy(dplan);

  cudaMemcpy(fk, d_fk, N1 * N2 * ntransf * sizeof(cuFloatComplex),
             cudaMemcpyDeviceToHost);

  std::cout << std::endl << "Accuracy check:" << std::endl;
  int N = N1 * N2;
  for (int i = 0; i < ntransf; i += 1) {
    int nt1 = (int)(0.37 * N1), nt2 = (int)(0.26 * N2); // choose some mode index to check
    std::complex<float> Ft = std::complex<float>(0, 0),
                        J  = std::complex<float>(0, 1) * (float)iflag;
    for (CUFINUFFT_BIGINT j = 0; j < M; ++j)
      Ft += c[j + i * M] * exp(J * (nt1 * x[j] + nt2 * y[j])); // crude direct
    int it = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2); // index in complex F as 1d array
    printf("[gpu %3d] one mode: abs err in F[%d,%d] is %.3g\n", i, nt1, nt2,
           abs(Ft - fk[it + i * N]));
    printf("[gpu %3d] one mode: rel err in F[%d,%d] is %.3g\n", i, nt1, nt2,
           abs(Ft - fk[it + i * N]) / infnorm(N, fk + i * N));
  }

  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(c);
  cudaFreeHost(fk);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_c);
  cudaFree(d_fk);
  return 0;
}
