/* This is an example of performing 2d3many
   in double precision.
*/

#include <complex>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>

#include <cufinufft.h>
#include <cufinufft/utils.h>

#include <cuda_runtime.h>

// FIXME: This isn't actually public, though maybe it should be?
using cufinufft::utils::infnorm;

int main(int argc, char *argv[])
/*
 * example code for 2D Type 3 transformation.
 *
 * To compile the code:
 * nvcc example2d3many.cpp -o example2d3many loc/to/cufinufft/lib-static/libcufinufft.a
 * -I/loc/to/cufinufft/include -lcudart -lcufft -lnvToolsExt
 *
 * or
 * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/loc/to/cufinufft/lib
 * nvcc example2d3many.cpp -example2d3many -L/loc/to/cufinufft/lib/
 * -I/loc/to/cufinufft/include -lcufinufft
 *
 *
 */
{
  std::cout << std::scientific << std::setprecision(3);

  int ier;
  int M            = 10;
  int N            = 20;
  int ntransf      = 4;
  int maxbatchsize = 4;
  int iflag        = 1;
  double tol       = 1e-6;

  double *x, *y, *s, *t;
  std::complex<double> *c, *fk;
  cudaMallocHost(&x, M * sizeof(double));
  cudaMallocHost(&y, M * sizeof(double));
  cudaMallocHost(&s, N * sizeof(double));
  cudaMallocHost(&t, N * sizeof(double));
  cudaMallocHost(&c, M * ntransf * sizeof(std::complex<double>));
  cudaMallocHost(&fk, N * ntransf * sizeof(std::complex<double>));

  double *d_x, *d_y, *d_s, *d_t;
  cuDoubleComplex *d_c, *d_fk;
  cudaMalloc(&d_x, M * sizeof(double));
  cudaMalloc(&d_y, M * sizeof(double));
  cudaMalloc(&d_c, M * ntransf * sizeof(cuDoubleComplex));
  cudaMalloc(&d_s, N * sizeof(double));
  cudaMalloc(&d_t, N * sizeof(double));
  cudaMalloc(&d_fk, N * ntransf * sizeof(cuDoubleComplex));

  std::default_random_engine eng(1);
  std::uniform_real_distribution<double> distr(-1, 1);

  for (int i = 0; i < M; i++) {
    x[i] = M_PI * distr(eng);
    y[i] = M_PI * distr(eng);
  }

  for (int i = 0; i < N; i++) {
    s[i] = distr(eng);
    t[i] = distr(eng);
  }

  for (int i = 0; i < M * ntransf; i++) {
    c[i].real(distr(eng));
    c[i].imag(distr(eng));
  }

  cudaMemcpy(d_x, x, M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s, s, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_t, t, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * ntransf * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  cufinufft_plan dplan;

  int dim           = 2;
  int64_t nmodes[3] = {N, 1, 1};
  int type          = 3;

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);

  ier = cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, &dplan, &opts);

  ier = cufinufft_setpts(dplan, M, d_x, d_y, NULL, N, d_s, d_t, NULL);

  ier = cufinufft_execute(dplan, d_c, d_fk);

  ier = cufinufft_destroy(dplan);

  cudaMemcpy(fk, d_fk, N * ntransf * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  std::cout << std::endl << "Accuracy check:" << std::endl;
  std::complex<double> *fkstart;
  std::complex<double> *cstart;
  for (int tr = 0; tr < ntransf; tr++) {
    fkstart = fk + tr * N;
    cstart  = c + tr * M;
    int jt  = N / 2; // check arbitrary choice of one targ pt
    std::complex<double> J(0, iflag * 1);
    std::complex<double> fkt(0, 0);
    for (int m = 0; m < M; m++) fkt += cstart[m] * exp(J * (s[jt] * x[m] + t[jt] * y[m]));

    printf("[gpu %3d] one targ: rel err in c[%d] is %.3g\n", tr, jt,
           abs(fkstart[jt] - fkt) / infnorm(N, fkstart));
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
