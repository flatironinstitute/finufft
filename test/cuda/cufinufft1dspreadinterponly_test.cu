#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>

#include <cufinufft.h>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/impl.h>
#include <cufinufft/utils.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using cufinufft::utils::infnorm;

template<typename T>
int run_test(int N1, int M, T tol, T checktol, int iflag, double upsampfac) {
  // tol and upsamplefac are used to determine the kernel

  std::cout << std::scientific << std::setprecision(3);
  int ier{};

  const int dim = 1;

  // Here we setup our own opts, for gpu_method.
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);

  // opts.gpu_method       = method;
  opts.gpu_maxbatchsize     = 1;
  opts.gpu_spreadinterponly = 1;
  opts.upsampfac            = upsampfac;

  int ntransf   = 1;
  int nmodes[3] = {N1, 1, 1};

  cufinufft_plan_t<T> *dplan;

  thrust::host_vector<T> x(M);
  thrust::host_vector<thrust::complex<T>> c(M);
  thrust::host_vector<thrust::complex<T>> fk(M);

  thrust::device_vector<T> d_x(M);
  thrust::device_vector<thrust::complex<T>> d_c(M);
  thrust::device_vector<thrust::complex<T>> d_fk(M);

  x[0] = 0.0;
  c[0] = 1.0;
  d_x  = x;
  d_c  = c;

  cudaEvent_t start, stop;
  float milliseconds = 0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  ier = cufinufft_makeplan_impl<T>(1, dim, nmodes, iflag, ntransf, tol, &dplan, &opts);
  if (ier != 0) {
    printf("err: cufinufft1d_plan (ier=%d)\n", ier);
    return ier;
  }
  ier = cufinufft_setpts_impl<T>(M, d_x.data().get(), nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr, dplan);
  if (ier != 0) {
    printf("err: cufinufft_setpts (ier=%d)\n", ier);
    return ier;
  }
  ier = cufinufft_execute_impl<T>((cuda_complex<T> *)d_c.data().get(),
                                  (cuda_complex<T> *)d_fk.data().get(), dplan);
  if (ier != 0) {
    printf("err: cufinufft1d_exec (ier=%d)\n", ier);
    return ier;
  }
  cufinufft_destroy_impl(dplan);

  fk = d_fk;
  const auto kersum =
      std::accumulate(fk.begin(), fk.end(), thrust::complex<T>(T(0), T(0)));

  // making data
  std::default_random_engine eng(1);
  std::uniform_real_distribution<T> dist11(-1, 1);
  auto randm11 = [&eng, &dist11]() {
    return dist11(eng);
  };

  // Making data
  for (int i = 0; i < M; i++) {
    x[i] = M_PI * randm11(); // x in [-pi,pi)
  }

  for (int i = 0; i < M; i++) {
    c[i].real(randm11());
    c[i].imag(randm11());
  }

  d_x = x;
  d_c = c;

  printf("spread-only test 1d:\n"); // ............................................

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  ier = cufinufft_makeplan_impl<T>(1, dim, nmodes, iflag, ntransf, tol, &dplan, &opts);
  if (ier != 0) {
    printf("err: cufinufft1d_plan (ier=%d)\n", ier);
    return ier;
  }
  ier = cufinufft_setpts_impl<T>(M, d_x.data().get(), nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr, dplan);
  if (ier != 0) {
    printf("err: cufinufft_setpts (ier=%d)\n", ier);
    return ier;
  }
  ier = cufinufft_execute_impl<T>((cuda_complex<T> *)d_c.data().get(),
                                  (cuda_complex<T> *)d_fk.data().get(), dplan);

  if (ier != 0) {
    printf("err: cufinufft1d_exec (ier=%d)\n", ier);
    return ier;
  }
  cufinufft_destroy_impl(dplan);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("\t%lld NU pts spread to %lld grid in %.3g s \t%.3g NU pts/s\n",
         static_cast<long long>(M), static_cast<long long>(N1), milliseconds / 1000,
         T(M) / (milliseconds / 1000));

  fk = d_fk;

  auto csum       = std::accumulate(c.begin(), c.end(), thrust::complex<T>(T(0), T(0)));
  const auto mass = std::accumulate(fk.begin(), fk.end(), thrust::complex<T>(T(0), T(0)));
  const auto rel_mass_err = thrust::abs(mass - kersum * csum) / thrust::abs(mass);
  printf("\trel mass err %.3g\n", rel_mass_err);

  printf("interp-only test 1d:\n"); // ............................................

  std::fill(fk.begin(), fk.end(), thrust::complex<T>(T(1), T(0)));
  d_fk = fk;

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  ier = cufinufft_makeplan_impl<T>(2, dim, nmodes, iflag, ntransf, tol, &dplan, &opts);
  if (ier != 0) {
    printf("err: cufinufft1d_plan (ier=%d)\n", ier);
    return ier;
  }
  ier = cufinufft_setpts_impl<T>(M, d_x.data().get(), nullptr, nullptr, 0, nullptr,
                                 nullptr, nullptr, dplan);
  if (ier != 0) {
    printf("err: cufinufft_setpts (ier=%d)\n", ier);
    return ier;
  }

  ier = cufinufft_execute_impl<T>((cuda_complex<T> *)d_c.data().get(),
                                  (cuda_complex<T> *)d_fk.data().get(), dplan);

  if (ier != 0) {
    printf("err: cufinufft1d_exec (ier=%d)\n", ier);
    return ier;
  }
  cufinufft_destroy_impl(dplan);

  cudaEventSynchronize(stop);
  cudaEventRecord(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("\t%lld NU pts interp from %lld grid in %.3g s \t%.3g NU pts/s\n",
         static_cast<long long>(M), static_cast<long long>(N1), milliseconds / 1000,
         T(M) / (milliseconds / 1000));

  c = d_c;

  csum         = std::accumulate(c.begin(), c.end(), thrust::complex<T>(T(0), T(0)));
  auto sup_err = T(0.0);
  for (auto cj : c) sup_err = std::max(sup_err, abs(cj - kersum));
  const auto rel_sup_err = sup_err / thrust::abs(kersum);
  printf("\trel sup err %.3g\n", rel_sup_err);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const auto rel_error = std::max(rel_mass_err, rel_sup_err);
  return std::isnan(rel_error) || rel_error > checktol;
}

int main(int argc, char *argv[]) {
  if (argc != 7) {
    fprintf(stderr,
            "Usage: cufinufft1dspreadinterponly_test N1 M tol checktol prec upsampfac\n"
            "Arguments:\n"
            "  N1: Number of fourier modes\n"
            "  M: The number of non-uniform points\n"
            "  tol: NUFFT tolerance\n"
            "  checktol:  relative error to pass test\n"
            "  precision: f or d\n"
            "  upsampfac: upsampling factor\n");
    return 1;
  }
  const int N1           = atof(argv[1]);
  const int M            = atof(argv[2]);
  const double tol       = atof(argv[3]);
  const double checktol  = atof(argv[4]);
  const int iflag        = 1;
  const char prec        = argv[5][0];
  const double upsampfac = atof(argv[6]);
  if (prec == 'f')
    return run_test<float>(N1, M, tol, checktol, iflag, upsampfac);
  else if (prec == 'd')
    return run_test<double>(N1, M, tol, checktol, iflag, upsampfac);
  else
    return -1;
}
