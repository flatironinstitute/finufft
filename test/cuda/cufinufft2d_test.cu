#include <cmath>
#include <complex>
#include <cufinufft/contrib/helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include <cufinufft.h>

#include <cufinufft/impl.h>
#include <cufinufft/utils.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using cufinufft::utils::infnorm;

template<typename T>
int run_test(int method, int type, int N1, int N2, int M, T tol, T checktol, int iflag,
             double upsampfac) {
  std::cout << std::scientific << std::setprecision(3);

  thrust::host_vector<T> x(M), y(M);
  thrust::host_vector<thrust::complex<T>> c(M), fk(N1 * N2);

  thrust::device_vector<T> d_x(M), d_y(M);
  thrust::device_vector<thrust::complex<T>> d_c(M), d_fk(N1 * N2);

  std::default_random_engine eng(1);
  std::uniform_real_distribution<T> dist11(-1, 1);
  auto randm11 = [&eng, &dist11]() {
    return dist11(eng);
  };

  // Making data
  for (int i = 0; i < M; i++) {
    x[i] = M_PI * randm11(); // x in [-pi,pi)
    y[i] = M_PI * randm11();
  }
  if (type == 1) {
    for (int i = 0; i < M; i++) {
      c[i].real(randm11());
      c[i].imag(randm11());
    }
  } else if (type == 2) {
    for (int i = 0; i < N1 * N2; i++) {
      fk[i].real(randm11());
      fk[i].imag(randm11());
    }
  } else {
    std::cerr << "Invalid type " << type << " supplied\n";
    return 1;
  }

  d_x = x;
  d_y = y;
  if (type == 1)
    d_c = c;
  else if (type == 2)
    d_fk = fk;

  cudaEvent_t start, stop;
  float milliseconds = 0;
  float totaltime    = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warm up CUFFT (is slow, takes around 0.2 sec... )
  cudaEventRecord(start);
  {
    int nf1 = 1;
    cufftHandle fftplan;
    cufftPlan1d(&fftplan, nf1, cufft_type<T>(), 1);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] dummy warmup call to CUFFT\t %.3g s\n", milliseconds / 1000);

  // now to our tests...
  cufinufft_plan_t<T> *dplan;
  const int dim = 2;

  // Here we setup our own opts, for gpu_method.
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);

  opts.gpu_method       = method;
  opts.gpu_maxbatchsize = 1;
  opts.upsampfac        = upsampfac;
  int nmodes[3]         = {N1, N2, 1};
  int ntransf           = 1;
  cudaEventRecord(start);
  int ier =
      cufinufft_makeplan_impl(type, dim, nmodes, iflag, ntransf, tol, &dplan, &opts);
  if (ier != 0) {
    printf("err: cufinufft2d_plan\n");
    return ier;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  totaltime += milliseconds;
  printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds / 1000);

  cudaEventRecord(start);
  ier = cufinufft_setpts_impl<T>(M, d_x.data().get(), d_y.data().get(), nullptr, 0,
                                 nullptr, nullptr, nullptr, dplan);
  if (ier != 0) {
    printf("err: cufinufft_setpts\n");
    return ier;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  totaltime += milliseconds;
  printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds / 1000);

  cudaEventRecord(start);
  ier = cufinufft_execute_impl<T>((cuda_complex<T> *)d_c.data().get(),
                                  (cuda_complex<T> *)d_fk.data().get(), dplan);
  if (ier != 0) {
    printf("err: cufinufft2d1_exec\n");
    return ier;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  totaltime += milliseconds;
  float exec_ms = milliseconds;
  printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds / 1000);

  cudaEventRecord(start);
  ier = cufinufft_destroy_impl<T>(dplan);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  totaltime += milliseconds;
  printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds / 1000);

  if (type == 1)
    fk = d_fk;
  else if (type == 2)
    c = d_c;

  printf("[Method %d] %d NU pts to %d U pts in %.3g s:      %.3g NU pts/s\n",
         opts.gpu_method, M, N1 * N2, totaltime / 1000, M / totaltime * 1000);
  printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n", M / exec_ms * 1000);

  T rel_error = std::numeric_limits<T>::max();
  if (type == 1) {
    const int nt1         = 0.37 * N1;
    const int nt2         = 0.26 * N2; // choose some mode index to check
    thrust::complex<T> Ft = thrust::complex<T>(0, 0), J = thrust::complex<T>(0.0, iflag);
    for (int j = 0; j < M; ++j)
      Ft += c[j] * exp(J * (nt1 * x[j] + nt2 * y[j])); // crude direct
    const int it = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2); // index in complex F as 1d
                                                       // array

    rel_error = abs(Ft - fk[it]) / infnorm(N1, (std::complex<T> *)fk.data());
    printf("[gpu   ] one mode: rel err in F[%d,%d] is %.3g\n", nt1, nt2, rel_error);
  } else if (type == 2) {
    int jt                = M / 2; // check arbitrary choice of one targ pt
    thrust::complex<T> J  = thrust::complex<T>(0, iflag);
    thrust::complex<T> ct = thrust::complex<T>(0, 0);

    int m = 0;
    for (int m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order over F
      for (int m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
        ct += fk[m++] * exp(J * (m1 * x[jt] + m2 * y[jt])); // crude direct

    rel_error = abs(c[jt] - ct) / infnorm(M, (std::complex<T> *)c.data());
    printf("[gpu   ] one targ: rel err in c[%d] is %.3g\n", jt, rel_error);
  }

  return std::isnan(rel_error) || rel_error > checktol;
}

int main(int argc, char *argv[]) {
  if (argc != 10) {
    fprintf(stderr, "Usage: cufinufft2d1_test method N1 N2 M tol checktol\n"
                    "Arguments:\n"
                    "  method: One of\n"
                    "    1: nupts driven,\n"
                    "    2: sub-problem, or\n"
                    "  type: Type of transform (1, 2)"
                    "  N1, N2: The size of the 2D array\n"
                    "  M: The number of non-uniform points\n"
                    "  tol: NUFFT tolerance\n"
                    "  checktol: relative error to pass test\n"
                    "  prec:  'f' or 'd' (float/double)\n"
                    "  upsampfac: upsampling factor\n");
    return 1;
  }
  const int method       = atoi(argv[1]);
  const int type         = atoi(argv[2]);
  const int N1           = atof(argv[3]);
  const int N2           = atof(argv[4]);
  const int M            = atof(argv[5]);
  const double tol       = atof(argv[6]);
  const double checktol  = atof(argv[7]);
  const char prec        = argv[8][0];
  const double upsampfac = atof(argv[9]);
  const int iflag        = 1;

  if (prec == 'f')
    return run_test<float>(method, type, N1, N2, M, tol, checktol, iflag, upsampfac);
  else if (prec == 'd')
    return run_test<double>(method, type, N1, N2, M, tol, checktol, iflag, upsampfac);
  else
    return -1;
}
