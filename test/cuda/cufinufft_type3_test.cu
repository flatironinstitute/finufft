#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <random>

#include <cufinufft.h>
#include <cufinufft/impl.h>
#include <cufinufft/utils.h>

#include <finufft/test_defs.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename T, typename V> bool equal(V *d_vec, T *cpu, const std::size_t size) {
  // copy d_vec to cpu
  thrust::host_vector<T> h_vec(size);
  // this implicitly converts cuda_complex to std::complex... which is fine, but it may
  // cause issues use it with case
  assert(cudaMemcpy(h_vec.data(), d_vec, size * sizeof(T), cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  for (std::size_t i = 0; i < size; ++i) {
    if (h_vec[i] != cpu[i]) {
      std::cout << " gpu[" << i << "]: " << h_vec[i] << " cpu[" << i << "]: " << cpu[i]
                << std::endl;
      return false;
    }
  }
  return true;
}

template<typename T>
T infnorm(std::complex<T> *a, std::complex<T> *b, const std::size_t n) {
  T err{0}, max_element{0};
  for (std::size_t m = 0; m < n; ++m) {
    //    std::cout << "a[" << m << "]: " << a[m] << " b[" << m << "]: " << b[m] << "\n";
    err         = std::max(err, std::abs(a[m] - b[m]));
    max_element = std::max(std::max(std::abs(a[m]), std::abs(b[m])), max_element);
  }
  return err / max_element;
}
// max error divide by max element
// max ( abs(a-b)) / max(abs(a))
// 10*(machine precision)
template<typename T>
T relerrtwonorm(std::complex<T> *a, std::complex<T> *b, const std::size_t n) {
  T err{0}, nrm{0};
  for (std::size_t m = 0; m < n; ++m) {
    //    std::cout << "a[" << m << "]: " << a[m] << " b[" << m << "]: " << b[m] << "\n";
    nrm += std::real(std::conj(a[m]) * a[m]);
    const auto diff = a[m] - b[m];
    auto this_err   = std::real(std::conj(diff) * diff);
    if (this_err > 1e-12) {
      std::cout << "a[" << m << "]: " << a[m] << " b[" << m << "]: " << b[m] << "\n";
      std::cout << "diff: " << diff << " this_err: " << this_err << std::endl;
    }
    err += this_err;
  }
  return std::sqrt(err / nrm);
}

template<typename T, typename V, typename contained = typename T::value_type>
auto almost_equal(V *d_vec, T *cpu, const std::size_t size,
                  const contained tol = std::numeric_limits<contained>::epsilon(),
                  bool print          = false) {
  // copy d_vec to cpu
  std::vector<T> h_vec(size);
  // this implicitly converts cuda_complex to std::complex... which is fine, but it may
  // cause issues use it with case
  assert(cudaMemcpy(h_vec.data(), d_vec, size * sizeof(T), cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  cudaDeviceSynchronize();
  // print h_vec and cpu
  if (print) {
    std::cout << std::setprecision(15);
    for (std::size_t i = 0; i < size; ++i) {
      std::cout << "gpu[" << i << "]: " << h_vec[i] << " cpu[" << i << "]: " << cpu[i]
                << '\n';
    }
    std::cout << std::setprecision(6);
  }
  const auto error = relerrtwonorm(h_vec.data(), cpu, size);
  std::cout << "relerrtwonorm: " << error << std::endl;
  // compare the l2 norm of the difference between the two vectors
  return (error < tol);
}

int main() {
  // for now, once finufft is demacroized we can test float
  using test_t = double;

  // defaults. tests should shadow them to override
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.debug           = 2;
  opts.upsampfac       = 2.00;
  opts.gpu_kerevalmeth = 0;
  opts.gpu_method      = 1;
  opts.gpu_sort        = 1;
  opts.modeord         = 0;
  finufft_opts fin_opts;
  finufft_default_opts(&fin_opts);
  fin_opts.debug              = opts.debug;
  fin_opts.spread_kerevalmeth = opts.gpu_kerevalmeth;
  fin_opts.upsampfac          = opts.upsampfac;
  fin_opts.spread_sort        = opts.gpu_sort;
  fin_opts.modeord            = opts.modeord;
  const int iflag             = 1;
  const int ntransf           = 1;
  const int dim               = 3;
  const double tol            = 1e-13;
  const int n_modes[]         = {5, 4, 2};
  const int N = n_modes[0] * (dim > 1 ? n_modes[1] : 1) * (dim > 2 ? n_modes[2] : 1);
  const int M = 15;
  const double bandwidth = 1.0;

  thrust::host_vector<test_t> x(M * ntransf), y(M * ntransf), z(M * ntransf),
      s(N * ntransf), t(N * ntransf), u(N * ntransf);
  thrust::host_vector<std::complex<test_t>> c(M * ntransf), fk(N * ntransf);

  thrust::device_vector<test_t> d_x(M * ntransf), d_y(M * ntransf), d_z(M * ntransf),
      d_s(N * ntransf), d_t(N * ntransf), d_u(N * ntransf);
  thrust::device_vector<std::complex<test_t>> d_c(M * ntransf), d_fk(N * ntransf);

  std::default_random_engine eng(42);
  std::uniform_real_distribution<test_t> dist11(-1, 1);
  auto rand_util_11 = [&eng, &dist11]() {
    return dist11(eng);
  };

  // Making data
  for (int64_t i = 0; i < M; i++) {
    x[i] = rand_util_11(); // x in [-pi,pi)
    y[i] = rand_util_11();
    z[i] = rand_util_11();
  }
  for (int64_t i = 0; i < N; i++) {
    s[i] = M_PI * rand_util_11() * bandwidth; // shifted so D1 is 8
    t[i] = M_PI * rand_util_11() * bandwidth; // shifted so D2 is 8
    u[i] = M_PI * rand_util_11() * bandwidth; // shifted so D3 is 8
  }

  const double deconv_tol = std::numeric_limits<double>::epsilon() * bandwidth * 1000;

  for (int64_t i = M; i < M * ntransf; ++i) {
    int64_t j = i % M;
    x[i]      = x[j];
    y[i]      = y[j];
    z[i]      = z[j];
  }
  for (int64_t i = N; i < N * ntransf; ++i) {
    int64_t j = i % N;
    s[i]      = s[j];
    t[i]      = t[j];
    u[i]      = u[j];
  }

  // copy x, y, z, s, t, u to device d_x, d_y, d_z, d_s, d_t, d_u
  d_x = x;
  d_y = y;
  d_z = z;
  d_s = s;
  d_t = t;
  d_u = u;
  cudaDeviceSynchronize();

  const auto cpu_planer = [iflag, tol, ntransf, dim, M, N, n_modes, &x, &y, &z, &s, &t,
                           &u, &fin_opts](const auto type) {
    finufft_plan_s *plan{nullptr};
    std::int64_t nl[] = {n_modes[0], n_modes[1], n_modes[2]};
    assert(finufft_makeplan(type, dim, nl, iflag, ntransf, tol, &plan, &fin_opts) == 0);
    assert(finufft_setpts(plan, M, x.data(), y.data(), z.data(), N, s.data(), t.data(),
                          u.data()) == 0);
    return plan;
  };

  const auto test_type1 = [iflag, tol, ntransf, dim, cpu_planer, M, N, n_modes, &d_x,
                           &d_y, &d_z, &c, &d_c, &fk, &d_fk, &opts,
                           &rand_util_11](auto plan) {
    // plan is a pointer to a type that contains real_t
    using T             = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type      = 1;
    const auto cpu_plan = cpu_planer(type);
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)n_modes, iflag, ntransf, T(tol),
                                      &plan, &opts) == 0);
    cudaDeviceSynchronize();
    assert(
        cufinufft_setpts_impl<T>(M, d_x.data().get(), d_y.data().get(), d_z.data().get(),
                                 0, nullptr, nullptr, nullptr, plan) == 0);
    cudaDeviceSynchronize();
    assert(plan->nf1 == cpu_plan->nf1);
    assert(plan->nf2 == cpu_plan->nf2);
    assert(plan->nf3 == cpu_plan->nf3);
    assert(plan->spopts.nspread == cpu_plan->spopts.nspread);
    assert(plan->spopts.upsampfac == cpu_plan->spopts.upsampfac);
    assert(plan->spopts.ES_beta == cpu_plan->spopts.ES_beta);
    assert(plan->spopts.ES_halfwidth == cpu_plan->spopts.ES_halfwidth);
    assert(plan->spopts.ES_c == cpu_plan->spopts.ES_c);

    for (int i = 0; i < M; i++) {
      c[i].real(rand_util_11());
      c[i].imag(rand_util_11());
    }
    d_c = c;
    cudaDeviceSynchronize();
    cufinufft_execute_impl((cuda_complex<T> *)d_c.data().get(),
                           (cuda_complex<T> *)d_fk.data().get(), plan);
    finufft_execute(cpu_plan, (std::complex<T> *)c.data(), (std::complex<T> *)fk.data());
    std::cout << "type " << type << ": ";
    assert(almost_equal(d_fk.data().get(), fk.data(), N, tol * 10));
    assert(cufinufft_destroy_impl<T>(plan) == 0);
    assert(finufft_destroy(cpu_plan) == 0);
    cudaDeviceSynchronize();
    plan = nullptr;
  };

  const auto test_type2 = [iflag, tol, ntransf, dim, cpu_planer, M, N, n_modes, &d_x,
                           &d_y, &d_z, &c, &d_c, &fk, &d_fk, &opts,
                           &rand_util_11](auto plan) {
    // plan is a pointer to a type that contains real_t
    using T             = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type      = 2;
    const auto cpu_plan = cpu_planer(type);
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)n_modes, iflag, ntransf, T(tol),
                                      &plan, &opts) == 0);
    cudaDeviceSynchronize();
    assert(
        cufinufft_setpts_impl<T>(M, d_x.data().get(), d_y.data().get(), d_z.data().get(),
                                 0, nullptr, nullptr, nullptr, plan) == 0);
    cudaDeviceSynchronize();
    assert(plan->nf1 == cpu_plan->nf1);
    assert(plan->nf2 == cpu_plan->nf2);
    assert(plan->nf3 == cpu_plan->nf3);
    assert(plan->spopts.nspread == cpu_plan->spopts.nspread);
    assert(plan->spopts.upsampfac == cpu_plan->spopts.upsampfac);
    assert(plan->spopts.ES_beta == cpu_plan->spopts.ES_beta);
    assert(plan->spopts.ES_halfwidth == cpu_plan->spopts.ES_halfwidth);
    assert(plan->spopts.ES_c == cpu_plan->spopts.ES_c);

    for (int i = 0; i < N; i++) {
      fk[i].real(rand_util_11());
      fk[i].imag(rand_util_11());
    }
    d_fk = fk;
    cudaDeviceSynchronize();
    cufinufft_execute_impl((cuda_complex<T> *)d_c.data().get(),
                           (cuda_complex<T> *)d_fk.data().get(), plan);
    finufft_execute(cpu_plan, c.data(), fk.data());
    cudaDeviceSynchronize();
    std::cout << "type " << type << ": ";
    assert(almost_equal(d_c.data().get(), c.data(), M, tol));
    assert(cufinufft_destroy_impl<T>(plan) == 0);
    assert(finufft_destroy(cpu_plan) == 0);
    cudaDeviceSynchronize();
    plan = nullptr;
  };

  auto test_type3 = [iflag, tol, ntransf, dim, cpu_planer, deconv_tol, M, N, n_modes,
                     &d_x, &d_y, &d_z, &d_s, &d_t, &d_u, &c, &d_c, &fk, &d_fk, &opts,
                     &rand_util_11](auto plan) {
    // plan is a pointer to a type that contains real_t
    using T             = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type      = 3;
    const auto cpu_plan = cpu_planer(type);
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)n_modes, iflag, ntransf, T(tol),
                                      &plan, &opts) == 0);
    cudaDeviceSynchronize();
    assert(cufinufft_setpts_impl<T>(M, d_x.data().get(), d_y.data().get(),
                                    d_z.data().get(), N, d_s.data().get(),
                                    d_t.data().get(), d_u.data().get(), plan) == 0);
    cudaDeviceSynchronize();
    assert(plan->type3_params.X1 == cpu_plan->t3P.X1);
    if (dim > 1) assert(plan->type3_params.X2 == cpu_plan->t3P.X2);
    if (dim > 2) assert(plan->type3_params.X3 == cpu_plan->t3P.X3);
    assert(plan->type3_params.C1 == cpu_plan->t3P.C1);
    if (dim > 1) assert(plan->type3_params.C2 == cpu_plan->t3P.C2);
    if (dim > 2) assert(plan->type3_params.C3 == cpu_plan->t3P.C3);
    assert(plan->type3_params.D1 == cpu_plan->t3P.D1);
    if (dim > 1) assert(plan->type3_params.D2 == cpu_plan->t3P.D2);
    if (dim > 2) assert(plan->type3_params.D3 == cpu_plan->t3P.D3);
    assert(plan->type3_params.gam1 == cpu_plan->t3P.gam1);
    if (dim > 1) assert(plan->type3_params.gam2 == cpu_plan->t3P.gam2);
    if (dim > 2) assert(plan->type3_params.gam3 == cpu_plan->t3P.gam3);
    assert(plan->type3_params.h1 == cpu_plan->t3P.h1);
    if (dim > 1) assert(plan->type3_params.h2 == cpu_plan->t3P.h2);
    if (dim > 2) assert(plan->type3_params.h3 == cpu_plan->t3P.h3);
    assert(plan->nf1 == cpu_plan->nf1);
    if (dim > 1) assert(plan->nf2 == cpu_plan->nf2);
    if (dim > 2) assert(plan->nf3 == cpu_plan->nf3);
    assert(equal(plan->kx, cpu_plan->X, M));
    if (dim > 1) assert(equal(plan->ky, cpu_plan->Y, M));
    if (dim > 2) assert(equal(plan->kz, cpu_plan->Z, M));
    assert(equal(plan->d_s, cpu_plan->Sp, N));
    if (dim > 1) assert(equal(plan->d_t, cpu_plan->Tp, N));
    if (dim > 2) assert(equal(plan->d_u, cpu_plan->Up, N));
    assert(plan->spopts.nspread == cpu_plan->spopts.nspread);
    assert(plan->spopts.upsampfac == cpu_plan->spopts.upsampfac);
    assert(plan->spopts.ES_beta == cpu_plan->spopts.ES_beta);
    assert(plan->spopts.ES_halfwidth == cpu_plan->spopts.ES_halfwidth);
    assert(plan->spopts.ES_c == cpu_plan->spopts.ES_c);
    std::cout << "prephase :\n";
    assert(almost_equal(plan->prephase, cpu_plan->prephase, M,
                        std::numeric_limits<T>::epsilon() * 100));
    std::cout << "deconv :\n";
    assert(almost_equal(plan->deconv, cpu_plan->deconv, N, deconv_tol));

    assert(plan->t2_plan->nf1 == cpu_plan->innerT2plan->nf1);
    if (dim > 1) assert(plan->t2_plan->nf2 == cpu_plan->innerT2plan->nf2);
    if (dim > 2) assert(plan->t2_plan->nf3 == cpu_plan->innerT2plan->nf3);
    assert(plan->t2_plan->nf == cpu_plan->innerT2plan->nf);

    assert(plan->t2_plan->spopts.nspread == cpu_plan->innerT2plan->spopts.nspread);
    assert(plan->t2_plan->spopts.upsampfac == cpu_plan->innerT2plan->spopts.upsampfac);
    assert(plan->t2_plan->spopts.ES_beta == cpu_plan->innerT2plan->spopts.ES_beta);
    assert(
        plan->t2_plan->spopts.ES_halfwidth == cpu_plan->innerT2plan->spopts.ES_halfwidth);
    assert(plan->t2_plan->spopts.ES_c == cpu_plan->innerT2plan->spopts.ES_c);
    assert(plan->t2_plan->ms == cpu_plan->innerT2plan->ms);
    assert(plan->t2_plan->mt == cpu_plan->innerT2plan->mt);
    assert(plan->t2_plan->mu == cpu_plan->innerT2plan->mu);
    int nf[]       = {plan->t2_plan->nf1, plan->t2_plan->nf2, plan->t2_plan->nf3};
    T *fwkerhalf[] = {plan->t2_plan->fwkerhalf1, plan->t2_plan->fwkerhalf2,
                      plan->t2_plan->fwkerhalf3};
    T *phiHat[]    = {cpu_plan->innerT2plan->phiHat1, cpu_plan->innerT2plan->phiHat2,
                      cpu_plan->innerT2plan->phiHat3};
    for (int idx = 0; idx < dim; ++idx) {
      std::cout << "nf[" << idx << "]: " << nf[idx] << std::endl;
      const auto size = (nf[idx] / 2 + 1);
      std::vector<T> fwkerhalf_host(size, -1);
      const auto ier = cudaMemcpy(fwkerhalf_host.data(), fwkerhalf[idx], size * sizeof(T),
                                  cudaMemcpyDeviceToHost);
      if (ier != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(ier) << std::endl;
      }
      assert(ier == cudaSuccess);
      cudaDeviceSynchronize();
      for (int i = 0; i < size; i++) {
        const auto error = abs(1 - fwkerhalf_host[i] / phiHat[idx][i]);
        if (error > tol) {
          std::cout << "fwkerhalf[" << idx << "][" << i << "]: " << fwkerhalf_host[i]
                    << " phiHat[" << idx << "][" << i << "]: " << phiHat[idx][i]
                    << std::endl;
          std::cout << "error: " << error << std::endl;
        }
        //        assert(error < tol * 1000);
      }
    }
    for (int i = 0; i < M; i++) {
      c[i].real(rand_util_11());
      c[i].imag(rand_util_11());
    }
    d_c = c;
    // for (int i = 0; i < N; i++) {
    // fk[i] = {randm11(), randm11()};
    // }
    // d_fk = fk;
    cufinufft_execute_impl((cuda_complex<T> *)d_c.data().get(),
                           (cuda_complex<T> *)d_fk.data().get(), plan);
    finufft_execute(cpu_plan, c.data(), fk.data());
    cudaDeviceSynchronize();
    std::cout << "t2_plan->fw : ";
    assert(almost_equal(plan->t2_plan->fw, cpu_plan->innerT2plan->fwBatch,
                        plan->t2_plan->nf, std::numeric_limits<T>::epsilon() * 100));
    std::cout << "CpBatch : ";
    assert(almost_equal(plan->c_batch, cpu_plan->CpBatch, M, tol, false));
    std::cout << "fw : ";
    assert(almost_equal(plan->fw, cpu_plan->fwBatch, plan->nf, tol * 10, false));
    std::cout << "fk : ";
    assert(almost_equal(d_fk.data().get(), fk.data(), N, tol * 10, false));
    assert(cufinufft_destroy_impl<T>(plan) == 0);
    assert(finufft_destroy(cpu_plan) == 0);
    plan = nullptr;
    cudaDeviceSynchronize();
  };
  // testing correctness of the plan creation
  //  cufinufft_plan_t<float> *single_plan{nullptr};
  cufinufft_plan_t<test_t> *double_plan{nullptr};
  test_type1(double_plan);
  test_type2(double_plan);
  test_type3(double_plan);
  return 0;
}
