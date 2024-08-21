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
    err += std::real(std::conj(diff) * diff);
  }
  return std::sqrt(err / nrm);
}

template<typename T, typename V, typename contained = typename T::value_type>
auto almost_equal(V *d_vec,
                  T *cpu,
                  const std::size_t size,
                  const contained tol = std::numeric_limits<contained>::epsilon()) {
  // copy d_vec to cpu
  std::vector<T> h_vec(size);
  // this implicitly converts cuda_complex to std::complex... which is fine, but it may
  // cause issues use it with case
  assert(cudaMemcpy(h_vec.data(), d_vec, size * sizeof(T), cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  // print h_vec and cpu
  // for (std::size_t i = 0; i < size; ++i) {
  // std::cout << "gpu[" << i << "]: " << h_vec[i] << " cpu[" << i << "]: " << cpu[i]
  // << '\n';
  // }
  std::cout << "relerrtwonorm: " << infnorm(h_vec.data(), cpu, size) << std::endl;
  // compare the l2 norm of the difference between the two vectors
  if (relerrtwonorm(h_vec.data(), cpu, size) < tol) {
    return true;
  }
  return false;
}

int main() {
  // for now, once finufft is demacroized we can test float
  using test_t = double;

  // defaults. tests should shadow them to override
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.debug           = 2;
  opts.upsampfac       = 1.25;
  opts.gpu_kerevalmeth = 1;
  // opts.gpu_sort = 0;
  finufft_opts fin_opts;
  finufft_default_opts(&fin_opts);
  fin_opts.debug              = 2;
  fin_opts.spread_kerevalmeth = 1;
  fin_opts.upsampfac          = 1.25;
  const int iflag             = 1;
  const int ntransf           = 1;
  const int dim               = 3;
  const double tol            = 1e-9;
  const int n_modes[]         = {10, 5, 3};
  const int N                 = n_modes[0] * n_modes[1] * n_modes[2];
  const int M                 = 1000;
  const double bandwidth      = 50.0;

  thrust::host_vector<test_t> x(M * ntransf), y(M * ntransf), z(M * ntransf),
      s(N * ntransf), t(N * ntransf), u(N * ntransf);
  thrust::host_vector<std::complex<test_t>> c(M * ntransf), fk(N * ntransf);

  thrust::device_vector<test_t> d_x{}, d_y{}, d_z{}, d_s{}, d_t{}, d_u{};
  thrust::device_vector<std::complex<test_t>> d_c(M * ntransf), d_fk(N * ntransf);

  std::default_random_engine eng(42);
  std::uniform_real_distribution<test_t> dist11(-1, 1);
  auto rand_util_11 = [&eng, &dist11]() {
    return dist11(eng);
  };

  // Making data
  for (int64_t i = 0; i < M; i++) {
    x[i] = M_PI * rand_util_11() + 4; // x in [-pi,pi)
    y[i] = M_PI * rand_util_11() + 4;
    z[i] = M_PI * rand_util_11() + 4;
  }
  for (int64_t i = 0; i < N; i++) {
    s[i] = M_PI * rand_util_11() * bandwidth + 8; // shifted so D1 is 8
    t[i] = M_PI * rand_util_11() * bandwidth + 8; // shifted so D2 is 8
    u[i] = M_PI * rand_util_11() * bandwidth + 8; // shifted so D3 is 8
  }

  const double deconv_tol = std::numeric_limits<double>::epsilon() * bandwidth * 100;

  for (int64_t i = M; i < M * ntransf; ++i) {
    int64_t j = i % M;
    x[i]      = x[j];
    y[i]      = y[j];
    z[i]      = z[j];
  }
  for (int64_t i = M; i < N * ntransf; ++i) {
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

  const auto cpu_planer =
      [iflag, tol, ntransf, dim, M, N, n_modes, &x, &y, &z, &s, &t, &u, &fin_opts](
          const auto type) {
        finufft_plan_s *plan{nullptr};
        std::int64_t nl[] = {n_modes[0], n_modes[1], n_modes[2]};
        assert(
            finufft_makeplan(type, dim, nl, iflag, ntransf, tol, &plan, &fin_opts) == 0);
        assert(finufft_setpts(plan, M, x.data(), y.data(), z.data(), N, s.data(),
                              t.data(), u.data()) == 0);
        return plan;
      };

  const auto test_type1 = [iflag,
                           tol,
                           ntransf,
                           dim,
                           cpu_planer,
                           M,
                           N,
                           n_modes,
                           &d_x,
                           &d_y,
                           &d_z,
                           &c,
                           &d_c,
                           &fk,
                           &d_fk,
                           &opts](auto plan) {
    // plan is a pointer to a type that contains real_t
    using T             = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type      = 1;
    const auto cpu_plan = cpu_planer(type);
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)n_modes, iflag, ntransf, T(tol),
                                      &plan, &opts) == 0);
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
      c[i].real(randm11());
      c[i].imag(randm11());
    }
    d_c = c;
    cufinufft_execute_impl(
        (cuda_complex<T> *)d_c.data().get(), (cuda_complex<T> *)d_fk.data().get(), plan);
    finufft_execute(cpu_plan, (std::complex<T> *)c.data(), (std::complex<T> *)fk.data());
    std::cout << "type " << type << ": ";
    assert(almost_equal(d_fk.data().get(), fk.data(), N, tol));
    assert(cufinufft_destroy_impl<T>(plan) == 0);
    assert(finufft_destroy(cpu_plan) == 0);
    cudaDeviceSynchronize();
    plan = nullptr;
  };

  const auto test_type2 = [iflag,
                           tol,
                           ntransf,
                           dim,
                           cpu_planer,
                           M,
                           N,
                           n_modes,
                           &d_x,
                           &d_y,
                           &d_z,
                           &c,
                           &d_c,
                           &fk,
                           &d_fk,
                           &opts](auto plan) {
    // plan is a pointer to a type that contains real_t
    using T             = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type      = 2;
    const auto cpu_plan = cpu_planer(type);
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)n_modes, iflag, ntransf, T(tol),
                                      &plan, &opts) == 0);
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
      fk[i].real(randm11());
      fk[i].imag(randm11());
    }
    d_fk = fk;
    cufinufft_execute_impl(
        (cuda_complex<T> *)d_c.data().get(), (cuda_complex<T> *)d_fk.data().get(), plan);
    finufft_execute(cpu_plan, (std::complex<T> *)c.data(), (std::complex<T> *)fk.data());
    std::cout << "type " << type << ": ";
    assert(almost_equal(d_c.data().get(), c.data(), M, tol));
    assert(cufinufft_destroy_impl<T>(plan) == 0);
    assert(finufft_destroy(cpu_plan) == 0);
    cudaDeviceSynchronize();
    plan = nullptr;
  };

  auto test_type3 = [iflag,
                     tol,
                     ntransf,
                     dim,
                     cpu_planer,
                     deconv_tol,
                     M,
                     N,
                     n_modes,
                     &d_x,
                     &d_y,
                     &d_z,
                     &d_s,
                     &d_t,
                     &d_u,
                     &c,
                     &d_c,
                     &fk,
                     &d_fk,
                     &opts](auto plan) {
    // plan is a pointer to a type that contains real_t
    using T             = typename std::remove_pointer<decltype(plan)>::type::real_t;
    const int type      = 3;
    const auto cpu_plan = cpu_planer(type);
    assert(cufinufft_makeplan_impl<T>(type, dim, (int *)n_modes, iflag, ntransf, T(tol),
                                      &plan, &opts) == 0);
    assert(cufinufft_setpts_impl<T>(M, d_x.data().get(), d_y.data().get(),
                                    d_z.data().get(), N, d_s.data().get(),
                                    d_t.data().get(), d_u.data().get(), plan) == 0);
    cudaDeviceSynchronize();
    assert(plan->type3_params.X1 == cpu_plan->t3P.X1);
    assert(plan->type3_params.X2 == cpu_plan->t3P.X2);
    assert(plan->type3_params.X3 == cpu_plan->t3P.X3);
    assert(plan->type3_params.C1 == cpu_plan->t3P.C1);
    assert(plan->type3_params.C2 == cpu_plan->t3P.C2);
    assert(plan->type3_params.C3 == cpu_plan->t3P.C3);
    assert(plan->type3_params.D1 == cpu_plan->t3P.D1);
    assert(plan->type3_params.D2 == cpu_plan->t3P.D2);
    assert(plan->type3_params.D3 == cpu_plan->t3P.D3);
    assert(plan->type3_params.gam1 == cpu_plan->t3P.gam1);
    assert(plan->type3_params.gam2 == cpu_plan->t3P.gam2);
    assert(plan->type3_params.gam3 == cpu_plan->t3P.gam3);
    assert(plan->nf1 == cpu_plan->nf1);
    assert(plan->nf2 == cpu_plan->nf2);
    assert(plan->nf3 == cpu_plan->nf3);
    assert(equal(plan->kx, cpu_plan->X, M));
    assert(equal(plan->ky, cpu_plan->Y, M));
    assert(equal(plan->kz, cpu_plan->Z, M));
    assert(equal(plan->d_s, cpu_plan->Sp, N));
    assert(equal(plan->d_t, cpu_plan->Tp, N));
    assert(equal(plan->d_u, cpu_plan->Up, N));
    assert(plan->spopts.nspread == cpu_plan->spopts.nspread);
    assert(plan->spopts.upsampfac == cpu_plan->spopts.upsampfac);
    assert(plan->spopts.ES_beta == cpu_plan->spopts.ES_beta);
    assert(plan->spopts.ES_halfwidth == cpu_plan->spopts.ES_halfwidth);
    assert(plan->spopts.ES_c == cpu_plan->spopts.ES_c);
    std::cout << "prephase :\n";
    assert(almost_equal(
        plan->prephase, cpu_plan->prephase, M, std::numeric_limits<T>::epsilon() * 100));
    std::cout << "deconv :\n";
    assert(almost_equal(plan->deconv, cpu_plan->deconv, N, deconv_tol));

    for (int i = 0; i < M; i++) {
      c[i].real(randm11());
      c[i].imag(randm11());
    }
    d_c = c;
    // for (int i = 0; i < N; i++) {
    // fk[i] = {randm11(), randm11()};
    // }
    // d_fk = fk;
    cudaDeviceSynchronize();
    cufinufft_execute_impl(
        (cuda_complex<T> *)d_c.data().get(), (cuda_complex<T> *)d_fk.data().get(), plan);
    finufft_execute(cpu_plan, (std::complex<T> *)c.data(), (std::complex<T> *)fk.data());
    assert(almost_equal(d_fk.data().get(), fk.data(), N, tol));
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
