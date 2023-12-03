#ifndef _SCTL_ODE_SOLVER_
#define _SCTL_ODE_SOLVER_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(math_utils.hpp)

#include <functional>

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;
template <class ValueType> class Matrix;

template <class Real> class SDC {
  public:

    using Fn = std::function<void(Vector<Real>*, const Vector<Real>&)>;

    /**
     * Constructor
     *
     * @param[in] Order the order of the method.
     */
    SDC(const Integer Order);

    /**
     * Apply one step of spectral deferred correction (SDC).
     * Compute: u = \int_0^{dt} F(u)
     *
     * @param[out] u the solution
     * @param[in] dt the step size
     * @param[in] u0 the initial value
     * @param[in] F the function du/dt
     * @param[in] N_picard the maximum number of picard iterations
     * @param[in] tol_picard the tolerance for stopping picard iterations
     * @param[out] error_interp an estimate of the truncation error of the solution interpolant
     * @param[out] error_picard the picard iteration error
     */
    void operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0, const Fn& F, Integer N_picard = -1, Real tol_picard = 0, Real* error_interp = nullptr, Real* error_picard = nullptr) const;

    /**
     * Solve ODE adaptively to required tolerance.
     * Compute: u = \int_0^{T} F(u)
     *
     * @param[out] u the final solution
     * @param[in] dt the initial step size guess
     * @param[in] T the final time
     * @param[in] u0 the initial value
     * @param[in] F the function du/dt
     * @param[in] tol_picard the required solution tolerance
     *
     * @return the final time (should equal T if no errors)
     */
    Real AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, const Fn& F, Real tol) const;

    static void test_one_step(const Integer Order = 5) {
      auto ref_sol = [](Real t) { return cos<Real>(-t); };
      auto fn = [](Vector<Real>* dudt, const Vector<Real>& u) {
        (*dudt)[0] = -u[1];
        (*dudt)[1] = u[0];
      };
      std::function<void(Vector<Real>*, const Vector<Real>&)> F(fn);

      const SDC<Real> ode_solver(Order);
      Real t = 0.0, dt = 1.0e-1;
      Vector<Real> u, u0(2);
      u0[0] = 1.0;
      u0[1] = 0.0;
      while (t < 10.0) {
        Real error_interp, error_picard;
        ode_solver(&u, dt, u0, F, -1, 0.0, &error_interp, &error_picard);
        { // Accept solution
          u0 = u;
          t = t + dt;
        }

        printf("t = %e;  ", t);
        printf("u = %e;  ", u0[0]);
        printf("error = %e;  ", ref_sol(t) - u0[0]);
        printf("time_step_error_estimate = %e;  \n", std::max(error_interp, error_picard));
      }
    }

    static void test_adaptive_solve(const Integer Order = 5, const Real tol = 1e-5) {
      auto ref_sol = [](Real t) { return cos(-t); };
      auto fn = [](Vector<Real>* dudt, const Vector<Real>& u) {
        (*dudt)[0] = -u[1];
        (*dudt)[1] = u[0];
      };
      std::function<void(Vector<Real>*, const Vector<Real>&)> F(fn);

      Vector<Real> u, u0(2);
      u0[0] = 1.0; u0[1] = 0.0;
      Real T = 10.0, dt = 1.0e-1;

      SDC<Real> ode_solver(Order);
      Real t = ode_solver.AdaptiveSolve(&u, dt, T, u0, F, tol);

      if (t == T) {
        printf("u = %e;  ", u[0]);
        printf("error = %e;  \n", ref_sol(T) - u[0]);
      }
    }

  private:
    Matrix<Real> M_time_step, M_error;
    Integer Order;
};

}

#include SCTL_INCLUDE(ode-solver.txx)

#endif  //_SCTL_ODE_SOLVER_
