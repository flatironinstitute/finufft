namespace SCTL_NAMESPACE {

  template <class Real> SDC<Real>::SDC(const Integer Order_) : Order(Order_) {
    // TODO: do this in a numerically stable way
    #ifdef SCTL_QUAD_T
    using ValueType = QuadReal;
    #else
    using ValueType = long double;
    #endif
    const ValueType eps = machine_eps<ValueType>();

    Vector<ValueType> x_cheb(Order);
    for (Long i = 0; i < Order; i++) {
      x_cheb[i] = 0.5 - 0.5 * cos(const_pi<ValueType>() * i / (Order - 1));
    }

    Matrix<ValueType> Mp(Order, Order);
    Matrix<ValueType> Mi(Order, Order);
    for (Long i = 0; i < Order; i++) {
      for (Long j = 0; j < Order; j++) {
        Mp[j][i] = pow<ValueType>(x_cheb[i],j);
        Mi[j][i] = pow<ValueType>(x_cheb[i],j+1) / (j+1);
      }
    }
    auto M_time_step_ = (Mp.pinv(eps) * Mi).Transpose(); // TODO: replace Mp.pinv()

    Mp.ReInit(Order,Order); Mp = 0;
    Mi.ReInit(Order,Order); Mi = 0;
    Integer TRUNC_Order = Order;
    if (Order >= 2) TRUNC_Order = Order - 1;
    if (Order >= 6) TRUNC_Order = Order - 1;
    if (Order >= 9) TRUNC_Order = Order - 1;
    for (Long j = 0; j < TRUNC_Order; j++) {
      for (Long i = 0; i < Order; i++) {
        Mp[j][i] = pow<ValueType>(x_cheb[i],j);
        Mi[j][i] = pow<ValueType>(x_cheb[i],j);
      }
    }
    auto M_error_ = (Mp.pinv(eps) * Mi).Transpose();
    for (Long i = 0; i < Order; i++) M_error_[i][i] -= 1;

    M_time_step.ReInit(M_time_step_.Dim(0), M_time_step_.Dim(1));
    for (Long i = 0; i < M_time_step.Dim(0)*M_time_step.Dim(1); i++) M_time_step[0][i] = (Real)M_time_step_[0][i];

    M_error.ReInit(M_error_.Dim(0), M_error_.Dim(1));
    for (Long i = 0; i < M_error.Dim(0)*M_error.Dim(1); i++) M_error[0][i] = (Real)M_error_[0][i];
  }

  // solve u = \int_0^{dt} F(u)
  template <class Real> void SDC<Real>::operator()(Vector<Real>* u, const Real dt, const Vector<Real>& u0_, const Fn& F, Integer N_picard, Real tol_picard, Real* error_interp, Real* error_picard) const {
    auto max_norm = [] (const Matrix<Real>& M) {
      Real max_val = 0;
      for (Long i = 0; i < M.Dim(0); i++) {
        for (Long j = 0; j < M.Dim(1); j++) {
          max_val = std::max<Real>(max_val, fabs(M[i][j]));
        }
      }
      return max_val;
    };
    if (N_picard < 0) N_picard = Order;

    const Long DOF = u0_.Dim();
    Matrix<Real> Mu0(Order, DOF);
    Matrix<Real> Mu1(Order, DOF);
    for (Long j = 0; j < Order; j++) { // Set u0
      for (Long k = 0; k < DOF; k++) {
        Mu0[j][k] = u0_[k];
      }
    }

    Matrix<Real> M_dudt(Order, DOF);
    { // Set M_dudt
      Vector<Real> dudt_(DOF, M_dudt[0], false);
      F(&dudt_, Vector<Real>(DOF, Mu0[0], false));
      for (Long i = 1; i < Order; i++) {
        for (Long j = 0; j < DOF; j++) {
          M_dudt[i][j] = M_dudt[0][j];
        }
      }
    }
    Mu1 = Mu0 + (M_time_step * M_dudt) * dt;

    Matrix<Real> Merr(Order, DOF);
    for (Long k = 0; k < N_picard; k++) { // Picard iteration
      auto Mu_previous = Mu1;
      for (Long i = 1; i < Order; i++) { // Set M_dudt
        Vector<Real> dudt_(DOF, M_dudt[i], false);
        F(&dudt_, Vector<Real>(DOF, Mu1[i], false));
      }
      Mu1 = Mu0 + (M_time_step * M_dudt) * dt;
      Merr = Mu1 - Mu_previous;
      if (max_norm(Merr) < tol_picard) break;
    }

    if (u->Dim() != DOF) u->ReInit(DOF);
    for (Long k = 0; k < DOF; k++) { // Set u
      u[0][k] = Mu1[Order - 1][k];
    }
    if (error_picard != nullptr) {
      error_picard[0] = max_norm(Merr);
    }
    if (error_interp != nullptr) {
      Merr = M_error * Mu1;
      error_interp[0] = max_norm(Merr);
    }
  }

  template <class Real> Real SDC<Real>::AdaptiveSolve(Vector<Real>* u, Real dt, const Real T, const Vector<Real>& u0, const Fn& F, Real tol) const {
    const Real eps = machine_eps<Real>();
    Vector<Real> u_, u0_ = u0;

    Real t = 0;
    while (t < T && dt > eps*T) {
      Real error_interp, error_picard;
      (*this)(&u_, dt, u0_, F, -1, tol*dt*0.1, &error_interp, &error_picard);
      Real max_err = std::max<Real>(error_interp, error_picard);

      if (max_err < tol*dt) { // Accept solution
        u0_.Swap(u_);
        t = t + dt;
      }

      // Adjust time-step size (Quaife, Biros - JCP 2016)
      dt = std::min<Real>(T-t, 0.9*dt*pow<Real>((tol*dt)/max_err, 1/(Real)(Order)));

      //std::cout<<t<<' '<<dt<<' '<<max_err<<'\n';
    }
    if (t < T) SCTL_WARN("Could not solve ODE to the requested tolerance.");

    (*u) = u0_;
    return t;
  }

}

