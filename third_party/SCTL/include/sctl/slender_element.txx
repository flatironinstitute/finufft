#include SCTL_INCLUDE(kernel_functions.hpp)
#include SCTL_INCLUDE(tensor.hpp)
#include SCTL_INCLUDE(quadrule.hpp)
#include SCTL_INCLUDE(ompUtils.hpp)
#include SCTL_INCLUDE(profile.hpp)
#include SCTL_INCLUDE(legendre_rule.hpp)
#include SCTL_INCLUDE(fft_wrapper.hpp)
#include SCTL_INCLUDE(vtudata.hpp)

#include <functional>

namespace SCTL_NAMESPACE {

  template <class Real> void LagrangeInterp<Real>::Interpolate(Vector<Real>& wts, const Vector<Real>& src_nds, const Vector<Real>& trg_nds) {
    static constexpr Integer digits = (Integer)(TypeTraits<Real>::SigBits*0.3010299957);
    static constexpr Integer VecLen = DefaultVecLen<Real>();
    using VecType = Vec<Real, VecLen>;
    VecType vec_one((Real)1);

    if (1) {
      const Long Nsrc = src_nds.Dim();
      const Long Ntrg = trg_nds.Dim();
      const Long Ntrg_ = (Ntrg/VecLen)*VecLen;
      if (wts.Dim() != Nsrc*Ntrg) wts.ReInit(Nsrc*Ntrg);

      Matrix<Real> M(Nsrc, Ntrg, wts.begin(), false);
      for (Long i1 = 0; i1 < Ntrg_; i1+=VecLen) {
        VecType x = VecType::Load(&trg_nds[i1]);
        for (Integer j = 0; j < Nsrc; j++) {
          VecType y(vec_one);
          VecType src_nds_j(src_nds[j]);
          for (Integer k = 0; k < j; k++) {
            VecType src_nds_k(src_nds[k]);
            VecType src_nds_kj = src_nds_k - src_nds_j;
            VecType src_nds_kj_inv = approx_rsqrt<digits>(src_nds_kj*src_nds_kj);
            y *= (src_nds_k - x) * src_nds_kj * src_nds_kj_inv * src_nds_kj_inv;
          }
          for (Integer k = j+1; k < Nsrc; k++) {
            VecType src_nds_k(src_nds[k]);
            VecType src_nds_kj = src_nds_k - src_nds_j;
            VecType src_nds_kj_inv = approx_rsqrt<digits>(src_nds_kj*src_nds_kj);
            y *= (src_nds_k - x) * src_nds_kj * src_nds_kj_inv * src_nds_kj_inv;
          }
          y.Store(&M[j][i1]);
        }
      }
      for (Long i1 = Ntrg_; i1 < Ntrg; i1++) {
        Real x = trg_nds[i1];
        for (Integer j = 0; j < Nsrc; j++) {
          Real y = 1;
          for (Integer k = 0; k < j; k++) {
            y *= (src_nds[k] - x) / (src_nds[k] - src_nds[j]);
          }
          for (Integer k = j+1; k < Nsrc; k++) {
            y *= (src_nds[k] - x) / (src_nds[k] - src_nds[j]);
          }
          M[j][i1] = y;
        }
      }
    }
    if (0) { // Barycentric, numerically unstable (TODO: diagnose)
      Long Nsrc = src_nds.Dim();
      Long Ntrg = trg_nds.Dim();
      if (wts.Dim() != Nsrc*Ntrg) wts.ReInit(Nsrc*Ntrg);
      if (!wts.Dim()) return;
      for (Long t = 0; t < Ntrg; t++) {
        Real scal = 0;
        Long overlap = -1;
        for (Long s = 0; s < Nsrc; s++) {
          if (src_nds[s] == trg_nds[t]) overlap = s;
          scal += 1.0/(src_nds[s]-trg_nds[t]) * (s%2?1.0:-1.0) * ((s==0)||(s==Nsrc-1)?0.5:1.0);
        }
        scal = 1.0 / scal;

        if (overlap == -1) {
          for (Long s = 0; s < Nsrc; s++) {
            wts[s*Ntrg+t] = 1.0/(src_nds[s]-trg_nds[t]) * (s%2?1.0:-1.0) * ((s==0)||(s==Nsrc-1)?0.5:1.0) * scal;
          }
        } else {
          for (Long s = 0; s < Nsrc; s++) wts[s*Ntrg+t] = 0;
          wts[overlap*Ntrg+t] = 1;
        }
      }
    }
  }

  template <class Real> void LagrangeInterp<Real>::Derivative(Vector<Real>& df, const Vector<Real>& f, const Vector<Real>& nds) {
    Long N = nds.Dim();
    Long dof = f.Dim() / N;
    SCTL_ASSERT(f.Dim() == N * dof);
    if (df.Dim() != N * dof) df.ReInit(N * dof);
    if (N*dof == 0) return;

    auto dp = [&nds,&N](Real x, Long i) {
      Real scal = 1;
      for (Long j = 0; j < N; j++) {
        if (i!=j) scal *= (nds[i] - nds[j]);
      }
      scal = 1/scal;
      Real wt = 0;
      for (Long k = 0; k < N; k++) {
        Real wt_ = 1;
        if (k!=i) {
          for (Long j = 0; j < N; j++) {
            if (j!=k && j!=i) wt_ *= (x - nds[j]);
          }
          wt += wt_;
        }
      }
      return wt * scal;
    };
    for (Long k = 0; k < dof; k++) {
      for (Long i = 0; i < N; i++) {
        Real df_ = 0;
        for (Long j = 0; j < N; j++) {
          df_ += f[k*N+j] * dp(nds[i],j);
        }
        df[k*N+i] = df_;
      }
    }
  }

  template <class Real> void LagrangeInterp<Real>::test() { // TODO: cleanup
    Matrix<Real> f(1,3);
    f[0][0] = 0; f[0][1] = 1; f[0][2] = 0.5;

    Vector<Real> src, trg;
    for (Long i = 0; i < 3; i++) src.PushBack(i);
    for (Long i = 0; i < 11; i++) trg.PushBack(i*0.2);
    Vector<Real> wts;
    Interpolate(wts,src,trg);
    Matrix<Real> Mwts(src.Dim(), trg.Dim(), wts.begin(), false);
    Matrix<Real> ff = f * Mwts;
    std::cout<<ff<<'\n';

    Vector<Real> df;
    Derivative(df, Vector<Real>(f.Dim(0)*f.Dim(1),f.begin()), src);
    std::cout<<df<<'\n';
  }






  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::Setup(const Kernel& ker, Real R0) {
    #ifdef SCTL_QUAD_T
    using ValueType = QuadReal;
    #else
    using ValueType = long double;
    #endif
    PrecompToroidalGreensFn<ValueType>(ker, R0);
  }

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::BuildOperatorModal(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const {
    constexpr Integer KDIM0 = Kernel::SrcDim();
    constexpr Integer KDIM1 = Kernel::TrgDim();
    constexpr Integer Nmm = (Nm/2+1)*2;
    constexpr Integer Ntt = (Nt/2+1)*2;

    StaticArray<Real,2*Nr> buff0;
    StaticArray<Real,Ntt> buff1;
    Vector<Real> r_basis(Nr,buff0,false);
    Vector<Real> interp_r(Nr,buff0+Nr,false);
    Vector<Real> interp_Ntt(Ntt,buff1,false);
    if (M.Dim(0) != KDIM0*Nmm || M.Dim(1) != KDIM1) M.ReInit(KDIM0*Nmm,KDIM1);
    { // Set M
      const Real r = sqrt<Real>(x0*x0 + x1*x1);
      const Real rho = sqrt<Real>((r-R0_)*(r-R0_) + x2*x2);
      if (rho < max_dist*R0_) {
        const Real r_inv = 1/r;
        const Real rho_inv = 1/rho;
        const Real cos_theta = x0*r_inv;
        const Real sin_theta = x1*r_inv;
        const Real cos_phi = x2*rho_inv;
        const Real sin_phi = (r-R0_)*rho_inv;

        { // Set interp_r
          interp_r = 0;
          const Real rho0 = (rho/R0_-min_dist)/(max_dist-min_dist);
          BasisFn<Real>::EvalBasis(r_basis, rho0);
          for (Long i = 0; i < Nr; i++) {
            Real fn_val = 0;
            for (Long j = 0; j < Nr; j++) {
              fn_val += Mnds2coeff1[0][i*Nr+j] * r_basis[j];
            }
            for (Long j = 0; j < Nr; j++) {
              interp_r[j] += Mnds2coeff0[0][i*Nr+j] * fn_val;
            }
          }
        }
        { // Set interp_Ntt
          interp_Ntt[0] = 0.5;
          interp_Ntt[1] = 0.0;
          Complex<Real> exp_t(cos_phi, sin_phi);
          Complex<Real> exp_jt(cos_phi, sin_phi);
          for (Long j = 1; j < Ntt/2; j++) {
            interp_Ntt[j*2+0] = exp_jt.real;
            interp_Ntt[j*2+1] =-exp_jt.imag;
            exp_jt *= exp_t;
          }
        }

        M = 0;
        for (Long j = 0; j < Nr; j++) {
          for (Long k = 0; k < Ntt; k++) {
            Real interp_wt = interp_r[j] * interp_Ntt[k];
            ConstIterator<Real> Ut_ = Ut.begin() + (j*Ntt+k)*KDIM0*Nmm*KDIM1;
            for (Long i = 0; i < KDIM0*Nmm*KDIM1; i++) { // Set M
              M[0][i] += Ut_[i] * interp_wt;
            }
          }
        }
        { // Rotate by theta
          Complex<Real> exp_iktheta(1,0), exp_itheta(cos_theta, -sin_theta);
          for (Long k = 0; k < Nmm/2; k++) {
            for (Long i = 0; i < KDIM0; i++) {
              for (Long j = 0; j < KDIM1; j++) {
                Complex<Real> c(M[i*Nmm+2*k+0][j],M[i*Nmm+2*k+1][j]);
                c *= exp_iktheta;
                M[i*Nmm+2*k+0][j] = c.real;
                M[i*Nmm+2*k+1][j] = c.imag;
              }
            }
            exp_iktheta *= exp_itheta;
          }
        }
      } else if (rho < max_dist*R0_*1.25) {
        BuildOperatorModalDirect<110>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*1.67) {
        BuildOperatorModalDirect<88>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*2.5) {
        BuildOperatorModalDirect<76>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*5) {
        BuildOperatorModalDirect<50>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*10) {
        BuildOperatorModalDirect<25>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*20) {
        BuildOperatorModalDirect<14>(M, x0, x1, x2, ker);
      } else {
        BuildOperatorModalDirect<Nm>(M, x0, x1, x2, ker);
      }
    }
  }

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType> ValueType ToroidalGreensFn<Real,Nm,Nr,Nt>::BasisFn<ValueType>::Eval(const Vector<ValueType>& coeff, ValueType x) {
    if (1) {
      ValueType sum = 0;
      ValueType log_x = log(x);
      Long Nsplit = std::max<Long>(0,(coeff.Dim()-1)/2);
      ValueType x_i = 1;
      for (Long i = 0; i < Nsplit; i++) {
        sum += coeff[i] * x_i;
        x_i *= x;
      }
      x_i = 1;
      for (Long i = coeff.Dim()-2; i >= Nsplit; i--) {
        sum += coeff[i] * log_x * x_i;
        x_i *= x;
      }
      if (coeff.Dim()-1 >= 0) sum += coeff[coeff.Dim()-1] / x;
      return sum;
    }
    if (0) {
      ValueType sum = 0;
      Long Nsplit = coeff.Dim()/2;
      for (Long i = 0; i < Nsplit; i++) {
        sum += coeff[i] * sctl::pow<ValueType,Long>(x,i);
      }
      for (Long i = Nsplit; i < coeff.Dim(); i++) {
        sum += coeff[i] * log(x) * sctl::pow<ValueType,Long>(x,coeff.Dim()-1-i);
      }
      return sum;
    }
  }
  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType> void ToroidalGreensFn<Real,Nm,Nr,Nt>::BasisFn<ValueType>::EvalBasis(Vector<ValueType>& f, ValueType x) {
    const Long N = f.Dim();
    const Long Nsplit = std::max<Long>(0,(N-1)/2);

    ValueType xi = 1;
    for (Long i = 0; i < Nsplit; i++) {
      f[i] = xi;
      xi *= x;
    }

    ValueType xi_logx = log(x);
    for (Long i = N-2; i >= Nsplit; i--) {
      f[i] = xi_logx;
      xi_logx *= x;
    }

    if (N-1 >= 0) f[N-1] = 1/x;
  }
  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType> const Vector<ValueType>& ToroidalGreensFn<Real,Nm,Nr,Nt>::BasisFn<ValueType>::nds(Integer ORDER) {
    ValueType fn_start = 1e-7, fn_end = 1.0;
    auto compute_nds = [&ORDER,&fn_start,&fn_end]() {
      Vector<ValueType> nds, wts;
      auto integrands = [&ORDER,&fn_start,&fn_end](const Vector<ValueType>& nds) {
        const Integer K = ORDER;
        const Long N = nds.Dim();
        Matrix<ValueType> M(N,K);
        for (Long j = 0; j < N; j++) {
          Vector<ValueType> f(K,M[j],false);
          EvalBasis(f, nds[j]*(fn_end-fn_start)+fn_start);
        }
        return M;
      };
      InterpQuadRule<ValueType>::Build(nds, wts, integrands, sqrt(machine_eps<ValueType>()), ORDER);
      return nds*(fn_end-fn_start)+fn_start;
    };
    static Vector<ValueType> nds = compute_nds();
    return nds;
  }

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType, class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::PrecompToroidalGreensFn(const Kernel& ker, ValueType R0) {
    SCTL_ASSERT(ker.CoordDim() == COORD_DIM);
    constexpr Integer KDIM0 = Kernel::SrcDim();
    constexpr Integer KDIM1 = Kernel::TrgDim();
    constexpr Long Nmm = (Nm/2+1)*2;
    constexpr Long Ntt = (Nt/2+1)*2;
    R0_ = (Real)R0;

    const auto& nds = BasisFn<ValueType>::nds(Nr);
    { // Set Mnds2coeff0, Mnds2coeff1
      Matrix<ValueType> M(Nr,Nr);
      Vector<ValueType> coeff(Nr); coeff = 0;
      for (Long i = 0; i < Nr; i++) {
        coeff[i] = 1;
        for (Long j = 0; j < Nr; j++) {
          M[i][j] = BasisFn<ValueType>::Eval(coeff, nds[j]);
        }
        coeff[i] = 0;
      }

      Matrix<ValueType> U, S, Vt;
      M.SVD(U, S, Vt);
      for (Long i = 0; i < S.Dim(0); i++) {
        S[i][i] = 1/S[i][i];
      }
      auto Mnds2coeff0_ = S * Vt;
      auto Mnds2coeff1_ = U.Transpose();
      Mnds2coeff0.ReInit(Mnds2coeff0_.Dim(0), Mnds2coeff0_.Dim(1));
      Mnds2coeff1.ReInit(Mnds2coeff1_.Dim(0), Mnds2coeff1_.Dim(1));
      for (Long i = 0; i < Mnds2coeff0.Dim(0)*Mnds2coeff0.Dim(1); i++) Mnds2coeff0[0][i] = (Real)Mnds2coeff0_[0][i];
      for (Long i = 0; i < Mnds2coeff1.Dim(0)*Mnds2coeff1.Dim(1); i++) Mnds2coeff1[0][i] = (Real)Mnds2coeff1_[0][i];
    }
    { // Setup fft_Nm_R2C
      Vector<Long> dim_vec(1);
      dim_vec[0] = Nm;
      fft_Nm_R2C.Setup(FFT_Type::R2C, KDIM0, dim_vec);
      fft_Nm_C2R.Setup(FFT_Type::C2R, KDIM0*KDIM1, dim_vec);
    }

    Vector<ValueType> Xtrg(Nr*Nt*COORD_DIM);
    for (Long i = 0; i < Nr; i++) {
      for (Long j = 0; j < Nt; j++) {
        Xtrg[(i*Nt+j)*COORD_DIM+0] = R0 * (1.0 + (min_dist+(max_dist-min_dist)*nds[i]) * sin<ValueType>(j*2*const_pi<ValueType>()/Nt));
        Xtrg[(i*Nt+j)*COORD_DIM+1] = R0 * (0.0);
        Xtrg[(i*Nt+j)*COORD_DIM+2] = R0 * (0.0 + (min_dist+(max_dist-min_dist)*nds[i]) * cos<ValueType>(j*2*const_pi<ValueType>()/Nt));
      }
    }

    Vector<ValueType> U0(KDIM0*Nmm*Nr*KDIM1*Nt);
    { // Set U0
      FFT<ValueType> fft_Nm_C2R;
      { // Setup fft_Nm_C2R
        Vector<Long> dim_vec(1);
        dim_vec[0] = Nm;
        fft_Nm_C2R.Setup(FFT_Type::C2R, KDIM0, dim_vec);
      }
      Vector<ValueType> Fcoeff(KDIM0*Nmm), F, U_;
      for (Long i = 0; i < KDIM0*Nmm; i++) {
        Fcoeff = 0; Fcoeff[i] = 1;
        { // Set F
          fft_Nm_C2R.Execute(Fcoeff, F);
          Matrix<ValueType> FF(KDIM0,Nm,F.begin(), false);
          FF = FF.Transpose();
        }
        ComputePotential<ValueType>(U_, Xtrg, R0, F, ker);
        SCTL_ASSERT(U_.Dim() == Nr*Nt*KDIM1);

        for (Long j = 0; j < Nr; j++) {
          for (Long l = 0; l < Nt; l++) {
            for (Long k = 0; k < KDIM1; k++) {
              U0[((i*Nr+j)*KDIM1+k)*Nt+l] = U_[(j*Nt+l)*KDIM1+k];
            }
          }
        }
      }
    }

    Vector<ValueType> U1(KDIM0*Nmm*Nr*KDIM1*Ntt);
    { // U1 <-- fft_Nt(U0)
      FFT<ValueType> fft_Nt;
      Vector<Long> dim_vec(1); dim_vec = Nt;
      fft_Nt.Setup(FFT_Type::R2C, KDIM0*Nmm*Nr*KDIM1, dim_vec);
      fft_Nt.Execute(U0, U1);
      if (Nt%2==0 && Nt) {
        for (Long i = Ntt-2; i < U1.Dim(); i += Ntt) {
          U1[i] *= 0.5;
        }
      }
      U1 *= 1.0/sqrt<ValueType>(Nt);
    }

    U.ReInit(KDIM0*Nmm*KDIM1*Nr*Ntt);
    { // U <-- rearrange(U1)
      for (Long i0 = 0; i0 < KDIM0*Nmm; i0++) {
        for (Long i1 = 0; i1 < Nr; i1++) {
          for (Long i2 = 0; i2 < KDIM1; i2++) {
            for (Long i3 = 0; i3 < Ntt; i3++) {
              U[((i0*Nr+i1)*KDIM1+i2)*Ntt+i3] = (Real)U1[((i0*KDIM1+i2)*Nr+i1)*Ntt+i3];
            }
          }
        }
      }
    }

    Ut.ReInit(Nr*Ntt*KDIM0*Nmm*KDIM1);
    { // Set Ut
      Matrix<Real> Ut_(Nr*Ntt,KDIM0*Nmm*KDIM1, Ut.begin(), false);
      Matrix<Real> U_(KDIM0*Nmm*KDIM1,Nr*Ntt, U.begin(), false);
      Ut_ = U_.Transpose()*2.0;
    }
  }

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType, class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::ComputePotential(Vector<ValueType>& U, const Vector<ValueType>& Xtrg, ValueType R0, const Vector<ValueType>& F_, const Kernel& ker, ValueType tol) {
   constexpr Integer KDIM0 = Kernel::SrcDim();
    Vector<ValueType> F_fourier_coeff;
    const Long Nt_ = F_.Dim() / KDIM0; // number of Fourier modes
    SCTL_ASSERT(F_.Dim() == Nt_ * KDIM0);

    { // Transpose F_
      Matrix<ValueType> FF(Nt_,KDIM0,(Iterator<ValueType>)F_.begin(), false);
      FF = FF.Transpose();
    }
    { // Set F_fourier_coeff
      FFT<ValueType> fft_plan;
      Vector<Long> dim_vec(1); dim_vec[0] = Nt_;
      fft_plan.Setup(FFT_Type::R2C, KDIM0, dim_vec);
      fft_plan.Execute(F_, F_fourier_coeff);
      if (Nt_%2==0 && F_fourier_coeff.Dim()) {
        F_fourier_coeff[F_fourier_coeff.Dim()-2] *= 0.5;
      }
    }
    auto EvalFourierExp = [&Nt_](Vector<ValueType>& F, const Vector<ValueType>& F_fourier_coeff, Integer dof, const Vector<ValueType>& theta) {
      const Long N = F_fourier_coeff.Dim() / dof / 2;
      SCTL_ASSERT(F_fourier_coeff.Dim() == dof * N * 2);
      const Long Ntheta = theta.Dim();
      if (F.Dim() != Ntheta*dof) F.ReInit(Ntheta*dof);
      for (Integer k = 0; k < dof; k++) {
        for (Long j = 0; j < Ntheta; j++) {
          Complex<ValueType> F_(0,0);
          for (Long i = 0; i < N; i++) {
            Complex<ValueType> c(F_fourier_coeff[(k*N+i)*2+0],F_fourier_coeff[(k*N+i)*2+1]);
            Complex<ValueType> exp_t(cos<ValueType>(theta[j]*i), sin<ValueType>(theta[j]*i));
            F_ += exp_t * c * (i==0?1:2);
          }
          F[j*dof+k] = F_.real/sqrt<ValueType>(Nt_);
        }
      }
    };

    constexpr Integer QuadOrder = 18;
    std::function<Vector<ValueType>(ValueType,ValueType,ValueType)>  compute_potential = [&](ValueType a, ValueType b, ValueType tol) -> Vector<ValueType> {
      auto GetGeomCircle = [&R0] (Vector<ValueType>& Xsrc, Vector<ValueType>& Nsrc, const Vector<ValueType>& nds) {
        Long N = nds.Dim();
        if (Xsrc.Dim() != N * COORD_DIM) Xsrc.ReInit(N*COORD_DIM);
        if (Nsrc.Dim() != N * COORD_DIM) Nsrc.ReInit(N*COORD_DIM);
        for (Long i = 0; i < N; i++) {
          Xsrc[i*COORD_DIM+0] = R0 * cos<ValueType>(nds[i]);
          Xsrc[i*COORD_DIM+1] = R0 * sin<ValueType>(nds[i]);
          Xsrc[i*COORD_DIM+2] = R0 * 0;
          Nsrc[i*COORD_DIM+0] = cos<ValueType>(nds[i]);
          Nsrc[i*COORD_DIM+1] = sin<ValueType>(nds[i]);
          Nsrc[i*COORD_DIM+2] = 0;
        }
      };

      const auto& nds0 = ChebQuadRule<ValueType>::nds(QuadOrder+1);
      const auto& wts0 = ChebQuadRule<ValueType>::wts(QuadOrder+1);
      const auto& nds1 = ChebQuadRule<ValueType>::nds(QuadOrder+0);
      const auto& wts1 = ChebQuadRule<ValueType>::wts(QuadOrder+0);

      Vector<ValueType> U0;
      Vector<ValueType> Xsrc, Nsrc, Fsrc;
      GetGeomCircle(Xsrc, Nsrc, a+(b-a)*nds0);
      EvalFourierExp(Fsrc, F_fourier_coeff, KDIM0, a+(b-a)*nds0);
      for (Long i = 0; i < nds0.Dim(); i++) {
        for (Long j = 0; j < KDIM0; j++) {
          Fsrc[i*KDIM0+j] *= ((b-a) * wts0[i]);
        }
      }
      ker.Eval(U0, Xtrg, Xsrc, Nsrc, Fsrc);

      Vector<ValueType> U1;
      GetGeomCircle(Xsrc, Nsrc, a+(b-a)*nds1);
      EvalFourierExp(Fsrc, F_fourier_coeff, KDIM0, a+(b-a)*nds1);
      for (Long i = 0; i < nds1.Dim(); i++) {
        for (Long j = 0; j < KDIM0; j++) {
          Fsrc[i*KDIM0+j] *= ((b-a) * wts1[i]);
        }
      }
      ker.Eval(U1, Xtrg, Xsrc, Nsrc, Fsrc);

      ValueType err = 0, max_val = 0;
      for (Long i = 0; i < U1.Dim(); i++) {
        err = std::max<ValueType>(err, fabs(U0[i]-U1[i]));
        max_val = std::max<ValueType>(max_val, fabs(U0[i]));
      }
      if (err < tol || (b-a)<tol) {
      //if ((a != 0 && b != 2*const_pi<ValueType>()) || (b-a)<tol) {
        std::cout<<a<<' '<<b-a<<' '<<err<<' '<<tol<<'\n';
        return U1;
      } else {
        U0 = compute_potential(a, (a+b)*0.5, tol);
        U1 = compute_potential((a+b)*0.5, b, tol);
        return U0 + U1;
      }
    };
    U = compute_potential(0, 2*const_pi<ValueType>(), tol);
  };

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <Integer Nnds, class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::BuildOperatorModalDirect(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const {
    constexpr Integer KDIM0 = Kernel::SrcDim();
    constexpr Integer KDIM1 = Kernel::TrgDim();
    constexpr Integer Nmm = (Nm/2+1)*2;

    auto get_sin_theta = [](Long N){
      Vector<Real> sin_theta(N);
      for (Long i = 0; i < N; i++) {
        sin_theta[i] = sin<Real>(2*const_pi<Real>()*i/N);
      }
      return sin_theta;
    };
    auto get_cos_theta = [](Long N){
      Vector<Real> cos_theta(N);
      for (Long i = 0; i < N; i++) {
        cos_theta[i] = cos<Real>(2*const_pi<Real>()*i/N);
      }
      return cos_theta;
    };
    auto get_circle_coord = [](Long N, Real R0){
      Vector<Real> X(N*COORD_DIM);
      for (Long i = 0; i < N; i++) {
        X[i*COORD_DIM+0] = R0*cos<Real>(2*const_pi<Real>()*i/N);
        X[i*COORD_DIM+1] = R0*sin<Real>(2*const_pi<Real>()*i/N);
        X[i*COORD_DIM+2] = 0;
      }
      return X;
    };
    constexpr Real scal = 2/sqrt<Real>(Nm);

    static const Vector<Real> sin_nds = get_sin_theta(Nnds);
    static const Vector<Real> cos_nds = get_cos_theta(Nnds);
    static const Vector<Real> Xn = get_circle_coord(Nnds,1);

    StaticArray<Real,Nnds*COORD_DIM> buff0;
    Vector<Real> Xs(Nnds*COORD_DIM,buff0,false);
    Xs = Xn * R0_;

    StaticArray<Real,COORD_DIM> Xt = {x0,x1,x2};
    StaticArray<Real,KDIM0*KDIM1*Nnds> mem_buff2;
    Matrix<Real> Mker(KDIM0*Nnds, KDIM1, mem_buff2, false);
    ker.KernelMatrix(Mker, Vector<Real>(COORD_DIM,(Iterator<Real>)Xt,false), Xs, Xn);

    StaticArray<Real,4*Nnds> mem_buff3;
    Vector<Complex<Real>> exp_itheta(Nnds, (Iterator<Complex<Real>>)(mem_buff3+0*Nnds), false);
    Vector<Complex<Real>> exp_iktheta_da(Nnds, (Iterator<Complex<Real>>)(mem_buff3+2*Nnds), false);
    for (Integer j = 0; j < Nnds; j++) {
      exp_itheta[j].real = cos_nds[j];
      exp_itheta[j].imag =-sin_nds[j];
      exp_iktheta_da[j].real = 2*const_pi<Real>()/Nnds*scal;
      exp_iktheta_da[j].imag = 0;
    }
    for (Integer k = 0; k < Nmm/2; k++) { // apply Mker to complex exponentials
      // TODO: FFT might be faster since points are uniform
      Tensor<Real,true,KDIM0,KDIM1> Mk0, Mk1;
      for (Integer i0 = 0; i0 < KDIM0; i0++) {
        for (Integer i1 = 0; i1 < KDIM1; i1++) {
          Mk0(i0,i1) = 0;
          Mk1(i0,i1) = 0;
        }
      }
      for (Integer j = 0; j < Nnds; j++) {
        Tensor<Real,false,KDIM0,KDIM1> Mker_(Mker[j*KDIM0]);
        Mk0 = Mk0 + Mker_ * exp_iktheta_da[j].real;
        Mk1 = Mk1 + Mker_ * exp_iktheta_da[j].imag;
      }
      for (Integer i0 = 0; i0 < KDIM0; i0++) {
        for (Integer i1 = 0; i1 < KDIM1; i1++) {
          M[i0*Nmm+(k*2+0)][i1] = Mk0(i0,i1);
          M[i0*Nmm+(k*2+1)][i1] = Mk1(i0,i1);
        }
      }
      exp_iktheta_da *= exp_itheta;
    }
    for (Integer i0 = 0; i0 < KDIM0; i0++) {
      for (Integer i1 = 0; i1 < KDIM1; i1++) {
        M[i0*Nmm+0][i1] *= 0.5;
        M[i0*Nmm+1][i1] *= 0.5;
        if (Nm%2 == 0) {
          M[(i0+1)*Nmm-2][i1] *= 0.5;
          M[(i0+1)*Nmm-1][i1] *= 0.5;
        }
      }
    }
  }






  template <class ValueType> static void ReadFile(Vector<Vector<ValueType>>& data, const std::string fname) {
    FILE* f = fopen(fname.c_str(), "r");
    if (f == nullptr) {
      std::cout << "Unable to open file for reading:" << fname << '\n';
    } else {
      uint64_t data_len;
      Long readlen = fread(&data_len, sizeof(uint64_t), 1, f);
      SCTL_ASSERT(readlen == 1);
      if (data_len) {
        data.ReInit(data_len);
        for (Long i = 0; i < data.Dim(); i++) {
          readlen = fread(&data_len, sizeof(uint64_t), 1, f);
          SCTL_ASSERT(readlen == 1);
          data[i].ReInit(data_len);
          if (data_len) {
            readlen = fread(&data[i][0], sizeof(ValueType), data_len, f);
            SCTL_ASSERT(readlen == (Long)data_len);
          }
        }
      }
      fclose(f);
    }
  }
  template <class ValueType> static void WriteFile(const Vector<Vector<ValueType>>& data, const std::string fname) {
    FILE* f = fopen(fname.c_str(), "wb+");
    if (f == nullptr) {
      std::cout << "Unable to open file for writing:" << fname << '\n';
      exit(0);
    }
    uint64_t data_len = data.Dim();
    fwrite(&data_len, sizeof(uint64_t), 1, f);

    for (Integer i = 0; i < data.Dim(); i++) {
      data_len = data[i].Dim();
      fwrite(&data_len, sizeof(uint64_t), 1, f);
      if (data_len) fwrite(&data[i][0], sizeof(ValueType), data_len, f);
    }
    fclose(f);
  }

  template <class ValueType> static ValueType dot_prod(const Tensor<ValueType,true,3,1>& u, const Tensor<ValueType,true,3,1>& v) {
    ValueType u_dot_v = 0;
    u_dot_v += u(0,0) * v(0,0);
    u_dot_v += u(1,0) * v(1,0);
    u_dot_v += u(2,0) * v(2,0);
    return u_dot_v;
  }
  template <class ValueType> static Tensor<ValueType,true,3,1> cross_prod(const Tensor<ValueType,true,3,1>& u, const Tensor<ValueType,true,3,1>& v) {
    Tensor<ValueType,true,3,1> uxv;
    uxv(0,0) = u(1,0) * v(2,0) - u(2,0) * v(1,0);
    uxv(1,0) = u(2,0) * v(0,0) - u(0,0) * v(2,0);
    uxv(2,0) = u(0,0) * v(1,0) - u(1,0) * v(0,0);
    return uxv;
  }

  template <class Real> static const Vector<Real>& sin_theta(const Integer ORDER) {
    constexpr Integer MaxOrder = 256;
    auto compute_sin_theta = [MaxOrder](){
      Vector<Vector<Real>> sin_theta_lst(MaxOrder);
      for (Long k = 0; k < MaxOrder; k++) {
        sin_theta_lst[k].ReInit(k);
        for (Long i = 0; i < k; i++) {
          sin_theta_lst[k][i] = sin<Real>(2*const_pi<Real>()*i/k);
        }
      }
      return sin_theta_lst;
    };
    static const auto sin_theta_lst = compute_sin_theta();

    SCTL_ASSERT(ORDER < MaxOrder);
    return sin_theta_lst[ORDER];
  }
  template <class Real> static const Vector<Real>& cos_theta(const Integer ORDER) {
    constexpr Integer MaxOrder = 256;
    auto compute_cos_theta = [MaxOrder](){
      Vector<Vector<Real>> cos_theta_lst(MaxOrder);
      for (Long k = 0; k < MaxOrder; k++) {
        cos_theta_lst[k].ReInit(k);
        for (Long i = 0; i < k; i++) {
          cos_theta_lst[k][i] = cos<Real>(2*const_pi<Real>()*i/k);
        }
      }
      return cos_theta_lst;
    };
    static const auto cos_theta_lst = compute_cos_theta();

    SCTL_ASSERT(ORDER < MaxOrder);
    return cos_theta_lst[ORDER];
  }
  template <class Real> static const Matrix<Real>& fourier_matrix(Integer Nmodes, Integer Nnodes) {
    constexpr Integer MaxOrder = 128;
    auto compute_fourier_matrix = [](Integer Nmodes, Integer Nnodes) {
      if (Nnodes == 0 || Nmodes == 0) return Matrix<Real>();
      Matrix<Real> M_fourier(2*Nmodes,Nnodes);
      for (Long i = 0; i < Nnodes; i++) {
        Real theta = 2*const_pi<Real>()*i/Nnodes;
        for (Long k = 0; k < Nmodes; k++) {
          M_fourier[k*2+0][i] = cos<Real>(k*theta);
          M_fourier[k*2+1][i] = sin<Real>(k*theta);
        }
      }
      return M_fourier;
    };
    auto compute_all = [&compute_fourier_matrix, MaxOrder]() {
      Matrix<Matrix<Real>> Mall(MaxOrder, MaxOrder);
      for (Long i = 0; i < MaxOrder; i++) {
        for (Long j = 0; j < MaxOrder; j++) {
          Mall[i][j] = compute_fourier_matrix(i,j);
        }
      }
      return Mall;
    };
    static const Matrix<Matrix<Real>> Mall = compute_all();

    SCTL_ASSERT(Nmodes < MaxOrder && Nnodes < MaxOrder);
    return Mall[Nmodes][Nnodes];
  }
  template <class Real> static const Matrix<Real>& fourier_matrix_inv(Integer Nnodes, Integer Nmodes) {
    constexpr Integer MaxOrder = 128;
    auto compute_fourier_matrix_inv = [](Integer Nnodes, Integer Nmodes) {
      if (Nmodes > Nnodes/2+1 || Nnodes == 0 || Nmodes == 0) return Matrix<Real>();
      const Real scal = 2/(Real)Nnodes;

      Matrix<Real> M_fourier_inv(Nnodes,2*Nmodes);
      for (Long i = 0; i < Nnodes; i++) {
        Real theta = 2*const_pi<Real>()*i/Nnodes;
        for (Long k = 0; k < Nmodes; k++) {
          M_fourier_inv[i][k*2+0] = cos<Real>(k*theta)*scal;
          M_fourier_inv[i][k*2+1] = sin<Real>(k*theta)*scal;
        }
      }
      for (Long i = 0; i < Nnodes; i++) {
        M_fourier_inv[i][0] *= 0.5;
      }
      if (Nnodes == (Nmodes-1)*2) {
        for (Long i = 0; i < Nnodes; i++) {
          M_fourier_inv[i][Nnodes] *= 0.5;
        }
      }
      return M_fourier_inv;
    };
    auto compute_all = [&compute_fourier_matrix_inv, MaxOrder]() {
      Matrix<Matrix<Real>> Mall(MaxOrder, MaxOrder);
      for (Long i = 0; i < MaxOrder; i++) {
        for (Long j = 0; j < MaxOrder; j++) {
          Mall[i][j] = compute_fourier_matrix_inv(i,j);
        }
      }
      return Mall;
    };
    static const Matrix<Matrix<Real>> Mall = compute_all();

    SCTL_ASSERT(Nnodes < MaxOrder && Nmodes < MaxOrder);
    return Mall[Nnodes][Nmodes];
  }
  template <class Real> static const Matrix<Real>& fourier_matrix_inv_transpose(Integer Nnodes, Integer Nmodes) {
    constexpr Integer MaxOrder = 128;
    auto compute_all = [MaxOrder]() {
      Matrix<Matrix<Real>> Mall(MaxOrder, MaxOrder);
      for (Long i = 0; i < MaxOrder; i++) {
        for (Long j = 0; j < MaxOrder; j++) {
          Mall[i][j] = fourier_matrix_inv<Real>(i,j).Transpose();
        }
      }
      return Mall;
    };
    static const Matrix<Matrix<Real>> Mall = compute_all();

    SCTL_ASSERT(Nnodes < MaxOrder && Nmodes < MaxOrder);
    return Mall[Nnodes][Nmodes];
  }

  template <class ValueType> static const std::pair<Vector<ValueType>,Vector<ValueType>>& LegendreQuadRule(Integer ORDER) {
    constexpr Integer max_order = 50;
    auto compute_nds_wts = [max_order]() {
      Vector<std::pair<Vector<ValueType>,Vector<ValueType>>> nds_wts(max_order);
      for (Integer order = 1; order < max_order; order++) {
        auto& x_ = nds_wts[order].first;
        auto& w_ = nds_wts[order].second;
        x_ = LegQuadRule<ValueType>::ComputeNds(order);
        w_ = LegQuadRule<ValueType>::ComputeWts(x_);
      }
      return nds_wts;
    };
    static const auto nds_wts = compute_nds_wts();

    SCTL_ASSERT(ORDER < max_order);
    return nds_wts[ORDER];
  }
  template <class ValueType> static const std::pair<Vector<ValueType>,Vector<ValueType>>& LogSingularityQuadRule(Integer ORDER) {
    constexpr Integer MaxOrder = 50;
    auto compute_nds_wts_lst = [MaxOrder]() {
      #ifdef SCTL_QUAD_T
      using RealType = QuadReal;
      #else
      using RealType = long double;
      #endif
      Vector<Vector<RealType>> data;
      ReadFile<RealType>(data, "data/log_quad");
      if (data.Dim() < MaxOrder*2) {
        data.ReInit(MaxOrder*2);
        #pragma omp parallel for
        for (Integer order = 1; order < MaxOrder; order++) {
          auto integrands = [order](const Vector<RealType>& nds) {
            const Integer K = order;
            const Long N = nds.Dim();
            Matrix<RealType> M(N,K);
            for (Long j = 0; j < N; j++) {
              for (Long i = 0; i < (K+1)/2; i++) {
                M[j][i] = pow<RealType,Long>(nds[j],i);
              }
              for (Long i = (K+1)/2; i < K; i++) {
                M[j][i] = pow<RealType,Long>(nds[j],i-(K+1)/2) * log<RealType>(nds[j]);
              }
            }
            return M;
          };
          InterpQuadRule<RealType>::Build(data[order*2+0], data[order*2+1], integrands, false, 1e-20, order, 2e-4, 1.0); // TODO: diagnose accuracy issues
        }
        WriteFile<RealType>(data, "data/log_quad");
      }

      Vector<std::pair<Vector<ValueType>,Vector<ValueType>>> nds_wts_lst(MaxOrder);
      #pragma omp parallel for
      for (Integer order = 1; order < MaxOrder; order++) {
        const auto& nds = data[order*2+0];
        const auto& wts = data[order*2+1];
        auto& nds_ = nds_wts_lst[order].first;
        auto& wts_ = nds_wts_lst[order].second;
        nds_.ReInit(nds.Dim());
        wts_.ReInit(wts.Dim());
        for (Long i = 0; i < nds.Dim(); i++) {
          nds_[i] = (ValueType)nds[i];
          wts_[i] = (ValueType)wts[i];
        }
      }
      return nds_wts_lst;
    };
    static const auto nds_wts_lst = compute_nds_wts_lst();

    SCTL_ASSERT(ORDER < MaxOrder);
    return nds_wts_lst[ORDER];
  }

  template <class RealType, class Kernel> static Vector<Vector<RealType>> BuildToroidalSpecialQuadRules(Integer Nmodes) {
    constexpr Integer COORD_DIM = 3;
    constexpr Integer max_adap_depth = 30; // build quadrature rules for points up to 2*pi*0.5^max_adap_depth from source loop
    constexpr Integer crossover_adap_depth = 2;
    constexpr Integer max_digits = 20;

    #ifdef SCTL_QUAD_T
    using ValueType = QuadReal;
    #else
    using ValueType = long double;
    #endif
    Vector<Vector<ValueType>> data;
    const std::string fname = std::string("data/toroidal_quad_rule_m") + std::to_string(Nmodes) + "_" + Kernel::Name();
    ReadFile(data, fname);
    if (data.Dim() != max_adap_depth*max_digits) { // If file is not-found then compute quadrature rule and write to file
      data.ReInit(max_adap_depth * max_digits);
      for (Integer idx = 0; idx < max_adap_depth; idx++) {
        Vector<Vector<ValueType>> quad_nds,  quad_wts;
        { // generate special quadrature rule
          Vector<ValueType> nds, wts;
          Matrix<ValueType> Mintegrands;
          auto discretize_basis_functions = [Nmodes](Matrix<ValueType>& Mintegrands, Vector<ValueType>& nds, Vector<ValueType>& wts, ValueType dist, const std::pair<Vector<ValueType>,Vector<ValueType>>& panel_quad_nds_wts) {
            auto trg_coord = [](ValueType dist, Long M) {
              Vector<ValueType> Xtrg; //(M*M*COORD_DIM);
              for (Long i = 0; i < M; i++) {
                for (Long j = 0; j < M; j++) {
                  ValueType theta = i*2*const_pi<ValueType>()/(M);
                  ValueType r = (0.5 + i*0.5/(M)) * dist;
                  ValueType x0 = r*cos<ValueType>(theta);
                  ValueType x1 = 0;
                  ValueType x2 = r*sin<ValueType>(theta);
                  if (x0 > 0) {
                    Xtrg.PushBack(x0);
                    Xtrg.PushBack(x1);
                    Xtrg.PushBack(x2);
                  }
                }
              }
              return Xtrg;
            };
            Vector<ValueType> Xtrg = trg_coord(dist, 15); // TODO: determine optimal sample count
            Long Ntrg = Xtrg.Dim()/COORD_DIM;

            auto adap_nds_wts = [&panel_quad_nds_wts](Vector<ValueType>& nds, Vector<ValueType>& wts, Integer levels){ // discretization in interval [-0.5,0.5]
              const auto& leg_nds = panel_quad_nds_wts.first;
              const auto& leg_wts = panel_quad_nds_wts.second;
              SCTL_ASSERT(levels);
              Long N = 2*levels;
              ValueType l = 0.5;
              nds.ReInit(N*leg_nds.Dim());
              wts.ReInit(N*leg_nds.Dim());
              for (Integer idx = 0; idx < levels; idx++) {
                l *= (idx<levels-1 ? 0.5 : 1.0);
                Vector<ValueType> nds0(leg_nds.Dim(), nds.begin()+(  idx  )*leg_nds.Dim(), false);
                Vector<ValueType> nds1(leg_nds.Dim(), nds.begin()+(N-idx-1)*leg_nds.Dim(), false);
                Vector<ValueType> wts0(leg_wts.Dim(), wts.begin()+(  idx  )*leg_wts.Dim(), false);
                Vector<ValueType> wts1(leg_wts.Dim(), wts.begin()+(N-idx-1)*leg_wts.Dim(), false);
                for (Long i = 0; i < leg_nds.Dim(); i++) {
                  ValueType s = leg_nds[i]*l + (idx<levels-1 ? l : 0);
                  nds0[                i] = s;
                  nds1[leg_nds.Dim()-1-i] =-s;
                  wts0[                i] = leg_wts[i]*l;
                  wts1[leg_nds.Dim()-1-i] = wts0[i];
                }
              }
            };
            adap_nds_wts(nds, wts, std::max<Integer>(1,(Integer)(log(dist/2/const_pi<ValueType>())/log(0.5)+0.5)));

            Long Nnds = nds.Dim();
            Vector<Complex<ValueType>> exp_itheta(Nnds), exp_iktheta(Nnds);
            Vector<ValueType> Xsrc(Nnds*COORD_DIM), Xn(Nnds*COORD_DIM);
            for (Long i = 0; i < Nnds; i++) {
              const ValueType cos_t = cos<ValueType>(2*const_pi<ValueType>()*nds[i]);
              const ValueType sin_t = sin<ValueType>(2*const_pi<ValueType>()*nds[i]);
              exp_iktheta[i].real = 1;
              exp_iktheta[i].imag = 0;
              exp_itheta[i].real = cos_t;
              exp_itheta[i].imag = sin_t;
              Xsrc[i*COORD_DIM+0] = -2*sin<ValueType>(const_pi<ValueType>()*nds[i])*sin<ValueType>(const_pi<ValueType>()*nds[i]); // == cos_t - 1
              Xsrc[i*COORD_DIM+1] = sin_t;
              Xsrc[i*COORD_DIM+2] = 0;
              Xn[i*COORD_DIM+0] = cos_t;
              Xn[i*COORD_DIM+1] = sin_t;
              Xn[i*COORD_DIM+2] = 0;
            }

            Kernel ker;
            Matrix<ValueType> Mker;
            ker.KernelMatrix(Mker, Xtrg, Xsrc, Xn);
            SCTL_ASSERT(Mker.Dim(0) == Nnds * Kernel::SrcDim());
            SCTL_ASSERT(Mker.Dim(1) == Ntrg * Kernel::TrgDim());

            Mintegrands.ReInit(Nnds, (Nmodes*2)*Kernel::SrcDim() * Ntrg*Kernel::TrgDim());
            for (Long k = 0; k < Nmodes; k++) {
              for (Long i = 0; i < Nnds; i++) {
                for (Long j = 0; j < Ntrg; j++) {
                  for (Long k0 = 0; k0 < Kernel::SrcDim(); k0++) {
                    for (Long k1 = 0; k1 < Kernel::TrgDim(); k1++) {
                      Mintegrands[i][(((k*2+0)*Kernel::SrcDim()+k0) *Ntrg+j)*Kernel::TrgDim()+k1] = Mker[i*Kernel::SrcDim()+k0][j*Kernel::TrgDim()+k1] * exp_iktheta[i].real;
                      Mintegrands[i][(((k*2+1)*Kernel::SrcDim()+k0) *Ntrg+j)*Kernel::TrgDim()+k1] = Mker[i*Kernel::SrcDim()+k0][j*Kernel::TrgDim()+k1] * exp_iktheta[i].imag;
                    }
                  }
                }
              }
              for (Long i = 0; i < Nnds; i++) {
                exp_iktheta[i] *= exp_itheta[i];
              }
            }
          };
          std::pair<Vector<ValueType>,Vector<ValueType>> panel_quad;
          { // Set panel_quad_rule
            auto leg_quad = LegendreQuadRule<ValueType>(45);
            const auto& leg_nds = leg_quad.first;
            const auto& leg_wts = leg_quad.second;
            auto& panel_nds = panel_quad.first;
            auto& panel_wts = panel_quad.second;

            const Long rep = 1;
            const ValueType scal = 1/(ValueType)rep;
            for (Long i = 0; i < rep; i++) {
              for (Long j = 0; j < leg_nds.Dim(); j++) {
                panel_nds.PushBack(leg_nds[j]*scal + i*scal);
                panel_wts.PushBack(leg_wts[j]*scal);
              }
            }
          }
          ValueType dist = 4*const_pi<ValueType>()*pow<ValueType,Long>(0.5,idx); // distance of target points from the source loop (which is a unit circle)
          discretize_basis_functions(Mintegrands, nds, wts, dist, panel_quad); // TODO: adaptively select Legendre order

          Vector<ValueType> eps_vec;
          for (Long k = 0; k < max_digits; k++) eps_vec.PushBack(pow<ValueType,Long>(0.1,k));
          std::cout<<"Level = "<<idx<<" of "<<max_adap_depth<<'\n';
          auto cond_num_vec = InterpQuadRule<ValueType>::Build(quad_nds, quad_wts,  Mintegrands, nds, wts, false, eps_vec);
        }
        for (Integer digits = 0; digits < max_digits; digits++) {
          Long N = quad_nds[digits].Dim();
          data[idx*max_digits+digits].ReInit(3*N);
          for (Long i = 0; i < N; i++) {
            data[idx*max_digits+digits][i*3+0] = cos<ValueType>(2*const_pi<ValueType>()*quad_nds[digits][i]);
            data[idx*max_digits+digits][i*3+1] = sin<ValueType>(2*const_pi<ValueType>()*quad_nds[digits][i]);
            data[idx*max_digits+digits][i*3+2] = (2*const_pi<ValueType>()*quad_wts[digits][i]);
          }
        }
      }
      WriteFile(data, fname);
    }
    for (Integer idx = 0; idx < crossover_adap_depth; idx++) { // Use trapezoidal rule up to crossover_adap_depth
      for (Integer digits = 0; digits < max_digits; digits++) {
        Long N = std::max<Long>(digits*pow<Long,Long>(2,idx), Nmodes); // TODO: determine optimal order by testing error or adaptively
        data[idx*max_digits+digits].ReInit(3*N);
        for (Long i = 0; i < N; i++) {
          ValueType quad_nds = i/(ValueType)N;
          ValueType quad_wts = 1/(ValueType)N;
          data[idx*max_digits+digits][i*3+0] = cos<ValueType>(2*const_pi<ValueType>()*quad_nds);
          data[idx*max_digits+digits][i*3+1] = sin<ValueType>(2*const_pi<ValueType>()*quad_nds);
          data[idx*max_digits+digits][i*3+2] = (2*const_pi<ValueType>()*quad_wts);
        }
      }
    }

    Vector<Vector<RealType>> quad_rule_lst;
    quad_rule_lst.ReInit(data.Dim()*3);
    for (Integer i = 0; i < data.Dim(); i++) {
      uint64_t data_len = data[i].Dim()/3;
      quad_rule_lst[i*3+0].ReInit(data_len);
      quad_rule_lst[i*3+1].ReInit(data_len);
      quad_rule_lst[i*3+2].ReInit(data_len);
      for (Long j = 0; j < (Long)data_len; j++) {
        quad_rule_lst[i*3+0][j] = (RealType)data[i][j*3+0];
        quad_rule_lst[i*3+1][j] = (RealType)data[i][j*3+1];
        quad_rule_lst[i*3+2][j] = (RealType)data[i][j*3+2];
      }
    }
    return quad_rule_lst;
  }
  template <class RealType, Integer VecLen, Integer ModalUpsample, class Kernel> static Complex<RealType> ToroidalSpecialQuadRule(Matrix<RealType>& Mfourier, Vector<RealType>& nds_cos_theta, Vector<RealType>& nds_sin_theta, Vector<RealType>& wts, const Integer Nmodes, const Tensor<RealType,true,3,1>& Xt_X0, const Tensor<RealType,true,3,1>& e1, const Tensor<RealType,true,3,1>& e2, const Tensor<RealType,true,3,1>& e1xe2, RealType R0, Integer digits) {
    static constexpr Integer max_adap_depth = 30; // build quadrature rules for points up to 2*pi*0.5^max_adap_depth from source loop
    static constexpr Integer crossover_adap_depth = 2;
    static constexpr Integer max_digits = 20;
    if (digits >= max_digits) digits = max_digits-1;
    //SCTL_ASSERT(digits<max_digits);

    const RealType XX = dot_prod(Xt_X0, e1);
    const RealType YY = dot_prod(Xt_X0, e2);
    const RealType ZZ = dot_prod(Xt_X0, e1xe2);
    const RealType R = sqrt<RealType>(XX*XX+YY*YY);
    const RealType Rinv = 1/R;
    const Complex<RealType> exp_theta0(XX*Rinv, YY*Rinv);
    Long adap_depth = 0;
    { // Set adap_depth
      const RealType dtheta = sqrt<RealType>((R-R0)*(R-R0) + ZZ*ZZ)/R0;
      for (RealType s = dtheta; s<2*const_pi<RealType>(); s*=2) adap_depth++;
      if (adap_depth >= max_adap_depth) {
        SCTL_WARN("Toroidal quadrature evaluation is outside of the range of precomputed quadratures; accuracy may be sverely degraded.");
        adap_depth = max_adap_depth-1;
      }
    }

    SCTL_ASSERT(Nmodes < 100);
    static Vector<Vector<Matrix<RealType>>> all_fourier_basis(100);
    static Vector<Vector<Vector<RealType>>> all_quad_nds_cos_theta(100);
    static Vector<Vector<Vector<RealType>>> all_quad_nds_sin_theta(100);
    static Vector<Vector<Vector<RealType>>> all_quad_wts(100);
    #pragma omp critical(SCTL_ToroidalSpecialQuadRule)
    if (all_quad_wts[Nmodes].Dim() == 0) {
      auto quad_rules = BuildToroidalSpecialQuadRules<RealType,Kernel>(Nmodes);
      const Long Nrules = quad_rules.Dim()/3;

      Vector<Matrix<RealType>> fourier_basis(Nrules);
      Vector<Vector<RealType>> quad_nds_cos_theta(Nrules);
      Vector<Vector<RealType>> quad_nds_sin_theta(Nrules);
      Vector<Vector<RealType>> quad_wts(Nrules);
      for (Long i = 0; i < Nrules; i++) { // Set quad_nds_cos_theta, quad_nds_sin_theta, quad_wts, fourier_basis
        const Integer Nnds_ = quad_rules[i*3+0].Dim();
        const Integer Nnds = ((Nnds_+VecLen-1)/VecLen)*VecLen;
        Vector<Complex<RealType>> exp_itheta(Nnds);
        Vector<Complex<RealType>> exp_iktheta(Nnds);
        quad_nds_cos_theta[i].ReInit(Nnds);
        quad_nds_sin_theta[i].ReInit(Nnds);
        quad_wts[i].ReInit(Nnds);
        Integer j;
        for (j = 0; j < Nnds_; j++) {
          exp_itheta[j].real = quad_nds_cos_theta[i][j] = quad_rules[i*3+0][j];
          exp_itheta[j].imag = quad_nds_sin_theta[i][j] = quad_rules[i*3+1][j];
          quad_wts[i][j] = quad_rules[i*3+2][j];
          exp_iktheta[j].real = 1;
          exp_iktheta[j].imag = 0;
        }
        for (; j < Nnds; j++) {
          exp_itheta[j].real = quad_nds_cos_theta[i][j] = quad_rules[i*3+0][0];
          exp_itheta[j].imag = quad_nds_sin_theta[i][j] = quad_rules[i*3+1][0];
          quad_wts[i][j] = 0;
          exp_iktheta[j].real = 1;
          exp_iktheta[j].imag = 0;
        }

        auto& Mexp_iktheta = fourier_basis[i];
        Mexp_iktheta.ReInit(Nnds, (Nmodes-ModalUpsample)*2);
        for (Integer k = 0; k < Nmodes-ModalUpsample; k++) {
          for (Integer j = 0; j < Nnds; j++) {
            Mexp_iktheta[j][k*2+0] = exp_iktheta[j].real;
            Mexp_iktheta[j][k*2+1] = exp_iktheta[j].imag;
          }
          exp_iktheta *= exp_itheta;
        }
      }
      all_fourier_basis[Nmodes].Swap(fourier_basis);
      all_quad_nds_cos_theta[Nmodes].Swap(quad_nds_cos_theta);
      all_quad_nds_sin_theta[Nmodes].Swap(quad_nds_sin_theta);
      all_quad_wts[Nmodes].Swap(quad_wts);
    }

    { // Set Mfourier, nds_cos_theta, nds_sin_theta, wts
      const Long quad_idx = adap_depth*max_digits+digits;
      const auto& Mfourier0 = all_fourier_basis[Nmodes][quad_idx];
      const auto& nds0_cos_theta = all_quad_nds_cos_theta[Nmodes][quad_idx];
      const auto& nds0_sin_theta = all_quad_nds_sin_theta[Nmodes][quad_idx];
      const auto& wts0 = all_quad_wts[Nmodes][quad_idx];
      const Long N = wts0.Dim();

      Mfourier.ReInit(Mfourier0.Dim(0), Mfourier0.Dim(1), (Iterator<RealType>)Mfourier0.begin(), false);
      nds_cos_theta.ReInit(N, (Iterator<RealType>)nds0_cos_theta.begin(), false);
      nds_sin_theta.ReInit(N, (Iterator<RealType>)nds0_sin_theta.begin(), false);
      wts.ReInit(N, (Iterator<RealType>)wts0.begin(), false);

      if (adap_depth >= crossover_adap_depth) return exp_theta0;
      else return Complex<RealType>(1,0);
    }
  }
  template <Integer digits, Integer ModalUpsample, bool trg_dot_prod, class RealType, class Kernel> static void toroidal_greens_fn_batched(Matrix<RealType>& M, const Tensor<RealType,true,3,1>& y_trg, const Tensor<RealType,true,3,1>& n_trg, const Matrix<RealType>& x_src, const Matrix<RealType>& dx_src, const Matrix<RealType>& d2x_src, const Matrix<RealType>& r_src, const Matrix<RealType>& dr_src, const Matrix<RealType>& e1_src, const Matrix<RealType>& e2_src, const Matrix<RealType>& de1_src, const Matrix<RealType>& de2_src, const Kernel& ker, const Integer FourierModes) {
    static constexpr Integer VecLen = DefaultVecLen<RealType>();
    using VecType = Vec<RealType, VecLen>;

    constexpr Integer COORD_DIM = 3;
    using Vec3 = Tensor<RealType,true,COORD_DIM,1>;
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1 = Kernel::TrgDim()/(trg_dot_prod?COORD_DIM:1);
    static constexpr Integer Nbuff = 10000; // TODO

    const Long BatchSize = M.Dim(0);
    SCTL_ASSERT(M.Dim(1) == KDIM0*KDIM1*FourierModes*2);
    SCTL_ASSERT(  x_src.Dim(1) == BatchSize &&   x_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT( dx_src.Dim(1) == BatchSize &&  dx_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT(d2x_src.Dim(1) == BatchSize && d2x_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT(  r_src.Dim(1) == BatchSize &&   r_src.Dim(0) ==         1);
    SCTL_ASSERT( dr_src.Dim(1) == BatchSize &&  dr_src.Dim(0) ==         1);
    SCTL_ASSERT( e1_src.Dim(1) == BatchSize &&  e1_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT( e2_src.Dim(1) == BatchSize &&  e2_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT(de1_src.Dim(1) == BatchSize && de1_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT(de2_src.Dim(1) == BatchSize && de2_src.Dim(0) == COORD_DIM);
    VecType n_trg_[COORD_DIM] = {n_trg(0,0),n_trg(1,0),n_trg(2,0)};
    for (Long ii = 0; ii < BatchSize; ii++) {
      RealType r = r_src[0][ii], dr = dr_src[0][ii];
      Vec3 x, dx, d2x, e1, e2, de1, de2;
      { // Set x, dx, d2x, e1, e2, de1, de2
        for (Integer k = 0; k < COORD_DIM; k++) {
          x  (k,0) =   x_src[k][ii];
          dx (k,0) =  dx_src[k][ii];
          d2x(k,0) = d2x_src[k][ii];
          e1 (k,0) =  e1_src[k][ii];
          e2 (k,0) =  e2_src[k][ii];
          de1(k,0) = de1_src[k][ii];
          de2(k,0) = de2_src[k][ii];
        }
      }

      auto toroidal_greens_fn = [&ker,&n_trg_](Matrix<RealType>& M, const Vec3& Xt, const Vec3& x, const Vec3& dx, const Vec3& d2x, const Vec3& e1, const Vec3& e2, const Vec3& de1, const Vec3& de2, const RealType r, const RealType dr, const Integer FourierModes) {
        SCTL_ASSERT(M.Dim(0) ==    KDIM0*KDIM1);
        SCTL_ASSERT(M.Dim(1) == FourierModes*2);

        Matrix<RealType> Mexp_iktheta;
        Vector<RealType> nds_cos_theta, nds_sin_theta, wts;
        const auto exp_theta = ToroidalSpecialQuadRule<RealType,VecLen,ModalUpsample,Kernel>(Mexp_iktheta, nds_cos_theta, nds_sin_theta, wts, FourierModes+ModalUpsample, Xt-x, e1, e2, cross_prod(e1,e2), r, digits);
        const Long Nnds = wts.Dim();
        SCTL_ASSERT(Nnds < Nbuff);

        { // Set M
          const VecType exp_theta_real(exp_theta.real);
          const VecType exp_theta_imag(exp_theta.imag);
          const VecType vec_dx[3] = {VecType(dx(0,0)), VecType(dx(1,0)), VecType(dx(2,0))};

          const VecType vec_dy0[3] = {Xt(0,0)-x(0,0), Xt(1,0)-x(1,0), Xt(2,0)-x(2,0)};
          const VecType vec_dy1[3] = {-e1(0,0)*r, -e1(1,0)*r, -e1(2,0)*r};
          const VecType vec_dy2[3] = {-e2(0,0)*r, -e2(1,0)*r, -e2(2,0)*r};

          const VecType vec_dy_ds1[3] = {e1(0,0)*dr+de1(0,0)*r, e1(1,0)*dr+de1(1,0)*r, e1(2,0)*dr+de1(2,0)*r};
          const VecType vec_dy_ds2[3] = {e2(0,0)*dr+de2(0,0)*r, e2(1,0)*dr+de2(1,0)*r, e2(2,0)*dr+de2(2,0)*r};
          const VecType vec_dy_dt1[3] = {e2(0,0)*r, e2(1,0)*r, e2(2,0)*r};
          const VecType vec_dy_dt2[3] = {e1(0,0)*r, e1(1,0)*r, e1(2,0)*r};

          alignas(sizeof(VecType)) StaticArray<RealType,KDIM0*KDIM1*Nbuff> mem_buff;
          Matrix<RealType> Mker_da(KDIM0*KDIM1, Nnds, mem_buff, false);
          for (Integer j = 0; j < Nnds; j+=VecLen) { // Set Mker_da
            VecType dy[3], n[3], da;
            { // Set dy, n, da
              VecType nds_cos_theta_j = VecType::LoadAligned(&nds_cos_theta[j]);
              VecType nds_sin_theta_j = VecType::LoadAligned(&nds_sin_theta[j]);
              VecType cost = nds_cos_theta_j*exp_theta_real - nds_sin_theta_j*exp_theta_imag;
              VecType sint = nds_cos_theta_j*exp_theta_imag + nds_sin_theta_j*exp_theta_real;

              dy[0] = vec_dy0[0] + vec_dy1[0]*cost + vec_dy2[0]*sint;
              dy[1] = vec_dy0[1] + vec_dy1[1]*cost + vec_dy2[1]*sint;
              dy[2] = vec_dy0[2] + vec_dy1[2]*cost + vec_dy2[2]*sint;

              VecType dy_ds[3], dy_dt[3];
              dy_ds[0] = vec_dx[0] + vec_dy_ds1[0]*cost + vec_dy_ds2[0]*sint;
              dy_ds[1] = vec_dx[1] + vec_dy_ds1[1]*cost + vec_dy_ds2[1]*sint;
              dy_ds[2] = vec_dx[2] + vec_dy_ds1[2]*cost + vec_dy_ds2[2]*sint;
              dy_dt[0] = vec_dy_dt1[0]*cost - vec_dy_dt2[0]*sint;
              dy_dt[1] = vec_dy_dt1[1]*cost - vec_dy_dt2[1]*sint;
              dy_dt[2] = vec_dy_dt1[2]*cost - vec_dy_dt2[2]*sint;
              n[0] = dy_ds[1] * dy_dt[2] - dy_ds[2] * dy_dt[1];
              n[1] = dy_ds[2] * dy_dt[0] - dy_ds[0] * dy_dt[2];
              n[2] = dy_ds[0] * dy_dt[1] - dy_ds[1] * dy_dt[0];

              VecType da2 = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];
              VecType inv_da = approx_rsqrt<digits>(da2);
              da = da2 * inv_da;

              n[0] = n[0] * inv_da;
              n[1] = n[1] * inv_da;
              n[2] = n[2] * inv_da;
            }

            VecType Mker[KDIM0][Kernel::TrgDim()];
            ker.template uKerMatrix<digits, VecType>(Mker, dy, n, ker.GetCtxPtr());
            VecType da_wts = VecType::LoadAligned(&wts[j]) * da;
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
              for (Integer k1 = 0; k1 < KDIM1; k1++) {
                if (trg_dot_prod) {
                  VecType Mker_dot_n = FMA(Mker[k0][k1*COORD_DIM+0],n_trg_[0],
                                       FMA(Mker[k0][k1*COORD_DIM+1],n_trg_[1],
                                           Mker[k0][k1*COORD_DIM+2]*n_trg_[2]));
                  (Mker_dot_n*da_wts).StoreAligned(&Mker_da[k0*KDIM1+k1][j]);
                } else {
                  (Mker[k0][k1]*da_wts).StoreAligned(&Mker_da[k0*KDIM1+k1][j]);
                }
              }
            }
          }
          Matrix<RealType>::GEMM(M, Mker_da, Mexp_iktheta);

          Complex<RealType> exp_iktheta(1,0);
          for (Integer j = 0; j < FourierModes; j++) {
            for (Integer k = 0; k < KDIM0*KDIM1; k++) {
              Complex<RealType> Mjk(M[k][j*2+0],M[k][j*2+1]);
              Mjk *= exp_iktheta;
              M[k][j*2+0] = Mjk.real;
              M[k][j*2+1] = Mjk.imag;
            }
            exp_iktheta *= exp_theta;
          }
        }
      };
      Matrix<RealType> M_toroidal_greens_fn(KDIM0*KDIM1, FourierModes*2, M[ii], false);
      toroidal_greens_fn(M_toroidal_greens_fn, y_trg, x, dx, d2x, e1, e2, de1, de2, r, dr, FourierModes);
    }
  }

  template <Integer ModalUpsample, class ValueType, class Kernel, bool trg_dot_prod> static void SpecialQuadBuildBasisMatrix(Matrix<ValueType>& M, Vector<ValueType>& quad_nds, Vector<ValueType>& quad_wts, const Integer Ncheb, const Integer FourierModes, const ValueType s_trg, const Integer max_digits, const ValueType elem_length, const Integer RefLevels, const Kernel& ker) {
    // TODO: cleanup
    constexpr Integer COORD_DIM = 3;
    using Vec3 = Tensor<ValueType,true,COORD_DIM,1>;

    const Long LegQuadOrder = 2*max_digits;
    constexpr Long LogQuadOrder = 16; // this has non-negative weights

    constexpr Integer KDIM0 = Kernel::SrcDim();
    constexpr Integer KDIM1 = Kernel::TrgDim() / (trg_dot_prod ? COORD_DIM : 1);

    Vec3 y_trg, n_trg;
    y_trg(0,0) = 1;
    y_trg(1,0) = 0;
    y_trg(2,0) = s_trg * elem_length;
    n_trg(0,0) = 1;
    n_trg(1,0) = 0;
    n_trg(2,0) = 0;

    Vector<ValueType> radius(          Ncheb);
    Vector<ValueType> coord (COORD_DIM*Ncheb);
    Vector<ValueType> dr    (          Ncheb);
    Vector<ValueType> dx    (COORD_DIM*Ncheb);
    Vector<ValueType> d2x   (COORD_DIM*Ncheb);
    Vector<ValueType> e1    (COORD_DIM*Ncheb);
    for (Long i = 0; i < Ncheb; i++) {
      radius[i] = 1;
      dr[i] = 0;

      coord[0*Ncheb+i] = 0;
      coord[1*Ncheb+i] = 0;
      coord[2*Ncheb+i] = ChebQuadRule<ValueType>::nds(Ncheb)[i] * elem_length;

      dx[0*Ncheb+i] = 0;
      dx[1*Ncheb+i] = 0;
      dx[2*Ncheb+i] = elem_length;

      d2x[0*Ncheb+i] = 0;
      d2x[1*Ncheb+i] = 0;
      d2x[2*Ncheb+i] = 0;

      e1[0*Ncheb+i] = 1;
      e1[1*Ncheb+i] = 0;
      e1[2*Ncheb+i] = 0;
    }

    auto adap_ref = [&LegQuadOrder](Vector<ValueType>& nds_, Vector<ValueType>& wts_, const ValueType s, const Integer levels) {
      const auto& log_quad_nds = LogSingularityQuadRule<ValueType>(LogQuadOrder).first;
      const auto& log_quad_wts = LogSingularityQuadRule<ValueType>(LogQuadOrder).second;
      const auto& leg_nds = LegendreQuadRule<ValueType>(LegQuadOrder).first;
      const auto& leg_wts = LegendreQuadRule<ValueType>(LegQuadOrder).second;
      Vector<ValueType> nds;
      Vector<ValueType> wts;

      ValueType len0 = std::min(pow<ValueType>(0.5,levels), std::min(s, (1-s)));
      ValueType len1 = std::min<ValueType>(s, 1-s);
      ValueType len2 = std::max<ValueType>(s, 1-s);

      for (Long i = 0; i < log_quad_nds.Dim(); i++) {
        nds.PushBack(s + len0*log_quad_nds[i]);
        nds.PushBack(s - len0*log_quad_nds[i]);
        wts.PushBack(len0*log_quad_wts[i]);
        wts.PushBack(len0*log_quad_wts[i]);
      }

      for (ValueType start = len0; start < len1; start*=2) {
        ValueType step_ = std::min(start, len1-start);
        for (Long i = 0; i < leg_nds.Dim(); i++) {
          nds.PushBack(s + start + step_*leg_nds[i]);
          nds.PushBack(s - start - step_*leg_nds[i]);
          wts.PushBack(step_*leg_wts[i]);
          wts.PushBack(step_*leg_wts[i]);
        }
      }

      for (ValueType start = len1; start < len2; start*=2) {
        ValueType step_ = std::min(start, len2-start);
        for (Long i = 0; i < leg_nds.Dim(); i++) {
          if (s + start + step_*leg_nds[i] <= 1.0) {
            nds.PushBack(s + start + step_*leg_nds[i]);
            wts.PushBack(step_*leg_wts[i]);
          }
          if (s - start - step_*leg_nds[i] >= 0.0) {
            nds.PushBack(s - start - step_*leg_nds[i]);
            wts.PushBack(step_*leg_wts[i]);
          }
        }
      }

      nds_.ReInit(nds.Dim());
      wts_.ReInit(wts.Dim());
      Vector<std::pair<ValueType,Long>> sort_pair;
      for (Long i = 0; i < nds.Dim(); i++) {
        sort_pair.PushBack(std::pair<ValueType,Long>{nds[i], i});
      }
      std::sort(sort_pair.begin(), sort_pair.end());
      for (Long i = 0; i < nds.Dim(); i++) {
        const Long idx = sort_pair[i].second;
        nds_[i] = nds[idx];
        wts_[i] = wts[idx];
      }
    };
    adap_ref(quad_nds, quad_wts, s_trg, RefLevels); // adaptive quadrature rule

    Matrix<ValueType> Minterp_quad_nds;
    { // Set Minterp_quad_nds
      Minterp_quad_nds.ReInit(Ncheb, quad_nds.Dim());
      Vector<ValueType> Vinterp_quad_nds(Ncheb*quad_nds.Dim(), Minterp_quad_nds.begin(), false);
      LagrangeInterp<ValueType>::Interpolate(Vinterp_quad_nds, ChebQuadRule<ValueType>::nds(Ncheb), quad_nds);
    }

    Matrix<ValueType> r_src, dr_src, x_src, dx_src, d2x_src, e1_src, e2_src, de1_src, de2_src;
    r_src  .ReInit(        1,quad_nds.Dim());
    dr_src .ReInit(        1,quad_nds.Dim());
    x_src  .ReInit(COORD_DIM,quad_nds.Dim());
    dx_src .ReInit(COORD_DIM,quad_nds.Dim());
    d2x_src.ReInit(COORD_DIM,quad_nds.Dim());
    e1_src .ReInit(COORD_DIM,quad_nds.Dim());
    e2_src .ReInit(COORD_DIM,quad_nds.Dim());
    de1_src.ReInit(COORD_DIM,quad_nds.Dim());
    de2_src.ReInit(COORD_DIM,quad_nds.Dim());
    Matrix<ValueType>::GEMM(  x_src, Matrix<ValueType>(COORD_DIM,Ncheb, coord.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM( dx_src, Matrix<ValueType>(COORD_DIM,Ncheb,    dx.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM(d2x_src, Matrix<ValueType>(COORD_DIM,Ncheb,   d2x.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM(  r_src, Matrix<ValueType>(        1,Ncheb,radius.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM( dr_src, Matrix<ValueType>(        1,Ncheb,    dr.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM( e1_src, Matrix<ValueType>(COORD_DIM,Ncheb,    e1.begin(),false), Minterp_quad_nds);
    for (Long j = 0; j < quad_nds.Dim(); j++) { // Set e2_src
      Vec3 e1, dx, d2x;
      for (Integer k = 0; k < COORD_DIM; k++) {
        e1(k,0) = e1_src[k][j];
        dx(k,0) = dx_src[k][j];
        d2x(k,0) = d2x_src[k][j];
      }
      ValueType inv_dx2 = 1/dot_prod(dx,dx);
      e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
      e1 = e1 * (1/sqrt<ValueType>(dot_prod(e1,e1)));

      Vec3 e2 = cross_prod(e1, dx);
      e2 = e2 * (1/sqrt<ValueType>(dot_prod(e2,e2)));
      Vec3 de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
      Vec3 de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
      for (Integer k = 0; k < COORD_DIM; k++) {
        e1_src[k][j] = e1(k,0);
        e2_src[k][j] = e2(k,0);
        de1_src[k][j] = de1(k,0);
        de2_src[k][j] = de2(k,0);
      }
    }

    Matrix<ValueType> M_tor(quad_nds.Dim(), KDIM0*KDIM1*FourierModes*2);
    constexpr Integer TorGreensFnDigits = (Integer)(TypeTraits<ValueType>::SigBits*0.3010299957);
    toroidal_greens_fn_batched<TorGreensFnDigits,ModalUpsample,trg_dot_prod>(M_tor, y_trg, n_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, e2_src, de1_src, de2_src, ker, FourierModes);

    M.ReInit(quad_nds.Dim(), Ncheb*FourierModes*2*KDIM0*KDIM1);
    for (Long i = 0; i < quad_nds.Dim(); i++) {
      for (Long j = 0; j < Ncheb; j++) {
        for (Long k = 0; k < KDIM0*KDIM1*FourierModes*2; k++) {
          M[i][j*KDIM0*KDIM1*FourierModes*2+k] = Minterp_quad_nds[j][i] * M_tor[i][k];
        }
      }
    }
  }
  template <Integer ModalUpsample, class ValueType, class Kernel, bool trg_dot_prod, bool symmetric=false/*must be set true for hypersingular kernels*/> static Vector<Vector<ValueType>> BuildSpecialQuadRules(const Integer Ncheb, const Integer FourierModes, const Integer trg_node_idx, const ValueType elem_length) {
    constexpr Integer Nlen = 10; // number of length samples in [elem_length/sqrt(2), elem_length*sqrt(2)]
    constexpr Integer max_digits = 19;
    const ValueType s_trg = ChebQuadRule<ValueType>::nds(Ncheb)[trg_node_idx];
    const Integer adap_depth = (Integer)(log<ValueType>(elem_length)/log<ValueType>(2)+4);
    const ValueType eps = 8*machine_eps<ValueType>();

    Kernel ker;
    Vector<ValueType> nds, wts;
    Matrix<ValueType> Mintegrands;
    { // Set nds, wts, Mintegrands
      Vector<Matrix<ValueType>> Mker(Nlen);
      Vector<Vector<ValueType>> nds_(Nlen), wts_(Nlen);
      //#pragma omp parallel for schedule(static) // TODO: prevents parallelization of precomputation of toroidal quadrature rule
      for (Long k = 0; k < Nlen; k++) {
        ValueType length = elem_length/sqrt<ValueType>(2.0)*k/(Nlen-1) + elem_length*sqrt<ValueType>(2.0)*(Nlen-k-1)/(Nlen-1);
        SpecialQuadBuildBasisMatrix<ModalUpsample,ValueType,Kernel,trg_dot_prod>(Mker[k], nds_[k], wts_[k], Ncheb, FourierModes, s_trg, max_digits, length, adap_depth, ker);
      }
      const Long N0 = nds_[0].Dim();

      Vector<Long> cnt(Nlen), dsp(Nlen); dsp[0] = 0;
      for (Long k = 0; k < Nlen; k++) {
        cnt[k] = Mker[k].Dim(1);
      }
      omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());

      const Long Nsplit = (symmetric ? std::lower_bound(nds_[0].begin(), nds_[0].end(), s_trg) - nds_[0].begin() : N0);
      const Long N = std::max<Long>(N0 - Nsplit, Nsplit);

      nds.ReInit(N);
      wts.ReInit(N);
      Mintegrands.ReInit(N, dsp[Nlen-1] + cnt[Nlen-1]);
      if (N == Nsplit) {
        #pragma omp parallel for schedule(static)
        for (Long k = 0; k < Nlen; k++) {
          for (Long i = 0; i < Nsplit; i++) {
            for (Long j = 0; j < cnt[k]; j++) {
              Mintegrands[i][dsp[k]+j] = Mker[k][i][j];
            }
          }

          for (Long i = Nsplit; i < N0; i++) {
            for (Long j = 0; j < cnt[k]; j++) {
              Mintegrands[2*Nsplit-i-1][dsp[k]+j] += Mker[k][i][j];
            }
          }
        }

        for (Long i = 0; i < Nsplit; i++) {
          nds[i] = nds_[0][i];
          wts[i] = wts_[0][i];
        }
        for (Long i = Nsplit; i < N0; i++) {
          SCTL_ASSERT(fabs(nds[2*Nsplit-i-1] + nds_[0][i] - 2*s_trg) < eps);
          SCTL_ASSERT(fabs(wts[2*Nsplit-i-1] - wts_[0][i]) < eps);
        }
      } else {
        #pragma omp parallel for schedule(static)
        for (Long k = 0; k < Nlen; k++) {
          for (Long i = Nsplit; i < N0; i++) {
            for (Long j = 0; j < cnt[k]; j++) {
              Mintegrands[i-Nsplit][dsp[k]+j] = Mker[k][i][j];
            }
          }

          for (Long i = 0; i < Nsplit; i++) {
            for (Long j = 0; j < cnt[k]; j++) {
              Mintegrands[Nsplit-i-1][dsp[k]+j] += Mker[k][i][j];
            }
          }
        }

        for (Long i = Nsplit; i < N0; i++) {
          nds[i-Nsplit] = nds_[0][i];
          wts[i-Nsplit] = wts_[0][i];
        }
        for (Long i = 0; i < Nsplit; i++) {
          SCTL_ASSERT(fabs(nds[Nsplit-i-1] + nds_[0][i] - 2*s_trg) < eps);
          SCTL_ASSERT(fabs(wts[Nsplit-i-1] - wts_[0][i]) < eps);
        }
      }
    }

    Vector<Vector<ValueType>> nds_wts(max_digits*2);
    { // Set nds_wts
      Vector<ValueType> eps_vec;
      Vector<Vector<ValueType>> quad_nds, quad_wts;
      for (Long k = 0; k < max_digits; k++) eps_vec.PushBack(pow<ValueType,Long>(0.1,k));
      InterpQuadRule<ValueType>::Build(quad_nds, quad_wts,  Mintegrands, nds, wts, false, eps_vec);
      SCTL_ASSERT(quad_nds.Dim() == max_digits);
      SCTL_ASSERT(quad_wts.Dim() == max_digits);
      for (Long k = 0; k < max_digits; k++) {
        for (Long i = 0; i < quad_nds[k].Dim(); i++) {
          const ValueType qx0 = quad_nds[k][i];
          const ValueType qx1 = 2*s_trg - qx0;
          const ValueType qw = quad_wts[k][i];

          nds_wts[k*2+0].PushBack(qx0);
          nds_wts[k*2+1].PushBack(qw);

          if (symmetric && 0 <= qx1 && qx1 <= (ValueType)1) {
            nds_wts[k*2+0].PushBack(qx1);
            nds_wts[k*2+1].PushBack(qw);
          }
        }
      }
    }
    return nds_wts;
  }
  template <Integer ModalUpsample, class Real, class Kernel, bool trg_dot_prod, bool adap_quad=false> static void SpecialQuadRule(Vector<Real>& nds, Vector<Real>& wts, Matrix<Real>& Minterp, const Integer ChebOrder, const Integer trg_node_idx, const Real elem_radius, const Real elem_length, const Integer digits) {
    constexpr Integer max_adap_depth = 23; // TODO
    constexpr Integer MaxFourierModes = 8; // TODO
    constexpr Integer MaxChebOrder = 100;
    constexpr Integer max_digits = 19;

    auto LogSingularQuadOrder = [](Integer digits) { return 2*digits; }; // TODO: determine optimal order
    auto LegQuadOrder = [](Integer digits) { return digits; }; // TODO: determine optimal order

    if (!adap_quad) {
      auto load_special_quad_rule = [&max_adap_depth](Vector<Vector<Real>>& nds_lst, Vector<Vector<Real>>& wts_lst, Vector<Matrix<Real>>& interp_mat_lst, const Integer ChebOrder){
        #ifdef SCTL_QUAD_T
        using ValueType = QuadReal;
        #else
        using ValueType = long double;
        #endif
        const std::string fname = std::string("data/special_quad_q") + std::to_string(ChebOrder) + "_" + Kernel::Name() + (trg_dot_prod ? "_dotXn" : "");

        Vector<Vector<ValueType>> data;
        ReadFile(data, fname);
        if (data.Dim() != max_adap_depth*ChebOrder*max_digits*2) { // build quadrature rules
          data.ReInit(max_adap_depth*ChebOrder*max_digits*2);
          ValueType length = 64*1024;
          for (Integer i = 0; i < max_adap_depth; i++) {
            std::cout<<"length = "<<length<<'\n';
            for (Integer trg_node_idx = 0; trg_node_idx < ChebOrder; trg_node_idx++) {
              auto nds_wts = BuildSpecialQuadRules<ModalUpsample,ValueType,Kernel,trg_dot_prod>(ChebOrder, MaxFourierModes, trg_node_idx, length);
              for (Long j = 0; j < max_digits; j++) {
                data[((i*ChebOrder+trg_node_idx) * max_digits+j)*2+0] = nds_wts[j*2+0];
                data[((i*ChebOrder+trg_node_idx) * max_digits+j)*2+1] = nds_wts[j*2+1];
              }
            }
            length *= (ValueType)0.5;
          }
          WriteFile(data, fname);
        }

        nds_lst.ReInit(max_adap_depth*ChebOrder*max_digits);
        wts_lst.ReInit(max_adap_depth*ChebOrder*max_digits);
        interp_mat_lst.ReInit(max_adap_depth*ChebOrder*max_digits);
        for (Long i = 0; i < max_adap_depth*ChebOrder*max_digits; i++) { // Set nds_wts_lst
          const auto& nds_ = data[i*2+0];
          const auto& wts_ = data[i*2+1];

          nds_lst[i].ReInit(nds_.Dim());
          wts_lst[i].ReInit(wts_.Dim());
          for (Long j = 0; j < nds_.Dim(); j++) {
            nds_lst[i][j] = (Real)nds_[j];
            wts_lst[i][j] = (Real)wts_[j];
          }

          Matrix<ValueType> Minterp(ChebOrder, nds_.Dim());
          Vector<ValueType> Vinterp(ChebOrder*nds_.Dim(), Minterp.begin(), false);
          LagrangeInterp<ValueType>::Interpolate(Vinterp, ChebQuadRule<ValueType>::nds(ChebOrder), nds_);

          interp_mat_lst[i].ReInit(ChebOrder, nds_.Dim());
          for (Long j = 0; j < ChebOrder*nds_.Dim(); j++) interp_mat_lst[i][0][j] = (Real)Minterp[0][j];
        }
      };
      static Vector<Vector<Vector<Real>>> nds_lst(MaxChebOrder);
      static Vector<Vector<Vector<Real>>> wts_lst(MaxChebOrder);
      static Vector<Vector<Matrix<Real>>> interp_mat_lst(MaxChebOrder);
      SCTL_ASSERT(ChebOrder < MaxChebOrder);
      #pragma omp critical(SCTL_SpecialQuadRule)
      if (!nds_lst[ChebOrder].Dim()) {
        load_special_quad_rule(nds_lst[ChebOrder], wts_lst[ChebOrder], interp_mat_lst[ChebOrder], ChebOrder);
      }

      Long quad_idx = (Long)(16 - log2((double)(elem_length/elem_radius*sqrt<Real>(0.5))));
      if (quad_idx < 0 || quad_idx > max_adap_depth-1) {
        SCTL_WARN("Slender element aspect-ratio is outside of the range of precomputed quadratures; accuracy may be sverely degraded.");
      }
      quad_idx = std::max<Integer>(0, std::min<Integer>(max_adap_depth-1, quad_idx));

      nds = nds_lst[ChebOrder][(quad_idx*ChebOrder+trg_node_idx) * max_digits+digits];
      wts = wts_lst[ChebOrder][(quad_idx*ChebOrder+trg_node_idx) * max_digits+digits];
      Minterp = interp_mat_lst[ChebOrder][(quad_idx*ChebOrder+trg_node_idx) * max_digits+digits];
    } else {
      const auto& log_sing_nds_wts = LogSingularityQuadRule<Real>(LogSingularQuadOrder(digits));
      const auto& leg_nds_wts = LegendreQuadRule<Real>(LegQuadOrder(digits));
      const auto& sing_nds = log_sing_nds_wts.first;
      const auto& sing_wts = log_sing_nds_wts.second;
      const auto& leg_nds = leg_nds_wts.first;
      const auto& leg_wts = leg_nds_wts.second;
      nds.ReInit(0);
      wts.ReInit(0);

      const Real s_trg = ChebQuadRule<Real>::nds(ChebOrder)[trg_node_idx];
      const Real sing_quad_len = std::min(0.8*elem_radius/elem_length, std::min(s_trg, (1-s_trg)));
      for (Long i = 0; i < sing_nds.Dim(); i++) {
        nds.PushBack(s_trg + sing_quad_len*sing_nds[i]);
        nds.PushBack(s_trg - sing_quad_len*sing_nds[i]);
        wts.PushBack(sing_quad_len*sing_wts[i]);
        wts.PushBack(sing_quad_len*sing_wts[i]);
      }

      Real a = s_trg - sing_quad_len;
      for (Real step = sing_quad_len; step < 2; step*=2) {
        Real a_ = a;
        Real b_ = a - step;
        b_ = std::min<Real>(b_, 1.0);
        b_ = std::max<Real>(b_, 0.0);
        Real len = fabs(a_ - b_);
        if (len > 0) {
          for (Long i = 0; i < leg_nds.Dim(); i++) {
            nds.PushBack(a_*leg_nds[i] + b_*(1-leg_nds[i]));
            wts.PushBack(len * leg_wts[i]);
          }
        }
        a = b_;
      }

      a = s_trg + sing_quad_len;
      for (Real step = sing_quad_len; step < 2; step*=2) {
        Real a_ = a;
        Real b_ = a_ + step;
        b_ = std::min<Real>(b_, 1.0);
        b_ = std::max<Real>(b_, 0.0);
        Real len = fabs(a_ - b_);
        if (len > 0) {
          for (Long i = 0; i < leg_nds.Dim(); i++) {
            nds.PushBack(a_*leg_nds[i] + b_*(1-leg_nds[i]));
            wts.PushBack(len * leg_wts[i]);
          }
        }
        a = b_;
      }

      // TODO: this should be done in higher precision
      Minterp.ReInit(ChebOrder, nds.Dim());
      Vector<Real> Vinterp(ChebOrder*nds.Dim(), Minterp.begin(), false);
      LagrangeInterp<Real>::Interpolate(Vinterp, ChebQuadRule<Real>::nds(ChebOrder), nds);
    }
  }




  template <class Real> SlenderElemList<Real>::SlenderElemList(const Vector<Long>& cheb_order0, const Vector<Long>& fourier_order0, const Vector<Real>& coord0, const Vector<Real>& radius0, const Vector<Real>& orientation0) {
    Init(cheb_order0, fourier_order0, coord0, radius0, orientation0);
  }
  template <class Real> void SlenderElemList<Real>::Init(const Vector<Long>& cheb_order0, const Vector<Long>& fourier_order0, const Vector<Real>& coord0, const Vector<Real>& radius0, const Vector<Real>& orientation0) {
    using Vec3 = Tensor<Real,true,COORD_DIM,1>;
    const Long Nelem = cheb_order0.Dim();
    SCTL_ASSERT(fourier_order0.Dim() == Nelem);

    cheb_order = cheb_order0;
    fourier_order = fourier_order0;
    elem_dsp.ReInit(Nelem);
    if (Nelem) elem_dsp[0] = 0;
    omp_par::scan(cheb_order.begin(), elem_dsp.begin(), Nelem);

    const Long Nnodes = (Nelem ? cheb_order[Nelem-1]+elem_dsp[Nelem-1] : 0);
    SCTL_ASSERT_MSG(coord0.Dim() == Nnodes * COORD_DIM, "Length of the coordinate vector does not match the number of nodes.");
    SCTL_ASSERT_MSG(radius0.Dim() == Nnodes, "Length of the radius vector does not match the number of nodes.");

    radius = radius0;
    coord.ReInit(COORD_DIM*Nnodes);
    e1   .ReInit(COORD_DIM*Nnodes);
    dr   .ReInit(          Nnodes);
    dx   .ReInit(COORD_DIM*Nnodes);
    d2x  .ReInit(COORD_DIM*Nnodes);
    for (Long i = 0; i < Nelem; i++) { // Set coord, radius, dr, ds, d2s
      const Long Ncheb = cheb_order[i];
      Vector<Real> radius_(          Ncheb, radius.begin()+          elem_dsp[i], false);
      Vector<Real>  coord_(COORD_DIM*Ncheb,  coord.begin()+COORD_DIM*elem_dsp[i], false);
      Vector<Real>     e1_(COORD_DIM*Ncheb,     e1.begin()+COORD_DIM*elem_dsp[i], false);
      Vector<Real>     dr_(          Ncheb,     dr.begin()+          elem_dsp[i], false);
      Vector<Real>     dx_(COORD_DIM*Ncheb,     dx.begin()+COORD_DIM*elem_dsp[i], false);
      Vector<Real>    d2x_(COORD_DIM*Ncheb,    d2x.begin()+COORD_DIM*elem_dsp[i], false);

      const Vector<Real> coord__(COORD_DIM*Ncheb, (Iterator<Real>)coord0.begin()+elem_dsp[i]*COORD_DIM, false);
      for (Long j = 0; j < Ncheb; j++) { // Set coord_
        for (Long k = 0; k < COORD_DIM; k++) {
          coord_[k*Ncheb+j] = coord__[j*COORD_DIM+k];
        }
      }

      LagrangeInterp<Real>::Derivative( dr_, radius_, CenterlineNodes(Ncheb));
      LagrangeInterp<Real>::Derivative( dx_,  coord_, CenterlineNodes(Ncheb));
      LagrangeInterp<Real>::Derivative(d2x_,     dx_, CenterlineNodes(Ncheb));
      if (orientation0.Dim()) { // Set e1_
        SCTL_ASSERT(orientation0.Dim() == Nnodes*COORD_DIM);
        const Vector<Real> orientation__(COORD_DIM*Ncheb, (Iterator<Real>)orientation0.begin()+elem_dsp[i]*COORD_DIM, false);
        for (Long j = 0; j < Ncheb; j++) {
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1_[k*Ncheb+j] = orientation__[j*COORD_DIM+k];
          }
        }
      } else {
        Integer orient_dir = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          e1_[k*Ncheb+0] = 0;
          if (fabs(dx_[k*Ncheb+0]) < fabs(dx_[orient_dir*Ncheb+0])) orient_dir = k;
        }
        e1_[orient_dir*Ncheb+0] = 1;
        for (Long j = 0; j < Ncheb; j++) {
          Vec3 e1_vec, dx_vec;
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1_vec(k,0) = (j==0 ? e1_[k*Ncheb] : e1_[k*Ncheb+j-1]);
            dx_vec(k,0) = dx_[k*Ncheb+j];
          }
          e1_vec = e1_vec - dx_vec*(dot_prod(dx_vec,e1_vec)/dot_prod(dx_vec,dx_vec));
          Real scal = (1.0/sqrt<Real>(dot_prod(e1_vec,e1_vec)));
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1_[k*Ncheb+j] = e1_vec(k,0) * scal;
          }
        }
      }
    }
  }

  template <class Real> Long SlenderElemList<Real>::Size() const {
    return cheb_order.Dim();
  }

  template <class Real> void SlenderElemList<Real>::GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const {
    const Long Nelem = cheb_order.Dim();
    Vector<Long> node_cnt(Nelem), node_dsp(Nelem);
    { // Set node_cnt, node_dsp
      for (Long i = 0; i < Nelem; i++) {
        node_cnt[i] = cheb_order[i] * fourier_order[i];
      }
      if (Nelem) node_dsp[0] = 0;
      omp_par::scan(node_cnt.begin(), node_dsp.begin(), Nelem);
    }

    const Long Nnodes = (Nelem ? node_dsp[Nelem-1]+node_cnt[Nelem-1] : 0);
    if (element_wise_node_cnt) (*element_wise_node_cnt) = node_cnt;
    if (X  != nullptr && X ->Dim() != Nnodes*COORD_DIM) X ->ReInit(Nnodes*COORD_DIM);
    if (Xn != nullptr && Xn->Dim() != Nnodes*COORD_DIM) Xn->ReInit(Nnodes*COORD_DIM);
    for (Long i = 0; i < Nelem; i++) {
      Vector<Real> X_, Xn_;
      if (X  != nullptr) X_ .ReInit(node_cnt[i]*COORD_DIM, X ->begin()+node_dsp[i]*COORD_DIM, false);
      if (Xn != nullptr) Xn_.ReInit(node_cnt[i]*COORD_DIM, Xn->begin()+node_dsp[i]*COORD_DIM, false);
      GetGeom((X==nullptr?nullptr:&X_), (Xn==nullptr?nullptr:&Xn_), nullptr,nullptr,nullptr, CenterlineNodes(cheb_order[i]), sin_theta<Real>(fourier_order[i]), cos_theta<Real>(fourier_order[i]), i);
    }
  }
  template <class Real> void SlenderElemList<Real>::GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const {
    const Long Nelem = cheb_order.Dim();
    Vector<Long> node_cnt(Nelem), node_dsp(Nelem);
    { // Set node_cnt, node_dsp
      for (Long i = 0; i < Nelem; i++) {
        node_cnt[i] = cheb_order[i]*FARFIELD_UPSAMPLE * fourier_order[i]*FARFIELD_UPSAMPLE;
      }
      if (Nelem) node_dsp[0] = 0;
      omp_par::scan(node_cnt.begin(), node_dsp.begin(), Nelem);
    }

    element_wise_node_cnt = node_cnt;
    const Long Nnodes = (Nelem ? node_dsp[Nelem-1]+node_cnt[Nelem-1] : 0);
    if (X       .Dim() != Nnodes*COORD_DIM) X       .ReInit(Nnodes*COORD_DIM);
    if (Xn      .Dim() != Nnodes*COORD_DIM) Xn      .ReInit(Nnodes*COORD_DIM);
    if (wts     .Dim() != Nnodes          ) wts     .ReInit(Nnodes          );
    if (dist_far.Dim() != Nnodes          ) dist_far.ReInit(Nnodes          );
    for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
      Vector<Real>        X_(node_cnt[elem_idx]*COORD_DIM,        X.begin()+node_dsp[elem_idx]*COORD_DIM, false);
      Vector<Real>       Xn_(node_cnt[elem_idx]*COORD_DIM,       Xn.begin()+node_dsp[elem_idx]*COORD_DIM, false);
      Vector<Real>      wts_(node_cnt[elem_idx]          ,      wts.begin()+node_dsp[elem_idx]          , false);
      Vector<Real> dist_far_(node_cnt[elem_idx]          , dist_far.begin()+node_dsp[elem_idx]          , false);

      Vector<Real> dX_ds, dX_dt; // TODO: pre-allocate
      const Long ChebOrder = cheb_order[elem_idx];
      const Long FourierOrder = fourier_order[elem_idx];
      const auto& leg_nds = LegendreQuadRule<Real>(ChebOrder*FARFIELD_UPSAMPLE).first;
      const auto& leg_wts = LegendreQuadRule<Real>(ChebOrder*FARFIELD_UPSAMPLE).second;
      GetGeom(&X_, &Xn_, &wts_, &dX_ds, &dX_dt, leg_nds, sin_theta<Real>(FourierOrder*FARFIELD_UPSAMPLE), cos_theta<Real>(FourierOrder*FARFIELD_UPSAMPLE), elem_idx);

      const Real theta_quad_wt = 2*const_pi<Real>()/(FourierOrder*FARFIELD_UPSAMPLE);
      for (Long i = 0; i < ChebOrder*FARFIELD_UPSAMPLE; i++) { // Set wts *= leg_wts * theta_quad_wt
        Real quad_wt = leg_wts[i] * theta_quad_wt;
        for (Long j = 0; j < FourierOrder*FARFIELD_UPSAMPLE; j++) {
          wts_[i*FourierOrder*FARFIELD_UPSAMPLE+j] *= quad_wt;
        }
      }
      for (Long i = 0; i < node_cnt[elem_idx]; i++) { // Set dist_far
        Real dxds = sqrt<Real>(dX_ds[i*COORD_DIM+0]*dX_ds[i*COORD_DIM+0] + dX_ds[i*COORD_DIM+1]*dX_ds[i*COORD_DIM+1] + dX_ds[i*COORD_DIM+2]*dX_ds[i*COORD_DIM+2])*const_pi<Real>()/2;
        Real dxdt = sqrt<Real>(dX_dt[i*COORD_DIM+0]*dX_dt[i*COORD_DIM+0] + dX_dt[i*COORD_DIM+1]*dX_dt[i*COORD_DIM+1] + dX_dt[i*COORD_DIM+2]*dX_dt[i*COORD_DIM+2])*const_pi<Real>()*2;
        Real h_s = dxds/(ChebOrder*FARFIELD_UPSAMPLE-2);
        Real h_t = dxdt/(FourierOrder*FARFIELD_UPSAMPLE-2);
        dist_far_[i] = -log(tol) * std::max(0.15*h_s, 0.30*h_t); // TODO: use better estimate
      }
    }
  }
  template <class Real> void SlenderElemList<Real>::GetFarFieldDensity(Vector<Real>& Fout, const Vector<Real>& Fin) const {
    constexpr Integer MaxOrderFourier = 128/FARFIELD_UPSAMPLE;
    constexpr Integer MaxOrderCheb = 50/FARFIELD_UPSAMPLE;
    auto compute_Mfourier_upsample_transpose = [MaxOrderFourier]() {
      Vector<Matrix<Real>> M_lst(MaxOrderFourier);
      for (Long k = 1; k < MaxOrderFourier; k++) {
        const Integer FourierOrder = k;
        const Integer FourierModes = FourierOrder/2+1;
        const Matrix<Real>& Mfourier_inv = fourier_matrix_inv<Real>(FourierOrder,FourierModes);
        const Matrix<Real>& Mfourier = fourier_matrix<Real>(FourierModes,FourierOrder*FARFIELD_UPSAMPLE);
        M_lst[k] = (Mfourier_inv * Mfourier).Transpose();
      }
      return M_lst;
    };
    auto compute_Mcheb_upsample_transpose = [MaxOrderCheb]() {
      Vector<Matrix<Real>> M_lst(MaxOrderCheb);
      for (Long k = 0; k < MaxOrderCheb; k++) {
        const Integer ChebOrder = k;
        Matrix<Real> Minterp(ChebOrder, ChebOrder*FARFIELD_UPSAMPLE);
        Vector<Real> Vinterp(ChebOrder*ChebOrder*FARFIELD_UPSAMPLE, Minterp.begin(), false);
        LagrangeInterp<Real>::Interpolate(Vinterp, CenterlineNodes(ChebOrder), LegendreQuadRule<Real>(ChebOrder*FARFIELD_UPSAMPLE).first);
        M_lst[k] = Minterp.Transpose();
      }
      return M_lst;
    };
    static const Vector<Matrix<Real>> Mfourier_transpose = compute_Mfourier_upsample_transpose();
    static const Vector<Matrix<Real>> Mcheb_transpose = compute_Mcheb_upsample_transpose();

    const Long Nelem = cheb_order.Dim();
    Vector<Long> node_cnt(Nelem), node_dsp(Nelem);
    { // Set node_cnt, node_dsp
      for (Long i = 0; i < Nelem; i++) {
        node_cnt[i] = cheb_order[i] * fourier_order[i];
      }
      if (Nelem) node_dsp[0] = 0;
      omp_par::scan(node_cnt.begin(), node_dsp.begin(), Nelem);
    }

    const Long Nnodes = (Nelem ? node_dsp[Nelem-1]+node_cnt[Nelem-1] : 0);
    const Long density_dof = (Nnodes ? Fin.Dim() / Nnodes : 0);
    SCTL_ASSERT(Fin.Dim() == Nnodes * density_dof);

    if (Fout.Dim() != Nnodes*(FARFIELD_UPSAMPLE*FARFIELD_UPSAMPLE) * density_dof) {
      Fout.ReInit(Nnodes*(FARFIELD_UPSAMPLE*FARFIELD_UPSAMPLE) * density_dof);
    }
    for (Long i = 0; i < Nelem; i++) {
      const Integer ChebOrder = cheb_order[i];
      const Integer FourierOrder = fourier_order[i];

      const auto& Mfourier_ = Mfourier_transpose[FourierOrder];
      const Matrix<Real> Fin_(ChebOrder, FourierOrder*density_dof, (Iterator<Real>)Fin.begin()+node_dsp[i]*density_dof, false);
      Matrix<Real> F0_(ChebOrder, FourierOrder*FARFIELD_UPSAMPLE*density_dof);
      for (Long l = 0; l < ChebOrder; l++) { // Set F0
        for (Long j0 = 0; j0 < FourierOrder*FARFIELD_UPSAMPLE; j0++) {
          for (Long k = 0; k < density_dof; k++) {
            Real f = 0;
            for (Long j1 = 0; j1 < FourierOrder; j1++) {
              f += Fin_[l][j1*density_dof+k] * Mfourier_[j0][j1];
            }
            F0_[l][j0*density_dof+k] = f;
          }
        }
      }

      Matrix<Real> Fout_(ChebOrder*FARFIELD_UPSAMPLE, FourierOrder*FARFIELD_UPSAMPLE*density_dof, Fout.begin()+node_dsp[i]*FARFIELD_UPSAMPLE*FARFIELD_UPSAMPLE*density_dof, false);
      Matrix<Real>::GEMM(Fout_, Mcheb_transpose[ChebOrder], F0_);
    }
  }
  template <class Real> void SlenderElemList<Real>::FarFieldDensityOperatorTranspose(Matrix<Real>& Mout, const Matrix<Real>& Min, const Long elem_idx) const {
    constexpr Integer MaxOrderFourier = 128/FARFIELD_UPSAMPLE;
    constexpr Integer MaxOrderCheb = 50/FARFIELD_UPSAMPLE;
    auto compute_Mfourier_upsample = [MaxOrderFourier]() {
      Vector<Matrix<Real>> M_lst(MaxOrderFourier);
      for (Long k = 1; k < MaxOrderFourier; k++) {
        const Integer FourierOrder = k;
        const Integer FourierModes = FourierOrder/2+1;
        const Matrix<Real>& Mfourier_inv = fourier_matrix_inv<Real>(FourierOrder,FourierModes);
        const Matrix<Real>& Mfourier = fourier_matrix<Real>(FourierModes,FourierOrder*FARFIELD_UPSAMPLE);
        M_lst[k] = Mfourier_inv * Mfourier;
      }
      return M_lst;
    };
    auto compute_Mcheb_upsample = [MaxOrderCheb]() {
      Vector<Matrix<Real>> M_lst(MaxOrderCheb);
      for (Long k = 0; k < MaxOrderCheb; k++) {
        const Integer ChebOrder = k;
        Matrix<Real> Minterp(ChebOrder, ChebOrder*FARFIELD_UPSAMPLE);
        Vector<Real> Vinterp(ChebOrder*ChebOrder*FARFIELD_UPSAMPLE, Minterp.begin(), false);
        LagrangeInterp<Real>::Interpolate(Vinterp, CenterlineNodes(ChebOrder), LegendreQuadRule<Real>(ChebOrder*FARFIELD_UPSAMPLE).first);
        M_lst[k] = Minterp;
      }
      return M_lst;
    };
    static const Vector<Matrix<Real>> Mfourier = compute_Mfourier_upsample();
    static const Vector<Matrix<Real>> Mcheb = compute_Mcheb_upsample();

    const Integer ChebOrder = cheb_order[elem_idx];
    const Integer FourierOrder = fourier_order[elem_idx];

    const Long N = Min.Dim(1);
    const Long density_dof = Min.Dim(0) / (ChebOrder*FARFIELD_UPSAMPLE*FourierOrder*FARFIELD_UPSAMPLE);
    SCTL_ASSERT(Min.Dim(0) == ChebOrder*FARFIELD_UPSAMPLE*FourierOrder*FARFIELD_UPSAMPLE*density_dof);
    if (Mout.Dim(0) != ChebOrder*FourierOrder*density_dof || Mout.Dim(1) != N) {
      Mout.ReInit(ChebOrder*FourierOrder*density_dof,N);
      Mout.SetZero();
    }

    Matrix<Real> Mtmp(ChebOrder*FARFIELD_UPSAMPLE, FourierOrder*density_dof*N);
    const Matrix<Real> Min_(ChebOrder*FARFIELD_UPSAMPLE, FourierOrder*FARFIELD_UPSAMPLE*density_dof*N, (Iterator<Real>)Min.begin(), false);
    if (FARFIELD_UPSAMPLE != 1) { // Appyl Mfourier // TODO: optimize
      const auto& Mfourier_ = Mfourier[FourierOrder];
      for (Long l = 0; l < ChebOrder*FARFIELD_UPSAMPLE; l++) {
        for (Long j0 = 0; j0 < FourierOrder; j0++) {
          for (Long k = 0; k < density_dof*N; k++) {
            Real f_tmp = 0;
            for (Long j1 = 0; j1 < FourierOrder*FARFIELD_UPSAMPLE; j1++) {
              f_tmp += Min_[l][j1*density_dof*N+k] * Mfourier_[j0][j1];
            }
            Mtmp[l][j0*density_dof*N+k] = f_tmp;
          }
        }
      }
    }else{
      Mtmp.ReInit(ChebOrder*FARFIELD_UPSAMPLE, FourierOrder*density_dof*N, (Iterator<Real>)Min.begin(), false);
    }

    Matrix<Real> Mout_(ChebOrder, FourierOrder*density_dof*N, Mout.begin(), false);
    Matrix<Real>::GEMM(Mout_, Mcheb[ChebOrder], Mtmp);
  }

  template <class Real> template <class Kernel> void SlenderElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    const auto& elem_lst = *dynamic_cast<const SlenderElemList*>(self);
    const Long Nelem = elem_lst.cheb_order.Dim();

    if (M_lst.Dim() != Nelem) M_lst.ReInit(Nelem);
    if (trg_dot_prod) {
      //#pragma omp parallel for schedule(static)
      for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
        if      (tol <= pow<15,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<15,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<14,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<14,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<13,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<13,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<12,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<12,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<11,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<11,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<10,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<10,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 9,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 9,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 8,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 8,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 7,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 7,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 6,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 6,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 5,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 5,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 4,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 4,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 3,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 3,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 2,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 2,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 1,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 1,true,Kernel>(ker, elem_idx);
        else                                     M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 0,true,Kernel>(ker, elem_idx);
      }
    } else {
      //#pragma omp parallel for schedule(static)
      for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
        if      (tol <= pow<15,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<15,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<14,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<14,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<13,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<13,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<12,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<12,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<11,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<11,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<10,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<10,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 9,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 9,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 8,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 8,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 7,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 7,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 6,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 6,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 5,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 5,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 4,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 4,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 3,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 3,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 2,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 2,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 1,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 1,false,Kernel>(ker, elem_idx);
        else                                     M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 0,false,Kernel>(ker, elem_idx);
      }
    }
  }
  template <class Real> template <class Kernel> void SlenderElemList<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    const auto& elem_lst = *dynamic_cast<const SlenderElemList*>(self);
    if (normal_trg.Dim()) {
      if      (tol <= pow<15,Real>((Real)0.1)) elem_lst.template NearInteracHelper<15,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<14,Real>((Real)0.1)) elem_lst.template NearInteracHelper<14,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<13,Real>((Real)0.1)) elem_lst.template NearInteracHelper<13,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<12,Real>((Real)0.1)) elem_lst.template NearInteracHelper<12,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<11,Real>((Real)0.1)) elem_lst.template NearInteracHelper<11,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<10,Real>((Real)0.1)) elem_lst.template NearInteracHelper<10,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 9,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 9,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 8,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 8,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 7,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 7,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 6,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 6,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 5,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 5,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 4,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 4,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 3,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 3,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 2,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 2,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 1,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 1,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else                                     elem_lst.template NearInteracHelper< 0,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
    } else {
      if      (tol <= pow<15,Real>((Real)0.1)) elem_lst.template NearInteracHelper<15,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<14,Real>((Real)0.1)) elem_lst.template NearInteracHelper<14,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<13,Real>((Real)0.1)) elem_lst.template NearInteracHelper<13,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<12,Real>((Real)0.1)) elem_lst.template NearInteracHelper<12,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<11,Real>((Real)0.1)) elem_lst.template NearInteracHelper<11,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<10,Real>((Real)0.1)) elem_lst.template NearInteracHelper<10,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 9,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 9,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 8,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 8,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 7,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 7,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 6,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 6,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 5,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 5,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 4,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 4,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 3,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 3,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 2,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 2,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 1,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 1,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else                                     elem_lst.template NearInteracHelper< 0,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
    }
  }
  template <class Real> template <Integer digits, bool trg_dot_prod, class Kernel> void SlenderElemList<Real>::NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx) const {
    using Vec3 = Tensor<Real,true,COORD_DIM,1>;
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1 = Kernel::TrgDim()/(trg_dot_prod?COORD_DIM:1);
    //const Integer digits = (Integer)(log(tol)/log(0.1)+0.5);
    static constexpr Real tol = pow<digits,Real>((Real)0.1);

    const Integer ChebOrder = cheb_order[elem_idx];
    const Integer FourierOrder = fourier_order[elem_idx];
    const Integer FourierModes = FourierOrder/2+1;
    const Matrix<Real> M_fourier_inv = fourier_matrix_inv_transpose<Real>(FourierOrder,FourierModes);

    const Vector<Real>  coord(COORD_DIM*ChebOrder,(Iterator<Real>)this-> coord.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real>     dx(COORD_DIM*ChebOrder,(Iterator<Real>)this->    dx.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real>    d2x(COORD_DIM*ChebOrder,(Iterator<Real>)this->   d2x.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real> radius(        1*ChebOrder,(Iterator<Real>)this->radius.begin()+          elem_dsp[elem_idx],false);
    const Vector<Real>     dr(        1*ChebOrder,(Iterator<Real>)this->    dr.begin()+          elem_dsp[elem_idx],false);
    const Vector<Real>     e1(COORD_DIM*ChebOrder,(Iterator<Real>)this->    e1.begin()+COORD_DIM*elem_dsp[elem_idx],false);

    const Long Ntrg = Xtrg.Dim() / COORD_DIM;
    if (M.Dim(0) != ChebOrder*FourierOrder*KDIM0 || M.Dim(1) != Ntrg*KDIM1) {
      M.ReInit(ChebOrder*FourierOrder*KDIM0, Ntrg*KDIM1);
    }

    for (Long i = 0; i < Ntrg; i++) {
      const Vec3 Xt((Iterator<Real>)Xtrg.begin()+i*COORD_DIM);
      const Vec3 n_trg = (trg_dot_prod ? Vec3((Iterator<Real>)normal_trg.begin()+i*COORD_DIM) : Vec3((Real)0));
      Matrix<Real> M_modal(ChebOrder, KDIM0*KDIM1*FourierModes*2);
      { // Set M_modal
        Vector<Real> quad_nds, quad_wts; // Quadrature rule in s
        auto adap_quad_rule = [&ChebOrder,&radius,&coord,&dx](Vector<Real>& quad_nds, Vector<Real>& quad_wts, const Vec3& x_trg) {
          const Long LegQuadOrder = (Long)(-0.24*log(tol)*const_pi<Real>()/2)+1;
          const auto& leg_nds = LegendreQuadRule<Real>(LegQuadOrder).first;
          const auto& leg_wts = LegendreQuadRule<Real>(LegQuadOrder).second;
          auto adap_ref = [&LegQuadOrder,&leg_nds,&leg_wts](Vector<Real>& nds, Vector<Real>& wts, Real a, Real b, Integer levels) {
            if (nds.Dim() != levels * LegQuadOrder) nds.ReInit(levels*LegQuadOrder);
            if (wts.Dim() != levels * LegQuadOrder) wts.ReInit(levels*LegQuadOrder);
            Vector<Real> nds_(nds.Dim(), nds.begin(), false);
            Vector<Real> wts_(wts.Dim(), wts.begin(), false);

            while (levels) {
              Vector<Real> nds0(LegQuadOrder, nds_.begin(), false);
              Vector<Real> wts0(LegQuadOrder, wts_.begin(), false);
              Vector<Real> nds1((levels-1)*LegQuadOrder, nds_.begin()+LegQuadOrder, false);
              Vector<Real> wts1((levels-1)*LegQuadOrder, wts_.begin()+LegQuadOrder, false);

              Real end_point = (levels==1 ? b : (a+b)*0.5);
              nds0 = leg_nds * (end_point-a) + a;
              wts0 = leg_wts * fabs<Real>(end_point-a);

              nds_.Swap(nds1);
              wts_.Swap(wts1);
              a = end_point;
              levels--;
            }
          };

          // TODO: develop special quadrature rule instead of adaptive integration
          if (0) { // adaptive/dyadic refinement on element ends
            const Integer levels = 6;
            quad_nds.ReInit(2*levels*LegQuadOrder);
            quad_wts.ReInit(2*levels*LegQuadOrder);
            Vector<Real> nds0(levels*LegQuadOrder,quad_nds.begin(),false);
            Vector<Real> wts0(levels*LegQuadOrder,quad_wts.begin(),false);
            Vector<Real> nds1(levels*LegQuadOrder,quad_nds.begin()+levels*LegQuadOrder,false);
            Vector<Real> wts1(levels*LegQuadOrder,quad_wts.begin()+levels*LegQuadOrder,false);
            adap_ref(nds0, wts0, 0.5, 0.0, levels);
            adap_ref(nds1, wts1, 0.5, 1.0, levels);
          } else { // dyadic refinement near target point
            Real dist_min, s_min, dxds;
            { // Set dist_min, s_min, dxds
              auto get_dist = [&ChebOrder,&radius,&coord,&dx] (const Vec3& x_trg, Real s) -> Real {
                Vector<Real> interp_wts(ChebOrder); // TODO: pre-allocate
                LagrangeInterp<Real>::Interpolate(interp_wts, CenterlineNodes(ChebOrder), Vector<Real>(1,Ptr2Itr<Real>(&s,1),false));

                Real r0 = 0;
                Vec3 x0, dx_ds0;
                for (Long i = 0; i < COORD_DIM; i++) {
                  x0(i,0) = 0;
                  dx_ds0(i,0) = 0;
                }
                for (Long i = 0; i < ChebOrder; i++) {
                  r0 += radius[i] * interp_wts[i];
                  x0(0,0) += coord[0*ChebOrder+i] * interp_wts[i];
                  x0(1,0) += coord[1*ChebOrder+i] * interp_wts[i];
                  x0(2,0) += coord[2*ChebOrder+i] * interp_wts[i];
                  dx_ds0(0,0) += dx[0*ChebOrder+i] * interp_wts[i];
                  dx_ds0(1,0) += dx[1*ChebOrder+i] * interp_wts[i];
                  dx_ds0(2,0) += dx[2*ChebOrder+i] * interp_wts[i];
                }
                Vec3 dx = x0 - x_trg;
                Vec3 n0 = dx_ds0 * sqrt<Real>(1/dot_prod(dx_ds0, dx_ds0));
                Real dz = dot_prod(dx, n0);
                Vec3 dr = dx - n0*dz;
                Real dR = sqrt<Real>(dot_prod(dr,dr)) - r0;
                return sqrt<Real>(dR*dR + dz*dz);
              };
              StaticArray<Real,2> dist;
              StaticArray<Real,2> s_val{0,1};
              dist[0] = get_dist(x_trg, s_val[0]);
              dist[1] = get_dist(x_trg, s_val[1]);
              for (Long i = 0; i < 20; i++) { // Binary search: set dist, s_val // TODO: use Netwon's method
                Real ss = (s_val[0] + s_val[1]) * 0.5;
                Real dd = get_dist(x_trg, ss);
                if (dist[0] > dist[1]) {
                  dist[0] = dd;
                  s_val[0] = ss;
                } else {
                  dist[1] = dd;
                  s_val[1] = ss;
                }
              }
              if (dist[0] < dist[1]) { // Set dis_min, s_min
                dist_min = dist[0];
                s_min = s_val[0];
              } else {
                dist_min = dist[1];
                s_min = s_val[1];
              }
              { // Set dx_ds;
                Vector<Real> interp_wts(ChebOrder); // TODO: pre-allocate
                LagrangeInterp<Real>::Interpolate(interp_wts, CenterlineNodes(ChebOrder), Vector<Real>(1,Ptr2Itr<Real>(&s_min,1),false));

                Vec3 dxds_vec;
                for (Long i = 0; i < COORD_DIM; i++) {
                  dxds_vec(i,0) = 0;
                }
                for (Long i = 0; i < ChebOrder; i++) {
                  dxds_vec(0,0) += dx[0*ChebOrder+i] * interp_wts[i];
                  dxds_vec(1,0) += dx[1*ChebOrder+i] * interp_wts[i];
                  dxds_vec(2,0) += dx[2*ChebOrder+i] * interp_wts[i];
                }
                dxds = sqrt<Real>(dot_prod(dxds_vec,dxds_vec))*const_pi<Real>()/2;
              }
            }
            Real h0 =   (s_min)*dxds/(LegQuadOrder-1);
            Real h1 = (1-s_min)*dxds/(LegQuadOrder-1);
            Real dist_far0 = -0.25 * log(tol)*h0; // TODO: use better estimate
            Real dist_far1 = -0.25 * log(tol)*h1; // TODO: use better estimate
            Integer adap_levels0 = (s_min==0 ? 0 : std::max<Integer>(0,(Integer)(log(dist_far0/dist_min)/log(2.0)+0.5))+1);
            Integer adap_levels1 = (s_min==1 ? 0 : std::max<Integer>(0,(Integer)(log(dist_far1/dist_min)/log(2.0)+0.5))+1);

            Long N0 = adap_levels0 * LegQuadOrder;
            Long N1 = adap_levels1 * LegQuadOrder;
            quad_nds.ReInit(N0+N1);
            quad_wts.ReInit(N0+N1);
            Vector<Real> nds0(N0, quad_nds.begin(), false);
            Vector<Real> wts0(N0, quad_wts.begin(), false);
            Vector<Real> nds1(N1, quad_nds.begin()+N0, false);
            Vector<Real> wts1(N1, quad_wts.begin()+N0, false);
            adap_ref(nds0, wts0, 0, s_min, adap_levels0);
            adap_ref(nds1, wts1, 1, s_min, adap_levels1);
          }
        };
        adap_quad_rule(quad_nds, quad_wts, Xt);

        Matrix<Real> Minterp_quad_nds;
        { // Set Minterp_quad_nds
          Minterp_quad_nds.ReInit(ChebOrder, quad_nds.Dim());
          Vector<Real> Vinterp_quad_nds(ChebOrder*quad_nds.Dim(), Minterp_quad_nds.begin(), false);
          LagrangeInterp<Real>::Interpolate(Vinterp_quad_nds, CenterlineNodes(ChebOrder), quad_nds);
        }

        Vec3 x_trg = Xt;
        Matrix<Real> r_src, dr_src, x_src, dx_src, d2x_src, e1_src, e2_src, de1_src, de2_src;
        r_src  .ReInit(        1,quad_nds.Dim());
        dr_src .ReInit(        1,quad_nds.Dim());
        x_src  .ReInit(COORD_DIM,quad_nds.Dim());
        dx_src .ReInit(COORD_DIM,quad_nds.Dim());
        d2x_src.ReInit(COORD_DIM,quad_nds.Dim());
        e1_src .ReInit(COORD_DIM,quad_nds.Dim());
        e2_src .ReInit(COORD_DIM,quad_nds.Dim());
        de1_src.ReInit(COORD_DIM,quad_nds.Dim());
        de2_src.ReInit(COORD_DIM,quad_nds.Dim());
        { // Set x_src, x_trg (improve numerical stability)
          Matrix<Real> x_nodes(COORD_DIM,ChebOrder, (Iterator<Real>)coord.begin(), true);
          for (Long j = 0; j < ChebOrder; j++) {
            for (Integer k = 0; k < COORD_DIM; k++) {
              x_nodes[k][j] -= x_trg(k,0);
            }
          }
          Matrix<Real>::GEMM(  x_src, x_nodes, Minterp_quad_nds);
          for (Integer k = 0; k < COORD_DIM; k++) {
            x_trg(k,0) = 0;
          }
        }
        //Matrix<Real>::GEMM(  x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>) coord.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM( dx_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    dx.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM(d2x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)   d2x.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM(  r_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)radius.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM( dr_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)    dr.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM( e1_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    e1.begin(),false), Minterp_quad_nds);
        for (Long j = 0; j < quad_nds.Dim(); j++) { // Set e2_src
          Vec3 e1, dx, d2x;
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1(k,0) = e1_src[k][j];
            dx(k,0) = dx_src[k][j];
            d2x(k,0) = d2x_src[k][j];
          }
          Real inv_dx2 = 1/dot_prod(dx,dx);
          e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
          e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

          Vec3 e2 = cross_prod(e1, dx);
          e2 = e2 * (1/sqrt<Real>(dot_prod(e2,e2)));
          Vec3 de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
          Vec3 de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1_src[k][j] = e1(k,0);
            e2_src[k][j] = e2(k,0);
            de1_src[k][j] = de1(k,0);
            de2_src[k][j] = de2(k,0);
          }
        }

        const Vec3 y_trg = x_trg;
        Matrix<Real> M_tor(quad_nds.Dim(), KDIM0*KDIM1*FourierModes*2); // TODO: pre-allocate
        toroidal_greens_fn_batched<digits,ModalUpsample,trg_dot_prod>(M_tor, y_trg, n_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, e2_src, de1_src, de2_src, ker, FourierModes);

        for (Long ii = 0; ii < M_tor.Dim(0); ii++) {
          for (Long jj = 0; jj < M_tor.Dim(1); jj++) {
            M_tor[ii][jj] *= quad_wts[ii];
          }
        }
        Matrix<Real>::GEMM(M_modal, Minterp_quad_nds, M_tor);
      }

      Matrix<Real> M_nodal(ChebOrder, KDIM0*KDIM1*FourierOrder);
      { // Set M_nodal
        Matrix<Real> M_nodal_(ChebOrder*KDIM0*KDIM1, FourierOrder, M_nodal.begin(), false);
        const Matrix<Real> M_modal_(ChebOrder*KDIM0*KDIM1, FourierModes*2, M_modal.begin(), false);
        Matrix<Real>::GEMM(M_nodal_, M_modal_, M_fourier_inv);
      }

      { // Set M
        for (Integer i0 = 0; i0 < ChebOrder; i0++) {
          for (Integer i1 = 0; i1 < FourierOrder; i1++) {
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
              for (Integer k1 = 0; k1 < KDIM1; k1++) {
                M[(i0*FourierOrder+i1)*KDIM0+k0][i*KDIM1+k1] = M_nodal[i0][(k0*KDIM1+k1)*FourierOrder+i1] * ker.template uKerScaleFactor<Real>();
              }
            }
          }
        }
      }
    }
  }

  template <class Real> const Vector<Real>& SlenderElemList<Real>::CenterlineNodes(Integer Order) {
    return ChebQuadRule<Real>::nds(Order);
  }

  template <class Real> void SlenderElemList<Real>::Write(const std::string& fname, const Comm& comm) const {
    auto allgather = [&comm](Vector<Real>& v_out, const Vector<Real>& v_in) {
      const Long Nproc = comm.Size();
      StaticArray<Long,1> len{v_in.Dim()};
      Vector<Long> cnt(Nproc), dsp(Nproc);
      comm.Allgather(len+0, 1, cnt.begin(), 1); dsp = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), Nproc);

      v_out.ReInit(dsp[Nproc-1]+cnt[Nproc-1]);
      comm.Allgatherv(v_in.begin(), v_in.Dim(), v_out.begin(), cnt.begin(), dsp.begin());
    };
    auto allgatherl = [&comm](Vector<Long>& v_out, const Vector<Long>& v_in) {
      const Long Nproc = comm.Size();
      StaticArray<Long,1> len{v_in.Dim()};
      Vector<Long> cnt(Nproc), dsp(Nproc);
      comm.Allgather(len+0, 1, cnt.begin(), 1); dsp = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), Nproc);

      v_out.ReInit(dsp[Nproc-1]+cnt[Nproc-1]);
      comm.Allgatherv(v_in.begin(), v_in.Dim(), v_out.begin(), cnt.begin(), dsp.begin());
    };

    Vector<Real> radius_, coord_, e1_;
    allgather(radius_, radius);
    allgather( coord_,  coord);
    allgather(    e1_,     e1);

    Vector<Long> cheb_order_, elem_dsp_, fourier_order_;
    allgatherl(   cheb_order_, cheb_order   );
    allgatherl(fourier_order_, fourier_order);
    elem_dsp_.ReInit(cheb_order_.Dim()); elem_dsp_ = 0;
    omp_par::scan(cheb_order_.begin(), elem_dsp_.begin(), cheb_order_.Dim());


    if (!comm.Rank()) return;
    const Integer precision = 18, width = 26;
    std::ofstream file;
    file.open(fname, std::ofstream::out | std::ofstream::trunc);
    if (!file.good()) {
      std::cout << "Unable to open file for writing:" << fname << '\n';
    }

    // Header
    file<<"#";
    file<<std::setw(width-1)<<"X";
    file<<std::setw(width)<<"Y";
    file<<std::setw(width)<<"Z";
    file<<std::setw(width)<<"r";
    file<<std::setw(width)<<"orient-x";
    file<<std::setw(width)<<"orient-y";
    file<<std::setw(width)<<"orient-z";
    file<<std::setw(width)<<"ChebOrder";
    file<<std::setw(width)<<"FourierOrder";
    file<<'\n';

    file<<std::scientific<<std::setprecision(precision);
    for (Long i = 0; i < cheb_order_.Dim(); i++) {
      for (Long j = 0; j < cheb_order_[i]; j++) {
        for (Integer k = 0; k < COORD_DIM; k++) {
          file<<std::setw(width)<<coord_[elem_dsp_[i]*COORD_DIM + k*cheb_order_[i]+j];
        }
        file<<std::setw(width)<<radius_[elem_dsp_[i] + j];
        for (Integer k = 0; k < COORD_DIM; k++) {
          file<<std::setw(width)<<e1_[elem_dsp_[i]*COORD_DIM + k*cheb_order_[i]+j];
        }
        if (!j) {
          file<<std::setw(width)<<cheb_order_[i];
          file<<std::setw(width)<<fourier_order_[i];
        }
        file<<"\n";
      }
    }
    file.close();
  }
  template <class Real> void SlenderElemList<Real>::Read(const std::string& fname, const Comm& comm) {
    std::ifstream file;
    file.open(fname, std::ifstream::in);
    if (!file.good()) {
      std::cout << "Unable to open file for reading:" << fname << '\n';
    }

    std::string line;
    Vector<Real> coord_, radius_, e1_;
    Vector<Integer> cheb_order_, fourier_order_;
    while (std::getline(file, line)) { // Set coord_, radius_, e1_, cheb_order_, fourier_order_
      size_t first_char_pos = line.find_first_not_of(' ');
      if (first_char_pos == std::string::npos || line[first_char_pos] == '#') continue;

      std::istringstream iss(line);
      for (Integer k = 0; k < COORD_DIM; k++) { // read coord_
        Real a;
        iss>>a;
        SCTL_ASSERT(!iss.fail());
        coord_.PushBack(a);
      }
      { // read radius_
        Real a;
        iss>>a;
        SCTL_ASSERT(!iss.fail());
        radius_.PushBack(a);
      }
      for (Integer k = 0; k < COORD_DIM; k++) { // read e1_
        Real a;
        iss>>a;
        SCTL_ASSERT(!iss.fail());
        e1.PushBack(a);
      }

      Integer ChebOrder, FourierOrder;
      if (iss>>ChebOrder>>FourierOrder) {
        cheb_order_.PushBack(ChebOrder);
        fourier_order_.PushBack(FourierOrder);
      } else {
        cheb_order_.PushBack(-1);
        fourier_order_.PushBack(-1);
      }
    }
    file.close();

    Long offset = 0;
    Vector<Integer> cheb_order__, fourier_order__;
    while (offset < cheb_order_.Dim()) { // Set cheb_order__, fourier_order__
      Integer ChebOrder = cheb_order_[offset];
      Integer FourierOrder = fourier_order_[offset];
      for (Integer j = 1; j < ChebOrder; j++) {
        SCTL_ASSERT(cheb_order_[offset+j] == ChebOrder || cheb_order_[offset+j] == -1);
        SCTL_ASSERT(fourier_order_[offset+j] == FourierOrder || fourier_order_[offset+j] == -1);
      }
      cheb_order__.PushBack(ChebOrder);
      fourier_order__.PushBack(FourierOrder);
      offset += ChebOrder;
    }
    { // Distribute across processes and init SlenderElemList
      const Long Np = comm.Size();
      const Long pid = comm.Rank();
      const Long Nelem = cheb_order__.Dim();

      const Long i0 = Nelem*(pid+0)/Np;
      const Long i1 = Nelem*(pid+1)/Np;

      Vector<Integer> cheb_order, fourier_order;
      cheb_order.ReInit(i1-i0, cheb_order__.begin()+i0, false);
      fourier_order.ReInit(i1-i0, fourier_order__.begin()+i0, false);

      Vector<Integer> elem_offset(Nelem+1); elem_offset = 0;
      omp_par::scan(cheb_order__.begin(), elem_offset.begin(), Nelem);
      elem_offset[Nelem] = (Nelem ? elem_offset[Nelem-1] + cheb_order__[Nelem-1] : 0);
      const Long j0 = elem_offset[i0];
      const Long j1 = elem_offset[i1];

      Vector<Real> radius, coord, e1;
      radius.ReInit((j1-j0), radius_.begin()+j0, false);
      coord.ReInit((j1-j0)*COORD_DIM, coord_.begin()+j0*COORD_DIM, false);
      if (e1_.Dim()) e1.ReInit((j1-j0)*COORD_DIM, e1_.begin()+j0*COORD_DIM, false);

      Init(cheb_order, fourier_order, coord, radius, e1);
    }
  }

  template <class Real> void SlenderElemList<Real>::GetVTUData(VTUData& vtu_data, const Vector<Real>& F, const Long elem_idx) const {
    if (elem_idx == -1) {
      const Long Nelem = cheb_order.Dim();
      Long dof = 0, offset = 0;
      if (F.Dim()) { // Set dof
        Long Nnodes = 0;
        for (Long i = 0; i < Nelem; i++) {
          Nnodes += cheb_order[i] * fourier_order[i];
        }
        dof = F.Dim() / Nnodes;
        SCTL_ASSERT(F.Dim() == Nnodes * dof);
      }
      for (Long i = 0; i < Nelem; i++) {
        const Vector<Real> F_(cheb_order[i]*fourier_order[i]*dof, (Iterator<Real>)F.begin()+offset, false);
        GetVTUData(vtu_data, F_, i);
        offset += F_.Dim();
      }
      return;
    }

    Vector<Real> X;
    const Integer ChebOrder = cheb_order[elem_idx];
    const Integer FourierOrder = fourier_order[elem_idx];
    GetGeom(&X,nullptr,nullptr,nullptr,nullptr, CenterlineNodes(ChebOrder), sin_theta<Real>(FourierOrder), cos_theta<Real>(FourierOrder), elem_idx);

    Long point_offset = vtu_data.coord.Dim() / COORD_DIM;
    for (const auto& x : X) vtu_data.coord.PushBack((VTUData::VTKReal)x);
    for (const auto& f : F) vtu_data.value.PushBack((VTUData::VTKReal)f);
    for (Long i = 0; i < ChebOrder-1; i++) {
      for (Long j = 0; j <= FourierOrder; j++) {
        vtu_data.connect.PushBack(point_offset + (i+0)*FourierOrder+(j%FourierOrder));
        vtu_data.connect.PushBack(point_offset + (i+1)*FourierOrder+(j%FourierOrder));
      }
      vtu_data.offset.PushBack(vtu_data.connect.Dim());
      vtu_data.types.PushBack(6);
    }
  }
  template <class Real> void SlenderElemList<Real>::WriteVTK(const std::string& fname, const Vector<Real>& F, const Comm& comm) const {
    VTUData vtu_data;
    GetVTUData(vtu_data, F);
    vtu_data.WriteVTK(fname, comm);
  }

  template <class Real> template <class Kernel> void SlenderElemList<Real>::test(const Comm& comm, Real tol) {
    sctl::Profile::Enable(false);
    const Long pid = comm.Rank();
    const Long Np = comm.Size();

    SlenderElemList<Real> elem_lst0;
    //elem_lst0.Read("data/geom.data"); // Read geometry from file
    if (1) { // Initialize elem_lst0 in code
      const Long Nelem = 16;
      const Long ChebOrder = 10;
      const Long FourierOrder = 8;

      Vector<Real> coord, radius;
      Vector<Long> cheb_order, fourier_order;
      const Long k0 = (Nelem*(pid+0))/Np;
      const Long k1 = (Nelem*(pid+1))/Np;
      for (Long k = k0; k < k1; k++) {
        cheb_order.PushBack(ChebOrder);
        fourier_order.PushBack(FourierOrder);
        const auto& nds = SlenderElemList<Real>::CenterlineNodes(ChebOrder);
        for (Long i = 0; i < nds.Dim(); i++) {
          Real theta = 2*const_pi<Real>()*(k+nds[i])/Nelem;
          coord.PushBack(cos<Real>(theta));
          coord.PushBack(sin<Real>(theta));
          coord.PushBack(0.1*sin<Real>(2*theta));
          radius.PushBack(0.01*(2+sin<Real>(theta+sqrt<Real>(2))));
        }
      }
      elem_lst0.Init(cheb_order, fourier_order, coord, radius);
    }

    Kernel ker_fn;
    BoundaryIntegralOp<Real,Kernel> BIOp(ker_fn, false, comm);
    BIOp.AddElemList(elem_lst0);
    BIOp.SetAccuracy(tol);

    // Warm-up run
    Vector<Real> F(BIOp.Dim(0)), U; F = 1;
    BIOp.ComputePotential(U,F);
    BIOp.ClearSetup();
    U = 0;

    Profile::Enable(true);
    Profile::Tic("Setup+Eval", &comm, true);
    BIOp.ComputePotential(U,F);
    Profile::Toc();

    Vector<Real> Uerr = U + 0.5;
    elem_lst0.WriteVTK("Uerr_", Uerr, comm); // Write VTK
    { // Print error
      StaticArray<Real,2> max_err{0,0};
      for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], fabs(x));
      comm.Allreduce(max_err+0, max_err+1, 1, Comm::CommOp::MAX);
      if (!pid) std::cout<<"Error = "<<max_err[1]<<'\n';
    }
    Profile::Enable(false);
    Profile::print(&comm);
  }
  template <class Real> void SlenderElemList<Real>::test_greens_identity(const Comm& comm, Real tol) {
    using KerSL = Laplace3D_FxU;
    using KerDL = Laplace3D_DxU;
    using KerGrad = Laplace3D_FxdU;

    const auto concat_vecs = [](Vector<Real>& v, const Vector<Vector<Real>>& vec_lst) {
      const Long N = vec_lst.Dim();
      Vector<Long> dsp(N+1); dsp[0] = 0;
      for (Long i = 0; i < N; i++) {
        dsp[i+1] = dsp[i] + vec_lst[i].Dim();
      }
      if (v.Dim() != dsp[N]) v.ReInit(dsp[N]);
      for (Long i = 0; i < N; i++) {
        Vector<Real> v_(vec_lst[i].Dim(), v.begin()+dsp[i], false);
        v_ = vec_lst[i];
      }
    };
    auto loop_geom = [](Real& x, Real& y, Real& z, Real& r, const Real theta){
      x = cos<Real>(theta);
      y = sin<Real>(theta);
      z = 0.1*sin<Real>(theta-sqrt<Real>(2));
      r = 0.01*(2+sin<Real>(theta+sqrt<Real>(2)));
    };
    sctl::Profile::Enable(false);
    const Long pid = comm.Rank();
    const Long Np = comm.Size();

    SlenderElemList elem_lst0;
    SlenderElemList elem_lst1;
    { // Set elem_lst0, elem_lst1
      const Long Nelem = 16;
      const Long idx0 = Nelem*(pid+0)/Np;
      const Long idx1 = Nelem*(pid+1)/Np;

      Vector<Real> coord0, radius0;
      Vector<Long> cheb_order0, fourier_order0;
      for (Long k = idx0; k < idx1; k++) { // Init elem_lst0
      const Integer ChebOrder = 8, FourierOrder = 14;
        const auto& nds = CenterlineNodes(ChebOrder);
        for (Long i = 0; i < nds.Dim(); i++) {
          Real x, y, z, r;
          loop_geom(x, y, z, r, const_pi<Real>()*(k+nds[i])/Nelem);
          coord0.PushBack(x);
          coord0.PushBack(y);
          coord0.PushBack(z);
          radius0.PushBack(r);
        }
        cheb_order0.PushBack(ChebOrder);
        fourier_order0.PushBack(FourierOrder);
      }
      elem_lst0.Init(cheb_order0, fourier_order0, coord0, radius0);

      Vector<Real> coord1, radius1;
      Vector<Long> cheb_order1, fourier_order1;
      for (Long k = idx0; k < idx1; k++) { // Init elem_lst1
        const Integer ChebOrder = 10, FourierOrder = 14;
        const auto& nds = CenterlineNodes(ChebOrder);
        for (Long i = 0; i < nds.Dim(); i++) {
          Real x, y, z, r;
          loop_geom(x, y, z, r, const_pi<Real>()*(1+(k+nds[i])/Nelem));
          coord1.PushBack(x);
          coord1.PushBack(y);
          coord1.PushBack(z);
          radius1.PushBack(r);
        }
        cheb_order1.PushBack(ChebOrder);
        fourier_order1.PushBack(FourierOrder);
      }
      elem_lst1.Init(cheb_order1, fourier_order1, coord1, radius1);
    }

    KerSL kernel_sl;
    KerDL kernel_dl;
    KerGrad kernel_grad;
    BoundaryIntegralOp<Real,KerSL> BIOpSL(kernel_sl, false, comm);
    BoundaryIntegralOp<Real,KerDL> BIOpDL(kernel_dl, false, comm);
    BIOpSL.AddElemList(elem_lst0, "elem_lst0");
    BIOpSL.AddElemList(elem_lst1, "elem_lst1");
    BIOpDL.AddElemList(elem_lst0, "elem_lst0");
    BIOpDL.AddElemList(elem_lst1, "elem_lst1");
    BIOpSL.SetAccuracy(tol);
    BIOpDL.SetAccuracy(tol);

    Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
    { // Get X, Xn
      Vector<Vector<Real>> X_(2), Xn_(2);
      elem_lst0.GetNodeCoord(&X_[0], &Xn_[0], nullptr);
      elem_lst1.GetNodeCoord(&X_[1], &Xn_[1], nullptr);
      concat_vecs(X, X_);
      concat_vecs(Xn, Xn_);
    }
    { // Set Fs, Fd, Uref
      Vector<Real> X0{0.3,0.6,0.2}, Xn0{0,0,0}, F0{1}, dU;
      kernel_sl.Eval(Uref, X, X0, Xn0, F0);
      kernel_grad.Eval(dU, X, X0, Xn0, F0);

      Fd = Uref;
      { // Set Fs <-- -dot_prod(dU, Xn)
        Fs.ReInit(X.Dim()/COORD_DIM);
        for (Long i = 0; i < Fs.Dim(); i++) {
          Real dU_dot_Xn = 0;
          for (Long k = 0; k < COORD_DIM; k++) {
            dU_dot_Xn += dU[i*COORD_DIM+k] * Xn[i*COORD_DIM+k];
          }
          Fs[i] = -dU_dot_Xn;
        }
      }
    }

    // Warm-up run
    BIOpSL.ComputePotential(Us,Fs);
    BIOpDL.ComputePotential(Ud,Fd);
    BIOpSL.ClearSetup();
    BIOpDL.ClearSetup();
    Us = 0; Ud = 0;

    sctl::Profile::Enable(true);
    Profile::Tic("Setup+Eval", &comm);
    BIOpSL.ComputePotential(Us,Fs);
    BIOpDL.ComputePotential(Ud,Fd);
    Profile::Toc();

    Vector<Real> Uerr = Fd*0.5 + (Us - Ud) - Uref;
    { // Write VTK
      Vector<Vector<Real>> X_(2);
      elem_lst0.GetNodeCoord(&X_[0], nullptr, nullptr);
      elem_lst1.GetNodeCoord(&X_[1], nullptr, nullptr);
      const Long N0 = X_[0].Dim()/COORD_DIM;
      const Long N1 = X_[1].Dim()/COORD_DIM;
      elem_lst0.WriteVTK("Uerr0", Vector<Real>(N0,Uerr.begin()+ 0,false), comm);
      elem_lst1.WriteVTK("Uerr1", Vector<Real>(N1,Uerr.begin()+N0,false), comm);
    }
    { // Print error
      StaticArray<Real,2> max_err{0,0};
      StaticArray<Real,2> max_val{0,0};
      for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], fabs(x));
      for (auto x : Uref) max_val[0] = std::max<Real>(max_val[0], fabs(x));
      comm.Allreduce(max_err+0, max_err+1, 1, Comm::CommOp::MAX);
      comm.Allreduce(max_val+0, max_val+1, 1, Comm::CommOp::MAX);
      if (!pid) std::cout<<"Error = "<<max_err[1]/max_val[1]<<'\n';
    }

    sctl::Profile::print(&comm);
    sctl::Profile::Enable(false);
  }

  template <class Real> void SlenderElemList<Real>::GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_ds, Vector<Real>* dX_dt, const Vector<Real>& s_param, const Vector<Real>& sin_theta_, const Vector<Real>& cos_theta_, const Long elem_idx) const {
    SCTL_ASSERT_MSG(elem_idx < Size(), "element index is greater than number of elements in the list!");
    using Vec3 = Tensor<Real,true,COORD_DIM,1>;
    const Integer ChebOrder = cheb_order[elem_idx];
    const Long Nt = sin_theta_.Dim();
    const Long Ns = s_param.Dim();
    const Long N = Ns * Nt;

    if (X     && X    ->Dim() != N*COORD_DIM) X    ->ReInit(N*COORD_DIM);
    if (Xn    && Xn   ->Dim() != N*COORD_DIM) Xn   ->ReInit(N*COORD_DIM);
    if (Xa    && Xa   ->Dim() != N          ) Xa   ->ReInit(N);
    if (dX_ds && dX_ds->Dim() != N*COORD_DIM) dX_ds->ReInit(N*COORD_DIM);
    if (dX_dt && dX_dt->Dim() != N*COORD_DIM) dX_dt->ReInit(N*COORD_DIM);

    Matrix<Real> M_lagrange_interp;
    { // Set M_lagrange_interp
      M_lagrange_interp.ReInit(ChebOrder, Ns);
      Vector<Real> V_lagrange_interp(ChebOrder*Ns, M_lagrange_interp.begin(), false);
      LagrangeInterp<Real>::Interpolate(V_lagrange_interp, CenterlineNodes(ChebOrder), s_param);
    }

    Matrix<Real> r_, dr_, x_, dx_, d2x_, e1_;
    r_  .ReInit(        1,Ns);
    x_  .ReInit(COORD_DIM,Ns);
    dx_ .ReInit(COORD_DIM,Ns);
    e1_ .ReInit(COORD_DIM,Ns);
    Matrix<Real>::GEMM(  x_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>) coord.begin()+COORD_DIM*elem_dsp[elem_idx],false), M_lagrange_interp);
    Matrix<Real>::GEMM( dx_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    dx.begin()+COORD_DIM*elem_dsp[elem_idx],false), M_lagrange_interp);
    Matrix<Real>::GEMM(  r_, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)radius.begin()+          elem_dsp[elem_idx],false), M_lagrange_interp);
    Matrix<Real>::GEMM( e1_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    e1.begin()+COORD_DIM*elem_dsp[elem_idx],false), M_lagrange_interp);
    if (Xn || Xa) { // Set dr_, d2x_
      dr_ .ReInit(        1,Ns);
      d2x_.ReInit(COORD_DIM,Ns);
      Matrix<Real>::GEMM(d2x_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)   d2x.begin()+COORD_DIM*elem_dsp[elem_idx],false), M_lagrange_interp);
      Matrix<Real>::GEMM( dr_, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)    dr.begin()+          elem_dsp[elem_idx],false), M_lagrange_interp);
    }
    auto compute_coord = [](Vec3& y, const Vec3& x, const Vec3& e1, const Vec3& e2, const Real r, const Real sint, const Real cost) {
      y = x + e1*(r*cost) + e2*(r*sint);
    };
    auto compute_normal_area_elem_tangents = [](Vec3& n, Real& da, Vec3& dy_ds, Vec3& dy_dt, const Vec3& dx, const Vec3& e1, const Vec3& e2, const Vec3& de1, const Vec3& de2, const Real r, const Real dr, const Real sint, const Real cost) {
      dy_ds = dx + e1*(dr*cost) + e2*(dr*sint) + de1*(r*cost) + de2*(r*sint);
      dy_dt = e1*(-r*sint) + e2*(r*cost);

      n = cross_prod(dy_ds, dy_dt);
      da = sqrt<Real>(dot_prod(n,n));
      n = n * (1/da);
    };

    for (Long j = 0; j < Ns; j++) {
      Real r, inv_dx2;
      Vec3 x, dx, e1, e2;
      { // Set x, dx, e1, r, inv_dx2
        for (Integer k = 0; k < COORD_DIM; k++) {
          x(k,0)  = x_[k][j];
          dx(k,0) = dx_[k][j];
          e1(k,0) = e1_[k][j];
        }
        inv_dx2 = 1/dot_prod(dx,dx);
        r = r_[0][j];

        e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
        e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

        e2 = cross_prod(e1, dx);
        e2 = e2 * (1/sqrt<Real>(dot_prod(e2,e2)));
      }

      if (X) {
        for (Integer i = 0; i < Nt; i++) { // Set X
          Vec3 y;
          compute_coord(y, x, e1, e2, r, sin_theta_[i], cos_theta_[i]);
          for (Integer k = 0; k < COORD_DIM; k++) {
            (*X)[(j*Nt+i)*COORD_DIM+k] = y(k,0);
          }
        }
      }
      if (Xn || Xa || dX_ds || dX_dt) {
        Vec3 d2x, de1, de2;
        for (Integer k = 0; k < COORD_DIM; k++) {
          d2x(k,0) = d2x_[k][j];
        }
        de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
        de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
        Real dr = dr_[0][j];

        for (Integer i = 0; i < Nt; i++) { // Set X, Xn, Xa, dX_ds, dX_dt
          Real da;
          Vec3 n, dx_ds, dx_dt;
          compute_normal_area_elem_tangents(n, da, dx_ds, dx_dt, dx, e1, e2, de1, de2, r, dr, sin_theta_[i], cos_theta_[i]);
          if (Xn) {
            for (Integer k = 0; k < COORD_DIM; k++) {
              (*Xn)[(j*Nt+i)*COORD_DIM+k] = n(k,0);
            }
          }
          if (Xa) {
            (*Xa)[j*Nt+i] = da;
          }
          if (dX_ds) {
            for (Integer k = 0; k < COORD_DIM; k++) {
              (*dX_ds)[(j*Nt+i)*COORD_DIM+k] = dx_ds(k,0);
            }
          }
          if (dX_dt) {
            for (Integer k = 0; k < COORD_DIM; k++) {
              (*dX_dt)[(j*Nt+i)*COORD_DIM+k] = dx_dt(k,0);
            }
          }
        }
      }
    }
  }

  template <class Real> template <Integer digits, bool trg_dot_prod, class Kernel> Matrix<Real> SlenderElemList<Real>::SelfInteracHelper(const Kernel& ker, const Long elem_idx) const {
    using Vec3 = Tensor<Real,true,COORD_DIM,1>;
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1 = Kernel::TrgDim()/(trg_dot_prod?COORD_DIM:1);
    //const Integer digits = (Integer)(log(tol)/log(0.1)+0.5);

    const Integer ChebOrder = cheb_order[elem_idx];
    const Integer FourierOrder = fourier_order[elem_idx];
    const Integer FourierModes = FourierOrder/2+1;
    const Matrix<Real> M_fourier_inv = fourier_matrix_inv_transpose<Real>(FourierOrder,FourierModes);

    const Vector<Real>  coord(COORD_DIM*ChebOrder,(Iterator<Real>)this-> coord.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real>     dx(COORD_DIM*ChebOrder,(Iterator<Real>)this->    dx.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real>    d2x(COORD_DIM*ChebOrder,(Iterator<Real>)this->   d2x.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real> radius(        1*ChebOrder,(Iterator<Real>)this->radius.begin()+          elem_dsp[elem_idx],false);
    const Vector<Real>     dr(        1*ChebOrder,(Iterator<Real>)this->    dr.begin()+          elem_dsp[elem_idx],false);
    const Vector<Real>     e1(COORD_DIM*ChebOrder,(Iterator<Real>)this->    e1.begin()+COORD_DIM*elem_dsp[elem_idx],false);

    const Real dtheta = 2*const_pi<Real>()/FourierOrder;
    const Complex<Real> exp_dtheta(cos<Real>(dtheta), sin<Real>(dtheta));

    Matrix<Real> M_modal(ChebOrder*FourierOrder, ChebOrder*KDIM0*KDIM1*FourierModes*2);
    for (Long i = 0; i < ChebOrder; i++) {
      Real r_trg = radius[i];
      Real dr_trg = dr[i];
      Vec3 x_trg, dx_trg, e1_trg, e2_trg, de1_trg, de2_trg;
      { // Set x_trg, e1_trg, e2_trg
        Vec3 d2x_trg;
        for (Integer k = 0; k < COORD_DIM; k++) {
          x_trg (k,0) = coord[k*ChebOrder+i];
          e1_trg(k,0) = e1[k*ChebOrder+i];
          dx_trg(k,0) = dx[k*ChebOrder+i];
          d2x_trg(k,0) = d2x[k*ChebOrder+i];
        }
        Real inv_dx2 = 1/dot_prod(dx_trg,dx_trg);
        e1_trg = e1_trg - dx_trg * dot_prod(e1_trg, dx_trg) * inv_dx2;
        e1_trg = e1_trg * (1/sqrt<Real>(dot_prod(e1_trg,e1_trg)));

        e2_trg = cross_prod(e1_trg, dx_trg);
        e2_trg = e2_trg * (1/sqrt<Real>(dot_prod(e2_trg,e2_trg)));

        de1_trg = dx_trg*(-dot_prod(e1_trg,d2x_trg) * inv_dx2);
        de2_trg = dx_trg*(-dot_prod(e2_trg,d2x_trg) * inv_dx2);
      }

      Matrix<Real> Minterp_quad_nds;
      Vector<Real> vec_tmp, quad_wts; // Quadrature rule in s
      SpecialQuadRule<ModalUpsample,Real,Kernel,trg_dot_prod>(vec_tmp, quad_wts, Minterp_quad_nds, ChebOrder, i, r_trg, sqrt<Real>(dot_prod(dx_trg, dx_trg)), digits);
      const Long Nq = Minterp_quad_nds.Dim(1);

      Matrix<Real> r_src, dr_src, x_src, dx_src, d2x_src, e1_src, e2_src, de1_src, de2_src;
      r_src  .ReInit(        1, Nq);
      dr_src .ReInit(        1, Nq);
      x_src  .ReInit(COORD_DIM, Nq);
      dx_src .ReInit(COORD_DIM, Nq);
      d2x_src.ReInit(COORD_DIM, Nq);
      e1_src .ReInit(COORD_DIM, Nq);
      e2_src .ReInit(COORD_DIM, Nq);
      de1_src.ReInit(COORD_DIM, Nq);
      de2_src.ReInit(COORD_DIM, Nq);
      { // Set x_src, x_trg (improve numerical stability)
        Matrix<Real> x_nodes(COORD_DIM,ChebOrder, (Iterator<Real>)coord.begin(), true);
        for (Long j = 0; j < ChebOrder; j++) {
          for (Integer k = 0; k < COORD_DIM; k++) {
            x_nodes[k][j] -= x_trg(k,0);
          }
        }
        Matrix<Real>::GEMM(  x_src, x_nodes, Minterp_quad_nds);
        for (Integer k = 0; k < COORD_DIM; k++) {
          x_trg(k,0) = 0;
        }
      }
      //Matrix<Real>::GEMM(  x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>) coord.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM( dx_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    dx.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM(d2x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)   d2x.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM(  r_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)radius.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM( dr_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)    dr.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM( e1_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    e1.begin(),false), Minterp_quad_nds);
      for (Long j = 0; j < Nq; j++) { // Set e2_src
        Vec3 e1, dx, d2x;
        for (Integer k = 0; k < COORD_DIM; k++) {
          e1(k,0) = e1_src[k][j];
          dx(k,0) = dx_src[k][j];
          d2x(k,0) = d2x_src[k][j];
        }
        Real inv_dx2 = 1/dot_prod(dx,dx);
        e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
        e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

        Vec3 e2 = cross_prod(e1, dx);
        e2 = e2 * (1/sqrt<Real>(dot_prod(e2,e2)));
        Vec3 de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
        Vec3 de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
        for (Integer k = 0; k < COORD_DIM; k++) {
          e1_src[k][j] = e1(k,0);
          e2_src[k][j] = e2(k,0);
          de1_src[k][j] = de1(k,0);
          de2_src[k][j] = de2(k,0);
        }
      }

      Complex<Real> exp_theta_trg(1,0);
      for (Long j = 0; j < FourierOrder; j++) {
        auto compute_Xn_trg = [&exp_theta_trg,&dx_trg,&e1_trg,&e2_trg,&de1_trg,&de2_trg,&r_trg,&dr_trg]() { // Set n_trg
          const Vec3 dy_ds1 = e1_trg * dr_trg + de1_trg * r_trg;
          const Vec3 dy_ds2 = e2_trg * dr_trg + de2_trg * r_trg;
          const Vec3 dy_dt1 = e2_trg * r_trg;
          const Vec3 dy_dt2 = e1_trg * r_trg;

          const Vec3 dy_ds = dx_trg + dy_ds1 * exp_theta_trg.real + dy_ds2 * exp_theta_trg.imag;
          const Vec3 dy_dt =          dy_dt1 * exp_theta_trg.real - dy_dt2 * exp_theta_trg.imag;

          Vec3 n_trg;
          n_trg(0,0) = dy_ds(1,0) * dy_dt(2,0) - dy_ds(2,0) * dy_dt(1,0);
          n_trg(1,0) = dy_ds(2,0) * dy_dt(0,0) - dy_ds(0,0) * dy_dt(2,0);
          n_trg(2,0) = dy_ds(0,0) * dy_dt(1,0) - dy_ds(1,0) * dy_dt(0,0);
          Real scale = 1/sqrt<Real>(n_trg(0,0)*n_trg(0,0) + n_trg(1,0)*n_trg(1,0) + n_trg(2,0)*n_trg(2,0));
          return n_trg*scale;
        };
        const Vec3 y_trg = x_trg + e1_trg*r_trg*exp_theta_trg.real + e2_trg*r_trg*exp_theta_trg.imag;
        const Vec3 n_trg(trg_dot_prod ? compute_Xn_trg() : Vec3((Real)0));

        Matrix<Real> M_tor(Nq, KDIM0*KDIM1*FourierModes*2); // TODO: pre-allocate
        toroidal_greens_fn_batched<digits+2,ModalUpsample,trg_dot_prod>(M_tor, y_trg, n_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, e2_src, de1_src, de2_src, ker, FourierModes);

        for (Long ii = 0; ii < M_tor.Dim(0); ii++) {
          for (Long jj = 0; jj < M_tor.Dim(1); jj++) {
            M_tor[ii][jj] *= quad_wts[ii];
          }
        }
        Matrix<Real> M_modal_(ChebOrder, KDIM0*KDIM1*FourierModes*2, M_modal[i*FourierOrder+j], false);
        Matrix<Real>::GEMM(M_modal_, Minterp_quad_nds, M_tor);
        exp_theta_trg *= exp_dtheta;
      }
    }

    Matrix<Real> M_nodal(ChebOrder*FourierOrder, ChebOrder*KDIM0*KDIM1*FourierOrder);
    { // Set M_nodal
      const Matrix<Real> M_modal_(ChebOrder*FourierOrder * ChebOrder*KDIM0*KDIM1, FourierModes*2, M_modal.begin(), false);
      Matrix<Real> M_nodal_(ChebOrder*FourierOrder * ChebOrder*KDIM0*KDIM1, FourierOrder, M_nodal.begin(), false);
      Matrix<Real>::GEMM(M_nodal_, M_modal_, M_fourier_inv);
    }

    Matrix<Real> M(ChebOrder*FourierOrder*KDIM0, ChebOrder*FourierOrder*KDIM1);
    { // Set M
      const Integer Nnds = ChebOrder*FourierOrder;
      for (Integer i = 0; i < Nnds; i++) {
        for (Integer j0 = 0; j0 < ChebOrder; j0++) {
          for (Integer k0 = 0; k0 < KDIM0; k0++) {
            for (Integer k1 = 0; k1 < KDIM1; k1++) {
              for (Integer j1 = 0; j1 < FourierOrder; j1++) {
                M[(j0*FourierOrder+j1)*KDIM0+k0][i*KDIM1+k1] = M_nodal[i][((j0*KDIM0+k0)*KDIM1+k1)*FourierOrder+j1] * ker.template uKerScaleFactor<Real>();
              }
            }
          }
        }
      }
    }
    return M;
  }

  template <class Real> template <class ValueType> void SlenderElemList<Real>::Copy(SlenderElemList<ValueType>& elem_lst) const {
    const Long N = radius.Dim();
    Vector<ValueType> radius_(N), coord_(N*COORD_DIM), e1_(N*COORD_DIM);
    for (Long i = 0; i < cheb_order.Dim(); i++) {
      for (Long j = 0; j < cheb_order[i]; j++) {
        radius_[elem_dsp[i]+j] = (ValueType)radius[elem_dsp[i]+j];
        for (Integer k = 0; k < COORD_DIM; k++) {
          Long idx_ = elem_dsp[i]*COORD_DIM+j*COORD_DIM+k;
          Long idx = elem_dsp[i]*COORD_DIM+k*cheb_order[i]+j;
          coord_[idx_] = (ValueType)coord[idx];
          e1_[idx_] = (ValueType)e1[idx];
        }
      }
    }
    elem_lst.Init(cheb_order, fourier_order, coord_, radius_, e1_);
  }
}

