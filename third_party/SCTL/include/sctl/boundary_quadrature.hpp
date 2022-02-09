#ifndef _SCTL_BOUNDARY_QUADRATURE_HPP_
#define _SCTL_BOUNDARY_QUADRATURE_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(kernel_functions.hpp)
#include SCTL_INCLUDE(tree.hpp)
#include SCTL_INCLUDE(cheb_utils.hpp)
#include SCTL_INCLUDE(fmm-wrapper.hpp)
#include SCTL_INCLUDE(tensor.hpp)
#include SCTL_INCLUDE(profile.hpp)

#include <mutex>
#include <atomic>
#include <tuple>
#include <functional>

namespace SCTL_NAMESPACE {

template <class Real, Integer DIM, Integer ORDER> class Basis {
  public:
    using ValueType = Real;

    // class EvalOperator {
    //   public:
    // };
    using EvalOpType = Matrix<ValueType>;

    static constexpr Long Dim() {
      return DIM;
    }
    static constexpr Long Size() {
      return pow<DIM,Long>(ORDER);
    }
    static const Matrix<ValueType>& Nodes() {
      static Matrix<ValueType> nodes_(DIM,Size());
      auto nodes_1d = [](Integer i) {
        return 0.5 - 0.5 * sctl::cos<ValueType>((2*i+1) * const_pi<ValueType>() / (2*ORDER));
      };
      { // Set nodes_
        static std::mutex mutex;
        static std::atomic<Integer> first_time(true);
        if (first_time.load(std::memory_order_relaxed)) {
          std::lock_guard<std::mutex> guard(mutex);
          if (first_time.load(std::memory_order_relaxed)) {
            Integer N = 1;
            for (Integer d = 0; d < DIM; d++) {
              for (Integer j = 0; j < ORDER; j++) {
                for (Integer i = 0; i < N; i++) {
                  for (Integer k = 0; k < d; k++) {
                    nodes_[k][j*N+i] = nodes_[k][i];
                  }
                  nodes_[d][j*N+i] = nodes_1d(j);
                }
              }
              N *= ORDER;
            }

            std::atomic_thread_fence(std::memory_order_seq_cst);
            first_time.store(false);
          }
        }
      }
      return nodes_;
    }

    static void Grad(Vector<Basis>& dX, const Vector<Basis>& X) {
      static Matrix<ValueType> GradOp[DIM];
      static std::mutex mutex;
      static std::atomic<Integer> first_time(true);
      if (first_time.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> guard(mutex);
        if (first_time.load(std::memory_order_relaxed)) {
          { // Set GradOp
            auto nodes = Basis<ValueType,1,ORDER>::Nodes();
            SCTL_ASSERT(nodes.Dim(1) == ORDER);
            Matrix<ValueType> M(ORDER, ORDER);
            for (Integer i = 0; i < ORDER; i++) { // Set M
              Real x = nodes[0][i];
              for (Integer j = 0; j < ORDER; j++) {
                M[j][i] = 0;
                for (Integer l = 0; l < ORDER; l++) {
                  if (l != j) {
                    Real M_ = 1;
                    for (Integer k = 0; k < ORDER; k++) {
                      if (k != j && k != l) M_ *= (x - nodes[0][k]);
                      if (k != j) M_ /= (nodes[0][j] - nodes[0][k]);
                    }
                    M[j][i] += M_;
                  }
                }
              }
            }
            for (Integer d = 0; d < DIM; d++) {
              GradOp[d].ReInit(Size(), Size());
              GradOp[d] = 0;
              Integer stride0 = sctl::pow<Integer>(ORDER, d);
              Integer repeat0 = sctl::pow<Integer>(ORDER, d);
              Integer stride1 = sctl::pow<Integer>(ORDER, d+1);
              Integer repeat1 = sctl::pow<Integer>(ORDER, DIM-d-1);
              for (Integer k1 = 0; k1 < repeat1; k1++) {
                for (Integer i = 0; i < ORDER; i++) {
                  for (Integer j = 0; j < ORDER; j++) {
                    for (Integer k0 = 0; k0 < repeat0; k0++) {
                      GradOp[d][k1*stride1 + i*stride0 + k0][k1*stride1 + j*stride0 + k0] = M[i][j];
                    }
                  }
                }
              }
            }
          }
          std::atomic_thread_fence(std::memory_order_seq_cst);
          first_time.store(false);
        }
      }

      if (dX.Dim() != X.Dim()*DIM) dX.ReInit(X.Dim()*DIM);
      for (Long i = 0; i < X.Dim(); i++) {
        const Matrix<ValueType> Vi(1, Size(), (Iterator<ValueType>)(ConstIterator<ValueType>)X[i].NodeValues_, false);
        for (Integer k = 0; k < DIM; k++) {
          Matrix<ValueType> Vo(1, Size(), dX[i*DIM+k].NodeValues_, false);
          Matrix<ValueType>::GEMM(Vo, Vi, GradOp[k]);
        }
      }
    }
    static EvalOpType SetupEval(const Matrix<ValueType>& X) {
      Long N = X.Dim(1);
      SCTL_ASSERT(X.Dim(0) == DIM);
      Matrix<ValueType> M(Size(), N);
      { // Set M
        auto nodes = Basis<ValueType,1,ORDER>::Nodes();
        Integer NN = nodes.Dim(1);
        Matrix<ValueType> M_(NN, DIM*N);
        for (Long i = 0; i < DIM*N; i++) {
          ValueType x = X[0][i];
          for (Integer j = 0; j < NN; j++) {
            ValueType y = 1;
            for (Integer k = 0; k < NN; k++) {
              y *= (j==k ? 1 : (nodes[0][k] - x) / (nodes[0][k] - nodes[0][j]));
            }
            M_[j][i] = y;
          }
        }
        if (DIM == 1) {
          SCTL_ASSERT(M.Dim(0) == M_.Dim(0));
          SCTL_ASSERT(M.Dim(1) == M_.Dim(1));
          M = M_;
        } else {
          Integer NNN = 1;
          M = 1;
          for (Integer d = 0; d < DIM; d++) {
            for (Integer k = 1; k < NN; k++) {
              for (Integer j = 0; j < NNN; j++) {
                for (Long i = 0; i < N; i++) {
                  M[k*NNN+j][i] = M[j][i] * M_[k][d*N+i];
                }
              }
            }
            { // k = 0
              for (Integer j = 0; j < NNN; j++) {
                for (Long i = 0; i < N; i++) {
                  M[j][i] *= M_[0][d*N+i];
                }
              }
            }
            NNN *= NN;
          }
        }
      }
      return M;
    }
    static void Eval(Matrix<ValueType>& Y, const Vector<Basis>& X, const EvalOpType& M) {
      Long N0 = X.Dim();
      Long N1 = M.Dim(1);
      SCTL_ASSERT(M.Dim(0) == Size());
      if (Y.Dim(0) != N0 || Y.Dim(1) != N1) Y.ReInit(N0, N1);
      for (Long i = 0; i < N0; i++) {
        const Matrix<ValueType> X_(1,Size(),(Iterator<ValueType>)(ConstIterator<ValueType>)X[i].NodeValues_,false);
        Matrix<ValueType> Y_(1,N1,Y[i],false);
        Matrix<ValueType>::GEMM(Y_,X_,M);
      }
    }

    const ValueType& operator[](Long i) const {
      SCTL_ASSERT(i < Size());
      return NodeValues_[i];
    }
    ValueType& operator[](Long i) {
      SCTL_ASSERT(i < Size());
      return NodeValues_[i];
    }

  private:
    StaticArray<ValueType,Size()> NodeValues_;
};

template <Integer COORD_DIM, class Basis> class ElemList {
  public:

    using CoordBasis = Basis;
    using CoordType = typename CoordBasis::ValueType;

    static constexpr Integer CoordDim() {
      return COORD_DIM;
    }
    static constexpr Integer ElemDim() {
      return CoordBasis::Dim();
    }

    ElemList(Long Nelem = 0) {
      ReInit(Nelem);
    }
    void ReInit(Long Nelem = 0) {
      Nelem_ = Nelem;
      X_.ReInit(Nelem_ * COORD_DIM);
    }
    void ReInit(const Vector<CoordBasis>& X) {
      Nelem_ = X.Dim() / COORD_DIM;
      SCTL_ASSERT(X.Dim() == Nelem_ * COORD_DIM);
      X_ = X;
    }
    Long NElem() const {
      return Nelem_;
    }

    CoordBasis& operator()(Long elem, Integer dim) {
      SCTL_ASSERT(elem >= 0 && elem < Nelem_);
      SCTL_ASSERT(dim >= 0 && dim < COORD_DIM);
      return X_[elem*COORD_DIM+dim];
    }
    const CoordBasis& operator()(Long elem, Integer dim) const {
      SCTL_ASSERT(elem >= 0 && elem < Nelem_);
      SCTL_ASSERT(dim >= 0 && dim < COORD_DIM);
      return X_[elem*COORD_DIM+dim];
    }
    const Vector<CoordBasis>& ElemVector() const {
      return X_;
    }

  private:
    static_assert(CoordBasis::Dim() <= CoordDim(), "Basis dimension can not be greater than COORD_DIM.");
    Vector<CoordBasis> X_;
    Long Nelem_;

    mutable Vector<CoordBasis> dX_;
};

template <class Real> class Quadrature {

    static Real machine_epsilon() {
      Real eps=1;
      while(eps*(Real)0.5+(Real)1.0>1.0) eps*=0.5;
      return eps;
    }

    template <Integer DIM> static void DuffyQuad(Matrix<Real>& nodes, Vector<Real>& weights, const Vector<Real>& coord, Integer order, Real adapt = -1.0) {
      SCTL_ASSERT(coord.Dim() == DIM);
      static Real eps = machine_epsilon()*16;

      Matrix<Real> qx;
      Vector<Real> qw;
      { // Set qx, qw
        Vector<Real> qx0, qw0;
        ChebBasis<Real>::quad_rule(order, qx0, qw0);

        Integer N = sctl::pow<DIM,Integer>(order);
        qx.ReInit(DIM,N);
        qw.ReInit(N);
        qw[0] = 1;

        Integer N_ = 1;
        for (Integer d = 0; d < DIM; d++) {
          for (Integer j = 0; j < order; j++) {
            for (Integer i = 0; i < N_; i++) {
              for (Integer k = 0; k < d; k++) {
                qx[k][j*N_+i] = qx[k][i];
              }
              qx[d][j*N_+i] = qx0[j];
              qw[j*N_+i] = qw[i];
            }
          }
          for (Integer j = 0; j < order; j++) {
            for (Integer i = 0; i < N_; i++) {
              qw[j*N_+i] *= qw0[j];
            }
          }
          N_ *= order;
        }
      }

      Vector<Real> X;
      { // Set X
        StaticArray<Real,2*DIM+2> X_;
        X_[0] = 0;
        X_[1] = adapt;
        for (Integer i = 0; i < DIM; i++) {
          X_[2*i+2] = sctl::fabs<Real>(coord[i]);
          X_[2*i+3] = sctl::fabs<Real>(coord[i]-1);
        }
        std::sort((Iterator<Real>)X_, (Iterator<Real>)X_+2*DIM+2);

        X.PushBack(std::max<Real>(0, X_[2*DIM]-1));
        for (Integer i = 0; i < 2*DIM+2; i++) {
          if (X[X.Dim()-1] < X_[i]) {
            if (X.Dim())
            X.PushBack(X_[i]);
          }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////
        Vector<Real> r(1);
        r[0] = X[0];
        for (Integer i = 1; i < X.Dim(); i++) {
          while (r[r.Dim() - 1] > 0.0 && (order*0.5) * r[r.Dim() - 1] < X[i]) r.PushBack((order*0.5) * r[r.Dim() - 1]); // TODO
          r.PushBack(X[i]);
        }
        X = r;
        /////////////////////////////////////////////////////////////////////////////////////////////////
      }

      Vector<Real> nds, wts;
      for (Integer k = 0; k < X.Dim()-1; k++) {
        for (Integer dd = 0; dd < 2*DIM; dd++) {
          Integer d0 = (dd>>1);
          StaticArray<Real,2*DIM> range0, range1;
          { // Set range0, range1
            Integer d1 = (dd%2?1:-1);
            for (Integer d = 0; d < DIM; d++) {
              range0[d*2+0] = std::max<Real>(0,std::min<Real>(1,coord[d] - X[k]  ));
              range0[d*2+1] = std::max<Real>(0,std::min<Real>(1,coord[d] + X[k]  ));
              range1[d*2+0] = std::max<Real>(0,std::min<Real>(1,coord[d] - X[k+1]));
              range1[d*2+1] = std::max<Real>(0,std::min<Real>(1,coord[d] + X[k+1]));
            }
            range0[d0*2+0] = std::max<Real>(0,std::min<Real>(1,coord[d0] + d1*X[k+0]));
            range0[d0*2+1] = std::max<Real>(0,std::min<Real>(1,coord[d0] + d1*X[k+0]));
            range1[d0*2+0] = std::max<Real>(0,std::min<Real>(1,coord[d0] + d1*X[k+1]));
            range1[d0*2+1] = std::max<Real>(0,std::min<Real>(1,coord[d0] + d1*X[k+1]));
          }
          { // if volume(range0, range1) == 0 then continue
            Real v0 = 1, v1 = 1;
            for (Integer d = 0; d < DIM; d++) {
              if (d == d0) {
                v0 *= sctl::fabs<Real>(range0[d*2+0]-range1[d*2+0]);
                v1 *= sctl::fabs<Real>(range0[d*2+0]-range1[d*2+0]);
              } else {
                v0 *= range0[d*2+1]-range0[d*2+0];
                v1 *= range1[d*2+1]-range1[d*2+0];
              }
            }
            if (v0 < eps && v1 < eps) continue;
          }
          for (Integer i = 0; i < qx.Dim(1); i++) { // Set nds, wts
            Real w = qw[i];
            Real z = qx[d0][i];
            for (Integer d = 0; d < DIM; d++) {
              Real y = qx[d][i];
              nds.PushBack((range0[d*2+0]*(1-y) + range0[d*2+1]*y)*(1-z) + (range1[d*2+0]*(1-y) + range1[d*2+1]*y)*z);
              if (d == d0) {
                w *= abs(range1[d*2+0] - range0[d*2+0]);
              } else {
                w *= (range0[d*2+1] - range0[d*2+0])*(1-z) + (range1[d*2+1] - range1[d*2+0])*z;
              }
            }
            wts.PushBack(w);
          }
        }
      }
      nodes = Matrix<Real>(nds.Dim()/DIM,DIM,nds.begin()).Transpose();
      weights = wts;
    }

    template <Integer DIM> static void TensorProductGaussQuad(Matrix<Real>& nodes, Vector<Real>& weights, Integer order) {
      Vector<Real> coord(DIM);
      coord = 0;
      coord[0] = -10;
      DuffyQuad<DIM>(nodes, weights, coord, order);
    }



    template <class DensityBasis, class ElemList, class Kernel> static void SetupSingular(Matrix<Real>& M_singular, const Matrix<Real>& trg_nds, const ElemList& elem_lst, const Kernel& kernel, Integer order_singular = 10, Integer order_direct = 10) {
      using CoordBasis = typename ElemList::CoordBasis;
      using CoordEvalOpType = typename CoordBasis::EvalOpType;
      using DensityEvalOpType = typename DensityBasis::EvalOpType;

      constexpr Integer CoordDim = ElemList::CoordDim();
      constexpr Integer ElemDim = ElemList::ElemDim();
      constexpr Integer KDIM0 = Kernel::SrcDim();
      constexpr Integer KDIM1 = Kernel::TrgDim();

      const Long Nelem = elem_lst.NElem();
      const Integer Ntrg = trg_nds.Dim(1);
      SCTL_ASSERT(trg_nds.Dim(0) == ElemDim);

      Vector<Real> Xt;
      { // Set Xt
        auto Meval = CoordBasis::SetupEval(trg_nds);
        eval_basis(Xt, elem_lst.ElemVector(), ElemList::CoordDim(), trg_nds.Dim(1), Meval);
      }
      SCTL_ASSERT(Xt.Dim() == Nelem * Ntrg * CoordDim);

      const Vector<CoordBasis>& X = elem_lst.ElemVector();
      Vector<CoordBasis> dX;
      CoordBasis::Grad(dX, X);

      auto& M = M_singular;
      M.ReInit(Nelem * KDIM0 * DensityBasis::Size(), KDIM1 * Ntrg);
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < Ntrg; i++) { // Set M (singular)
        Matrix<Real> quad_nds;
        Vector<Real> quad_wts;
        { // Set quad_nds, quad_wts
          StaticArray<Real,ElemDim> trg_node_;
          for (Integer k = 0; k < ElemDim; k++) {
            trg_node_[k] = trg_nds[k][i];
          }
          Vector<Real> trg_node(ElemDim, trg_node_, false);
          DuffyQuad<ElemDim>(quad_nds, quad_wts, trg_node, order_singular);
        }
        const CoordEvalOpType CoordEvalOp = CoordBasis::SetupEval(quad_nds);
        Integer Nnds = quad_wts.Dim();

        Vector<Real> X_, dX_, Xa_, Xn_;
        { // Set X_, dX_
          eval_basis(X_, X, CoordDim, Nnds, CoordEvalOp);
          eval_basis(dX_, dX, CoordDim * ElemDim, Nnds, CoordEvalOp);
        }
        if (CoordDim == 3 && ElemDim == 2) { // Compute Xa_, Xn_
          Long N = Nelem*Nnds;
          Xa_.ReInit(N);
          Xn_.ReInit(N*CoordDim);
          for (Long j = 0; j < N; j++) {
            StaticArray<Real,CoordDim> normal;
            normal[0] = dX_[j*6+2]*dX_[j*6+5] - dX_[j*6+4]*dX_[j*6+3];
            normal[1] = dX_[j*6+4]*dX_[j*6+1] - dX_[j*6+0]*dX_[j*6+5];
            normal[2] = dX_[j*6+0]*dX_[j*6+3] - dX_[j*6+2]*dX_[j*6+1];
            Xa_[j] = sctl::sqrt<Real>(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
            Real invXa = 1/Xa_[j];
            Xn_[j*3+0] = normal[0] * invXa;
            Xn_[j*3+1] = normal[1] * invXa;
            Xn_[j*3+2] = normal[2] * invXa;
          }
        }

        DensityEvalOpType DensityEvalOp;
        if (std::is_same<CoordBasis,DensityBasis>::value) {
          DensityEvalOp = CoordEvalOp;
        } else {
          DensityEvalOp = DensityBasis::SetupEval(quad_nds);
        }

        for (Long j = 0; j < Nelem; j++) {
          Matrix<Real> M__(Nnds * KDIM0, KDIM1);
          { // Set kernel matrix M__
            const Vector<Real> X0_(CoordDim, (Iterator<Real>)Xt.begin() + (j * Ntrg + i) * CoordDim, false);
            const Vector<Real> X__(Nnds * CoordDim, X_.begin() + j * Nnds * CoordDim, false);
            const Vector<Real> Xn__(Nnds * CoordDim, Xn_.begin() + j * Nnds * CoordDim, false);
            kernel.template KernelMatrix<Real>(M__, X0_, X__, Xn__);
          }
          for (Long k0 = 0; k0 < KDIM0; k0++) {
            for (Long k1 = 0; k1 < KDIM1; k1++) {
              for (Long l = 0; l < DensityBasis::Size(); l++) {
                Real M_lk = 0;
                for (Long n = 0; n < Nnds; n++) {
                  Real quad_wt = Xa_[j * Nnds + n] * quad_wts[n];
                  M_lk += DensityEvalOp[l][n] * quad_wt * M__[n*KDIM0+k0][k1];
                }
                M[(j * KDIM0 + k0) * DensityBasis::Size() + l][k1 * Ntrg + i] = M_lk;
              }
            }
          }
        }
      }
      { // Set M (subtract direct)
        Matrix<Real> quad_nds;
        Vector<Real> quad_wts;
        TensorProductGaussQuad<ElemDim>(quad_nds, quad_wts, order_direct);
        const CoordEvalOpType CoordEvalOp = CoordBasis::SetupEval(quad_nds);
        Integer Nnds = quad_wts.Dim();

        Vector<Real> X_, dX_, Xa_, Xn_;
        { // Set X_, dX_
          eval_basis(X_, X, CoordDim, Nnds, CoordEvalOp);
          eval_basis(dX_, dX, CoordDim * ElemDim, Nnds, CoordEvalOp);
        }
        if (CoordDim == 3 && ElemDim == 2) { // Compute Xa_, Xn_
          Long N = Nelem*Nnds;
          Xa_.ReInit(N);
          Xn_.ReInit(N*CoordDim);
          for (Long j = 0; j < N; j++) {
            StaticArray<Real,CoordDim> normal;
            normal[0] = dX_[j*6+2]*dX_[j*6+5] - dX_[j*6+4]*dX_[j*6+3];
            normal[1] = dX_[j*6+4]*dX_[j*6+1] - dX_[j*6+0]*dX_[j*6+5];
            normal[2] = dX_[j*6+0]*dX_[j*6+3] - dX_[j*6+2]*dX_[j*6+1];
            Xa_[j] = sctl::sqrt<Real>(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
            Real invXa = 1/Xa_[j];
            Xn_[j*3+0] = normal[0] * invXa;
            Xn_[j*3+1] = normal[1] * invXa;
            Xn_[j*3+2] = normal[2] * invXa;
          }
        }

        DensityEvalOpType DensityEvalOp;
        if (std::is_same<CoordBasis,DensityBasis>::value) {
          DensityEvalOp = CoordEvalOp;
        } else {
          DensityEvalOp = DensityBasis::SetupEval(quad_nds);
        }

        #pragma omp parallel for schedule(static)
        for (Long i = 0; i < Ntrg; i++) { // Subtract direct contribution
          for (Long j = 0; j < Nelem; j++) {
            Matrix<Real> M__(Nnds * KDIM0, KDIM1);
            { // Set kernel matrix M__
              const Vector<Real> X0_(CoordDim, (Iterator<Real>)Xt.begin() + (j * Ntrg + i) * CoordDim, false);
              const Vector<Real> X__(Nnds * CoordDim, X_.begin() + j * Nnds * CoordDim, false);
              const Vector<Real> Xn__(Nnds * CoordDim, Xn_.begin() + j * Nnds * CoordDim, false);
              kernel.template KernelMatrix<Real>(M__, X0_, X__, Xn__);
            }
            for (Long k0 = 0; k0 < KDIM0; k0++) {
              for (Long k1 = 0; k1 < KDIM1; k1++) {
                for (Long l = 0; l < DensityBasis::Size(); l++) {
                  Real M_lk = 0;
                  for (Long n = 0; n < Nnds; n++) {
                    Real quad_wt = Xa_[j * Nnds + n] * quad_wts[n];
                    M_lk += DensityEvalOp[l][n] * quad_wt * M__[n*KDIM0+k0][k1];
                  }
                  M[(j * KDIM0 + k0) * DensityBasis::Size() + l][k1 * Ntrg + i] -= M_lk;
                }
              }
            }
          }
        }
      }
    }

    template <class DensityBasis> static void EvalSingular(Matrix<Real>& U, const Vector<DensityBasis>& density, const Matrix<Real>& M, Integer KDIM0_, Integer KDIM1_) {
      if (M.Dim(0) == 0 || M.Dim(1) == 0) {
        U.ReInit(0,0);
        return;
      }

      const Long Ntrg = M.Dim(1) / KDIM1_;
      SCTL_ASSERT(M.Dim(1) == KDIM1_ * Ntrg);

      const Long Nelem = M.Dim(0) / (KDIM0_ * DensityBasis::Size());
      SCTL_ASSERT(M.Dim(0) == Nelem * KDIM0_ * DensityBasis::Size());

      const Integer dof = density.Dim() / (Nelem * KDIM0_);
      SCTL_ASSERT(density.Dim() == Nelem * dof * KDIM0_);

      if (U.Dim(0) != Nelem * dof * KDIM1_ || U.Dim(1) != Ntrg) {
        U.ReInit(Nelem * dof * KDIM1_, Ntrg);
        U = 0;
      }
      for (Long j = 0; j < Nelem; j++) {
        const Matrix<Real> M_(KDIM0_ * DensityBasis::Size(), KDIM1_ * Ntrg, (Iterator<Real>)M[j * KDIM0_ * DensityBasis::Size()], false);
        Matrix<Real> U_(dof, KDIM1_ * Ntrg, U[j*dof*KDIM1_], false);
        Matrix<Real> F_(dof, KDIM0_ * DensityBasis::Size());
        for (Long i = 0; i < dof; i++) {
          for (Long k = 0; k < KDIM0_; k++) {
            for (Long l = 0; l < DensityBasis::Size(); l++) {
              F_[i][k * DensityBasis::Size() + l] = density[(j * dof + i) * KDIM0_ + k][l];
            }
          }
        }
        Matrix<Real>::GEMM(U_, F_, M_);
      }
    }



    template <Integer DIM> struct PointData {
      bool operator<(const PointData& p) const {
        return mid < p.mid;
      }

      Long rank;
      Long surf_rank;
      Morton<DIM> mid;
      StaticArray<Real,DIM> coord;
      Real radius2;
    };

    template <class T1, class T2> struct Pair {
      Pair() {}

      Pair(T1 x, T2 y) : first(x), second(y) {}

      bool operator<(const Pair& p) const {
        return (first < p.first) || (((first == p.first) && (second < p.second)));
      }

      T1 first;
      T2 second;
    };

    template <class ElemList> static void BuildNbrList(Vector<Pair<Long,Long>>& pair_lst, const Vector<Real>& Xt, const Vector<Long>& trg_surf, const ElemList& elem_lst, Real distance_factor, Real period_length, const Comm& comm) {
      using CoordBasis = typename ElemList::CoordBasis;
      constexpr Integer CoordDim = ElemList::CoordDim();
      constexpr Integer ElemDim = ElemList::ElemDim();
      using PtData = PointData<CoordDim>;
      const Integer rank = comm.Rank();

      Real R0 = 0;
      StaticArray<Real,CoordDim> X0;
      { // Find bounding box
        Long N = Xt.Dim() / CoordDim;
        SCTL_ASSERT(Xt.Dim() == N * CoordDim);
        SCTL_ASSERT(N);

        StaticArray<Real,CoordDim*2> Xloc;
        StaticArray<Real,CoordDim*2> Xglb;
        for (Integer k = 0; k < CoordDim; k++) {
          Xloc[0*CoordDim+k] = Xt[k];
          Xloc[1*CoordDim+k] = Xt[k];
        }
        for (Long i = 0; i < N; i++) {
          for (Integer k = 0; k < CoordDim; k++) {
            Xloc[0*CoordDim+k] = std::min<Real>(Xloc[0*CoordDim+k], Xt[i*CoordDim+k]);
            Xloc[1*CoordDim+k] = std::max<Real>(Xloc[1*CoordDim+k], Xt[i*CoordDim+k]);
          }
        }
        comm.Allreduce((ConstIterator<Real>)Xloc+0*CoordDim, (Iterator<Real>)Xglb+0*CoordDim, CoordDim, Comm::CommOp::MIN);
        comm.Allreduce((ConstIterator<Real>)Xloc+1*CoordDim, (Iterator<Real>)Xglb+1*CoordDim, CoordDim, Comm::CommOp::MAX);
        for (Integer k = 0; k < CoordDim; k++) {
          R0 = std::max(R0, Xglb[1*CoordDim+k]-Xglb[0*CoordDim+k]);
        }

        R0 = R0 * 2.0;
        for (Integer k = 0; k < CoordDim; k++) {
          X0[k] = Xglb[k] - R0*0.25;
        }
      }
      if (period_length > 0) {
        R0 = period_length;
      }

      Vector<PtData> PtSrc, PtTrg;
      Integer order_upsample = (Integer)(const_pi<Real>() / distance_factor + 0.5);
      { // Set PtSrc
        const Vector<CoordBasis>& X_elem_lst = elem_lst.ElemVector();
        Vector<CoordBasis> dX_elem_lst;
        CoordBasis::Grad(dX_elem_lst, X_elem_lst);

        Matrix<Real> nds;
        Vector<Real> wts;
        TensorProductGaussQuad<ElemDim>(nds, wts, order_upsample);
        const Long Nnds = nds.Dim(1);

        Vector<Real> X, dX;
        const auto CoordEvalOp = CoordBasis::SetupEval(nds);
        eval_basis(X, X_elem_lst, CoordDim, Nnds, CoordEvalOp);
        eval_basis(dX, dX_elem_lst, CoordDim * ElemDim, Nnds, CoordEvalOp);

        const Long N = X.Dim() / CoordDim;
        const Long Nelem = elem_lst.NElem();
        SCTL_ASSERT(X.Dim() == N * CoordDim);
        SCTL_ASSERT(N == Nelem * Nnds);

        Long rank_offset, surf_rank_offset;
        { // Set rank_offset, surf_rank_offset
          comm.Scan(Ptr2ConstItr<Long>(&N,1), Ptr2Itr<Long>(&rank_offset,1), 1, Comm::CommOp::SUM);
          comm.Scan(Ptr2ConstItr<Long>(&Nelem,1), Ptr2Itr<Long>(&surf_rank_offset,1), 1, Comm::CommOp::SUM);
          surf_rank_offset -= Nelem;
          rank_offset -= N;
        }

        PtSrc.ReInit(N);
        const Real R0inv = 1.0 / R0;
        for (Long i = 0; i < N; i++) { // Set coord
          for (Integer k = 0; k < CoordDim; k++) {
            PtSrc[i].coord[k] = (X[i*CoordDim+k] - X0[k]) * R0inv;
          }
        }
        if (period_length > 0) { // Wrap-around coord
          for (Long i = 0; i < N; i++) {
            auto& x = PtSrc[i].coord;
            for (Integer k = 0; k < CoordDim; k++) {
              x[k] -= (Long)(x[k]);
            }
          }
        }
        for (Long i = 0; i < N; i++) { // Set radius2, mid, rank
          Integer depth = 0;
          { // Set radius2, depth
            Real radius2 = 0;
            for (Integer k0 = 0; k0 < ElemDim; k0++) {
              Real R2 = 0;
              for (Integer k1 = 0; k1 < CoordDim; k1++) {
                Real dX_ = dX[(i*CoordDim+k1)*ElemDim+k0];
                R2 += dX_*dX_;
              }
              radius2 = std::max(radius2, R2);
            }
            radius2 *= R0inv*R0inv * distance_factor*distance_factor;
            PtSrc[i].radius2 = radius2;

            Long Rinv = (Long)(1.0/radius2);
            while (Rinv > 0) {
              Rinv = (Rinv>>2);
              depth++;
            }
          }
          PtSrc[i].mid = Morton<CoordDim>((Iterator<Real>)PtSrc[i].coord, std::min(Morton<CoordDim>::MaxDepth(),depth));
          PtSrc[i].rank = rank_offset + i;
        }
        for (Long i = 0 ; i < Nelem; i++) { // Set surf_rank
          for (Long j = 0; j < Nnds; j++) {
            PtSrc[i*Nnds+j].surf_rank = surf_rank_offset + i;
          }
        }

        Vector<PtData> PtSrcSorted;
        comm.HyperQuickSort(PtSrc, PtSrcSorted);
        PtSrc.Swap(PtSrcSorted);
      }
      { // Set PtTrg
        const Long N = Xt.Dim() / CoordDim;
        SCTL_ASSERT(Xt.Dim() == N * CoordDim);

        Long rank_offset;
        { // Set rank_offset
          comm.Scan(Ptr2ConstItr<Long>(&N,1), Ptr2Itr<Long>(&rank_offset,1), 1, Comm::CommOp::SUM);
          rank_offset -= N;
        }

        PtTrg.ReInit(N);
        const Real R0inv = 1.0 / R0;
        for (Long i = 0; i < N; i++) { // Set coord
          for (Integer k = 0; k < CoordDim; k++) {
            PtTrg[i].coord[k] = (Xt[i*CoordDim+k] - X0[k]) * R0inv;
          }
        }
        if (period_length > 0) { // Wrap-around coord
          for (Long i = 0; i < N; i++) {
            auto& x = PtTrg[i].coord;
            for (Integer k = 0; k < CoordDim; k++) {
              x[k] -= (Long)(x[k]);
            }
          }
        }
        for (Long i = 0; i < N; i++) { // Set radius2, mid, rank
          PtTrg[i].radius2 = 0;
          PtTrg[i].mid = Morton<CoordDim>((Iterator<Real>)PtTrg[i].coord);
          PtTrg[i].rank = rank_offset + i;
        }
        if (trg_surf.Dim()) { // Set surf_rank
          SCTL_ASSERT(trg_surf.Dim() == N);
          for (Long i = 0; i < N; i++) {
            PtTrg[i].surf_rank = trg_surf[i];
          }
        } else {
          for (Long i = 0; i < N; i++) {
            PtTrg[i].surf_rank = -1;
          }
        }

        Vector<PtData> PtTrgSorted;
        comm.HyperQuickSort(PtTrg, PtTrgSorted);
        PtTrg.Swap(PtTrgSorted);
      }

      Tree<CoordDim> tree(comm);
      { // Init tree
        Vector<Real> Xall(PtSrc.Dim()+PtTrg.Dim());
        { // Set Xall
          Xall.ReInit((PtSrc.Dim()+PtTrg.Dim())*CoordDim);
          Long Nsrc = PtSrc.Dim();
          Long Ntrg = PtTrg.Dim();
          for (Long i = 0; i < Nsrc; i++) {
            for (Integer k = 0; k < CoordDim; k++) {
              Xall[i*CoordDim+k] = PtSrc[i].coord[k];
            }
          }
          for (Long i = 0; i < Ntrg; i++) {
            for (Integer k = 0; k < CoordDim; k++) {
              Xall[(Nsrc+i)*CoordDim+k] = PtTrg[i].coord[k];
            }
          }
        }
        tree.UpdateRefinement(Xall, 1000, true, period_length>0);
      }
      { // Repartition PtSrc, PtTrg
        PtData splitter;
        splitter.mid = tree.GetPartitionMID()[rank];
        comm.PartitionS(PtSrc, splitter);
        comm.PartitionS(PtTrg, splitter);
      }
      { // Add tree data PtSrc
        const auto& node_mid = tree.GetNodeMID();
        const Long N = node_mid.Dim();
        SCTL_ASSERT(N);

        Vector<Long> dsp(N), cnt(N);
        for (Long i = 0; i < N; i++) {
          PtData m0;
          m0.mid = node_mid[i];
          dsp[i] = std::lower_bound(PtSrc.begin(), PtSrc.end(), m0) - PtSrc.begin();
        }
        for (Long i = 0; i < N-1; i++) {
          cnt[i] = dsp[i+1] - dsp[i];
        }
        cnt[N-1] = PtSrc.Dim() - dsp[N-1];
        tree.AddData("PtSrc", PtSrc, cnt);
      }
      tree.template Broadcast<PtData>("PtSrc");

      { // Build pair_lst
        Vector<Long> cnt;
        Vector<PtData> PtSrc;
        tree.GetData(PtSrc, cnt, "PtSrc");
        const auto& node_mid = tree.GetNodeMID();
        const auto& node_attr = tree.GetNodeAttr();

        Vector<Morton<CoordDim>> nbr_mid_tmp;
        for (Long i = 0; i < node_mid.Dim(); i++) {
          if (node_attr[i].Leaf && !node_attr[i].Ghost) {
            Vector<Morton<CoordDim>> child_mid;
            node_mid[i].Children(child_mid);
            for (const auto& trg_mid : child_mid) {
              Integer d0 = trg_mid.Depth();
              Vector<PtData> Src, Trg;
              { // Set Trg
                PtData m0, m1;
                m0.mid = trg_mid;
                m1.mid = trg_mid.Next();
                Long a = std::lower_bound(PtTrg.begin(), PtTrg.end(), m0) - PtTrg.begin();
                Long b = std::lower_bound(PtTrg.begin(), PtTrg.end(), m1) - PtTrg.begin();
                Trg.ReInit(b-a, PtTrg.begin()+a, false);
                if (!Trg.Dim()) continue;
              }

              Vector<std::set<Long>> near_elem(Trg.Dim());
              for (Integer d = 0; d <= d0; d++) {
                trg_mid.NbrList(nbr_mid_tmp, d, period_length>0);
                for (const auto& src_mid : nbr_mid_tmp) { // Set Src
                  PtData m0, m1;
                  m0.mid = src_mid;
                  m1.mid = (d==d0 ? src_mid.Next() : src_mid.Ancestor(d+1));
                  Long a = std::lower_bound(PtSrc.begin(), PtSrc.end(), m0) - PtSrc.begin();
                  Long b = std::lower_bound(PtSrc.begin(), PtSrc.end(), m1) - PtSrc.begin();
                  Src.ReInit(b-a, PtSrc.begin()+a, false);
                  if (!Src.Dim()) continue;

                  for (Long t = 0; t < Trg.Dim(); t++) { // set near_elem[t] <-- {s : dist(s,t) < radius(s)}
                    for (Long s = 0; s < Src.Dim(); s++) {
                      if (Trg[t].surf_rank != Src[s].surf_rank) {
                        Real R2 = 0;
                        for (Integer k = 0; k < CoordDim; k++) {
                          Real dx = (Src[s].coord[k] - Trg[t].coord[k]);
                          R2 += dx * dx;
                        }
                        if (R2 < Src[s].radius2) {
                          near_elem[t].insert(Src[s].surf_rank);
                        }
                      }
                    }
                  }
                }
              }

              for (Long t = 0; t < Trg.Dim(); t++) { // Set pair_lst
                for (Long elem_idx : near_elem[t]) {
                  pair_lst.PushBack(Pair<Long,Long>(elem_idx,Trg[t].rank));
                }
              }
            }
          }
        }
      }
      { // Sort and repartition pair_lst
        Vector<Pair<Long,Long>> pair_lst_sorted;
        comm.HyperQuickSort(pair_lst, pair_lst_sorted);

        Long surf_rank_offset;
        const Long Nelem = elem_lst.NElem();
        comm.Scan(Ptr2ConstItr<Long>(&Nelem,1), Ptr2Itr<Long>(&surf_rank_offset,1), 1, Comm::CommOp::SUM);
        surf_rank_offset -= Nelem;

        comm.PartitionS(pair_lst_sorted, Pair<Long,Long>(surf_rank_offset,0));
        pair_lst.Swap(pair_lst_sorted);
      }
    }

    template <class ElemList> static void BuildNbrListDeprecated(Vector<Pair<Long,Long>>& pair_lst, const Vector<Real>& Xt, const ElemList& elem_lst, const Matrix<Real>& surf_nds, Real distance_factor) {
      using CoordBasis = typename ElemList::CoordBasis;
      constexpr Integer CoordDim = ElemList::CoordDim();
      constexpr Integer ElemDim = ElemList::ElemDim();
      const Long Nelem = elem_lst.NElem();

      const Long Ntrg = Xt.Dim() / CoordDim;
      SCTL_ASSERT(Xt.Dim() == Ntrg * CoordDim);

      Long Nnds, Nsurf_nds;
      Vector<Real> X_surf, X, dX;
      Integer order_upsample = (Integer)(const_pi<Real>() / distance_factor + 0.5);
      { // Set X, dX
        const Vector<CoordBasis>& X_elem_lst = elem_lst.ElemVector();
        Vector<CoordBasis> dX_elem_lst;
        CoordBasis::Grad(dX_elem_lst, X_elem_lst);

        Matrix<Real> nds_upsample;
        Vector<Real> wts_upsample;
        TensorProductGaussQuad<ElemDim>(nds_upsample, wts_upsample, order_upsample);

        Nnds = nds_upsample.Dim(1);
        const auto CoordEvalOp = CoordBasis::SetupEval(nds_upsample);
        eval_basis(X, X_elem_lst, CoordDim, nds_upsample.Dim(1), CoordEvalOp);
        eval_basis(dX, dX_elem_lst, CoordDim * ElemDim, nds_upsample.Dim(1), CoordEvalOp);

        Nsurf_nds = surf_nds.Dim(1);
        const auto CoordEvalOp_surf = CoordBasis::SetupEval(surf_nds);
        eval_basis(X_surf, X_elem_lst, CoordDim, Nsurf_nds, CoordEvalOp_surf);
      }

      Real d2 = distance_factor * distance_factor;
      for (Long i = 0; i < Nelem; i++) {
        std::set<Long> near_pts;
        std::set<Long> self_pts;
        for (Long j = 0; j < Nnds; j++) {
          Real R2_max = 0;
          StaticArray<Real, CoordDim> X0;
          for (Integer k = 0; k < CoordDim; k++) {
            X0[k] = X[(i*Nnds+j)*CoordDim+k];
          }
          for (Integer k0 = 0; k0 < ElemDim; k0++) {
            Real R2 = 0;
            for (Integer k1 = 0; k1 < CoordDim; k1++) {
              Real dX_ = dX[((i*Nnds+j)*CoordDim+k1)*ElemDim+k0];
              R2 += dX_*dX_;
            }
            R2_max = std::max(R2_max, R2*d2);
          }

          for (Long k = 0; k < Ntrg; k++) {
            Real R2 = 0;
            for (Integer l = 0; l < CoordDim; l++) {
              Real dX = Xt[k*CoordDim+l]- X0[l];
              R2 += dX * dX;
            }
            if (R2 < R2_max) near_pts.insert(k);
          }
        }
        for (Long j = 0; j < Nsurf_nds; j++) {
          StaticArray<Real, CoordDim> X0;
          for (Integer k = 0; k < CoordDim; k++) {
            X0[k] = X_surf[(i*Nsurf_nds+j)*CoordDim+k];
          }
          for (Long k = 0; k < Ntrg; k++) {
            Real R2 = 0;
            for (Integer l = 0; l < CoordDim; l++) {
              Real dX = Xt[k*CoordDim+l]- X0[l];
              R2 += dX * dX;
            }
            if (R2 == 0) self_pts.insert(k);
          }
        }
        for (Long trg_idx : self_pts) {
          near_pts.erase(trg_idx);
        }
        for (Long trg_idx : near_pts) {
          pair_lst.PushBack(Pair<Long,Long>(i,trg_idx));
        }
      }
    }

    template <class DensityBasis, class ElemList, class Kernel> static void SetupNearSingular(Matrix<Real>& M_near_singular, Vector<Pair<Long,Long>>& pair_lst, const Vector<Real>& Xt_, const Vector<Long>& trg_surf, const ElemList& elem_lst, const Kernel& kernel, Integer order_singular, Integer order_direct, Real period_length, const Comm& comm) {
      static_assert(std::is_same<Real,typename DensityBasis::ValueType>::value);
      static_assert(std::is_same<Real,typename ElemList::CoordType>::value);
      static_assert(DensityBasis::Dim() == ElemList::ElemDim());
      using CoordBasis = typename ElemList::CoordBasis;
      using CoordEvalOpType = typename CoordBasis::EvalOpType;
      using DensityEvalOpType = typename DensityBasis::EvalOpType;

      constexpr Integer CoordDim = ElemList::CoordDim();
      constexpr Integer ElemDim = ElemList::ElemDim();
      constexpr Integer KDIM0 = Kernel::SrcDim();
      constexpr Integer KDIM1 = Kernel::TrgDim();
      const Long Nelem = elem_lst.NElem();

      BuildNbrList(pair_lst, Xt_, trg_surf, elem_lst, 2.5/order_direct, period_length, comm);
      const Long Ninterac = pair_lst.Dim();

      Vector<Real> Xt;
      { // Set Xt
        Integer rank = comm.Rank();
        Integer np = comm.Size();

        Vector<Long> splitter_ranks;
        { // Set splitter_ranks
          Vector<Long> cnt(np);
          const Long N = Xt_.Dim() / CoordDim;
          comm.Allgather(Ptr2ConstItr<Long>(&N,1), 1, cnt.begin(), 1);
          scan(splitter_ranks, cnt);
        }

        Vector<Long> scatter_index, recv_index, recv_cnt(np), recv_dsp(np);
        { // Set scatter_index, recv_index, recv_cnt, recv_dsp
          { // Set scatter_index, recv_index
            Vector<Pair<Long,Long>> scatter_pair(pair_lst.Dim());
            for (Long i = 0; i < pair_lst.Dim(); i++) {
              scatter_pair[i] = Pair<Long,Long>(pair_lst[i].second,i);
            }
            omp_par::merge_sort(scatter_pair.begin(), scatter_pair.end());

            recv_index.ReInit(scatter_pair.Dim());
            scatter_index.ReInit(scatter_pair.Dim());
            for (Long i = 0; i < scatter_index.Dim(); i++) {
              recv_index[i] = scatter_pair[i].first;
              scatter_index[i] = scatter_pair[i].second;
            }
          }
          for (Integer i = 0; i < np; i++) {
            recv_dsp[i] = std::lower_bound(recv_index.begin(), recv_index.end(), splitter_ranks[i]) - recv_index.begin();
          }
          for (Integer i = 0; i < np-1; i++) {
            recv_cnt[i] = recv_dsp[i+1] - recv_dsp[i];
          }
          recv_cnt[np-1] = recv_index.Dim() - recv_dsp[np-1];
        }

        Vector<Long> send_index, send_cnt(np), send_dsp(np);
        { // Set send_index, send_cnt, send_dsp
          comm.Alltoall(recv_cnt.begin(), 1, send_cnt.begin(), 1);
          scan(send_dsp, send_cnt);
          send_index.ReInit(send_cnt[np-1] + send_dsp[np-1]);
          comm.Alltoallv(recv_index.begin(), recv_cnt.begin(), recv_dsp.begin(), send_index.begin(), send_cnt.begin(), send_dsp.begin());
        }

        Vector<Real> Xt_send(send_index.Dim() * CoordDim);
        for (Long i = 0; i < send_index.Dim(); i++) { // Set Xt_send
          Long idx = send_index[i] - splitter_ranks[rank];
          for (Integer k = 0; k < CoordDim; k++) {
            Xt_send[i*CoordDim+k] = Xt_[idx*CoordDim+k];
          }
        }

        Vector<Real> Xt_recv(recv_index.Dim() * CoordDim);
        { // Set Xt_recv
          for (Long i = 0; i < np; i++) {
            send_cnt[i] *= CoordDim;
            send_dsp[i] *= CoordDim;
            recv_cnt[i] *= CoordDim;
            recv_dsp[i] *= CoordDim;
          }
          comm.Alltoallv(Xt_send.begin(), send_cnt.begin(), send_dsp.begin(), Xt_recv.begin(), recv_cnt.begin(), recv_dsp.begin());
        }

        Xt.ReInit(scatter_index.Dim() * CoordDim);
        for (Long i = 0; i < scatter_index.Dim(); i++) { // Set Xt
          Long idx = scatter_index[i];
          for (Integer k = 0; k < CoordDim; k++) {
            Xt[idx*CoordDim+k] = Xt_recv[i*CoordDim+k];
          }
        }
      }

      const Vector<CoordBasis>& X = elem_lst.ElemVector();
      Vector<CoordBasis> dX;
      CoordBasis::Grad(dX, X);

      Long elem_rank_offset;
      { // Set elem_rank_offset
        comm.Scan(Ptr2ConstItr<Long>(&Nelem,1), Ptr2Itr<Long>(&elem_rank_offset,1), 1, Comm::CommOp::SUM);
        elem_rank_offset -= Nelem;
      }

      auto& M = M_near_singular;
      M.ReInit(Ninterac * KDIM0 * DensityBasis::Size(), KDIM1);
      #pragma omp parallel for schedule(static)
      for (Long j = 0; j < Ninterac; j++) { // Set M (near-singular)
        const Long src_idx = pair_lst[j].first - elem_rank_offset;

        Real adapt = -1.0;
        Tensor<Real,true,ElemDim,1> u0;
        { // Set u0 (project target point to the surface patch in parameter space)
          ConstIterator<Real> Xt_ = Xt.begin() + j * CoordDim;
          const auto& nodes = CoordBasis::Nodes();

          Long min_idx = -1;
          Real min_R2 = 1e10;
          for (Long i = 0; i < CoordBasis::Size(); i++) {
            Real R2 = 0;
            for (Integer k = 0; k < CoordDim; k++) {
              Real dX = X[src_idx * CoordDim + k][i] - Xt_[k];
              R2 += dX * dX;
            }
            if (R2 < min_R2) {
              min_R2 = R2;
              min_idx = i;
            }
          }
          SCTL_ASSERT(min_idx >= 0);
          for (Integer k = 0; k < ElemDim; k++) {
            u0(k,0) = nodes[k][min_idx];
          }

          for (Integer i = 0; i < 2; i++) { // iterate
            Matrix<Real> X_, dX_;
            for (Integer k = 0; k < ElemDim; k++) {
              u0(k,0) = std::min(1.0, u0(k,0));
              u0(k,0) = std::max(0.0, u0(k,0));
            }
            const auto eval_op = CoordBasis::SetupEval(Matrix<Real>(ElemDim,1,u0.begin(),false));
            CoordBasis::Eval(X_, Vector<CoordBasis>(CoordDim,(Iterator<CoordBasis>)X.begin()+src_idx*CoordDim,false),eval_op);
            CoordBasis::Eval(dX_, Vector<CoordBasis>(CoordDim*ElemDim,dX.begin()+src_idx*CoordDim*ElemDim,false),eval_op);

            const Tensor<Real,false,CoordDim,1> x0((Iterator<Real>)Xt_);
            const Tensor<Real,false,CoordDim,1> x(X_.begin());
            const Tensor<Real,false,CoordDim,ElemDim> x_u(dX_.begin());
            auto inv = [](const Tensor<Real,true,2,2>& M) {
              Tensor<Real,true,2,2> Minv;
              Real det_inv = 1.0 / (M(0,0)*M(1,1) - M(1,0)*M(0,1));
              Minv(0,0) = M(1,1) * det_inv;
              Minv(0,1) =-M(0,1) * det_inv;
              Minv(1,0) =-M(1,0) * det_inv;
              Minv(1,1) = M(0,0) * det_inv;
              return Minv;
            };
            auto du = inv(x_u.RotateRight()*x_u) * x_u.RotateRight()*(x0-x);
            u0 = u0 + du;

            auto x_u_squared = x_u.RotateRight() * x_u;
            adapt = sctl::sqrt<Real>( ((x0-x).RotateRight()*(x0-x))(0,0) / std::max<Real>(x_u_squared(0,0),x_u_squared(1,1)) );
          }
        }

        Matrix<Real> quad_nds;
        Vector<Real> quad_wts;
        DuffyQuad<ElemDim>(quad_nds, quad_wts, Vector<Real>(ElemDim,u0.begin(),false), order_singular, adapt);
        const CoordEvalOpType CoordEvalOp = CoordBasis::SetupEval(quad_nds);
        Integer Nnds = quad_wts.Dim();

        Vector<Real> X_, dX_, Xa_, Xn_;
        { // Set X_, dX_
          const Vector<CoordBasis> X__(CoordDim, (Iterator<CoordBasis>)X.begin() + src_idx * CoordDim, false);
          const Vector<CoordBasis> dX__(CoordDim * ElemDim, (Iterator<CoordBasis>)dX.begin() + src_idx * CoordDim * ElemDim, false);
          eval_basis(X_, X__, CoordDim, Nnds, CoordEvalOp);
          eval_basis(dX_, dX__, CoordDim * ElemDim, Nnds, CoordEvalOp);
        }
        if (CoordDim == 3 && ElemDim == 2) { // Compute Xa_, Xn_
          Xa_.ReInit(Nnds);
          Xn_.ReInit(Nnds*CoordDim);
          for (Long j = 0; j < Nnds; j++) {
            StaticArray<Real,CoordDim> normal;
            normal[0] = dX_[j*6+2]*dX_[j*6+5] - dX_[j*6+4]*dX_[j*6+3];
            normal[1] = dX_[j*6+4]*dX_[j*6+1] - dX_[j*6+0]*dX_[j*6+5];
            normal[2] = dX_[j*6+0]*dX_[j*6+3] - dX_[j*6+2]*dX_[j*6+1];
            Xa_[j] = sctl::sqrt<Real>(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
            Real invXa = 1/Xa_[j];
            Xn_[j*3+0] = normal[0] * invXa;
            Xn_[j*3+1] = normal[1] * invXa;
            Xn_[j*3+2] = normal[2] * invXa;
          }
        }

        DensityEvalOpType DensityEvalOp;
        if (std::is_same<CoordBasis,DensityBasis>::value) {
          DensityEvalOp = CoordEvalOp;
        } else {
          DensityEvalOp = DensityBasis::SetupEval(quad_nds);
        }

        Matrix<Real> M__(Nnds * KDIM0, KDIM1);
        { // Set kernel matrix M__
          const Vector<Real> X0_(CoordDim, (Iterator<Real>)Xt.begin() + j * CoordDim, false);
          kernel.template KernelMatrix<Real>(M__, X0_, X_, Xn_);
        }
        for (Long k0 = 0; k0 < KDIM0; k0++) {
          for (Long k1 = 0; k1 < KDIM1; k1++) {
            for (Long l = 0; l < DensityBasis::Size(); l++) {
              Real M_lk = 0;
              for (Long n = 0; n < Nnds; n++) {
                Real quad_wt = Xa_[n] * quad_wts[n];
                M_lk += DensityEvalOp[l][n] * quad_wt * M__[n*KDIM0+k0][k1];
              }
              M[(j * KDIM0 + k0) * DensityBasis::Size() + l][k1] = M_lk;
            }
          }
        }
      }
      { // Set M (subtract direct)
        Matrix<Real> quad_nds;
        Vector<Real> quad_wts;
        TensorProductGaussQuad<ElemDim>(quad_nds, quad_wts, order_direct);
        const CoordEvalOpType CoordEvalOp = CoordBasis::SetupEval(quad_nds);
        Integer Nnds = quad_wts.Dim();

        Vector<Real> X_, dX_, Xa_, Xn_;
        { // Set X_, dX_
          eval_basis(X_, X, CoordDim, Nnds, CoordEvalOp);
          eval_basis(dX_, dX, CoordDim * ElemDim, Nnds, CoordEvalOp);
        }
        if (CoordDim == 3 && ElemDim == 2) { // Compute Xa_, Xn_
          Long N = Nelem*Nnds;
          Xa_.ReInit(N);
          Xn_.ReInit(N*CoordDim);
          for (Long j = 0; j < N; j++) {
            StaticArray<Real,CoordDim> normal;
            normal[0] = dX_[j*6+2]*dX_[j*6+5] - dX_[j*6+4]*dX_[j*6+3];
            normal[1] = dX_[j*6+4]*dX_[j*6+1] - dX_[j*6+0]*dX_[j*6+5];
            normal[2] = dX_[j*6+0]*dX_[j*6+3] - dX_[j*6+2]*dX_[j*6+1];
            Xa_[j] = sctl::sqrt<Real>(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
            Real invXa = 1/Xa_[j];
            Xn_[j*3+0] = normal[0] * invXa;
            Xn_[j*3+1] = normal[1] * invXa;
            Xn_[j*3+2] = normal[2] * invXa;
          }
        }

        DensityEvalOpType DensityEvalOp;
        if (std::is_same<CoordBasis,DensityBasis>::value) {
          DensityEvalOp = CoordEvalOp;
        } else {
          DensityEvalOp = DensityBasis::SetupEval(quad_nds);
        }

        #pragma omp parallel for schedule(static)
        for (Long j = 0; j < Ninterac; j++) { // Subtract direct contribution
          const Long src_idx = pair_lst[j].first - elem_rank_offset;

          Matrix<Real> M__(Nnds * KDIM0, KDIM1);
          { // Set kernel matrix M__
            const Vector<Real> X0_(CoordDim, (Iterator<Real>)Xt.begin() + j * CoordDim, false);
            Vector<Real> X__(Nnds * CoordDim, X_.begin() + src_idx * Nnds * CoordDim, false);
            Vector<Real> Xn__(Nnds * CoordDim, Xn_.begin() + src_idx * Nnds * CoordDim, false);
            kernel.template KernelMatrix<Real>(M__, X0_, X__, Xn__);
          }
          for (Long k0 = 0; k0 < KDIM0; k0++) {
            for (Long k1 = 0; k1 < KDIM1; k1++) {
              for (Long l = 0; l < DensityBasis::Size(); l++) {
                Real M_lk = 0;
                for (Long n = 0; n < Nnds; n++) {
                  Real quad_wt = Xa_[src_idx * Nnds + n] * quad_wts[n];
                  M_lk += DensityEvalOp[l][n] * quad_wt * M__[n*KDIM0+k0][k1];
                }
                M[(j * KDIM0 + k0) * DensityBasis::Size() + l][k1] -= M_lk;
              }
            }
          }
        }
      }
    }

    template <class DensityBasis> static void EvalNearSingular(Vector<Real>& U, const Vector<DensityBasis>& density, const Matrix<Real>& M, const Vector<Pair<Long,Long>>& pair_lst, Long Nelem_, Long Ntrg_, Integer KDIM0_, Integer KDIM1_, const Comm& comm) {
      const Long Ninterac = pair_lst.Dim();
      const Integer dof = density.Dim() / Nelem_ / KDIM0_;
      SCTL_ASSERT(density.Dim() == Nelem_ * dof * KDIM0_);

      Long elem_rank_offset;
      { // Set elem_rank_offset
        comm.Scan(Ptr2ConstItr<Long>(&Nelem_,1), Ptr2Itr<Long>(&elem_rank_offset,1), 1, Comm::CommOp::SUM);
        elem_rank_offset -= Nelem_;
      }

      Vector<Real> U_loc(Ninterac*dof*KDIM1_);
      for (Long j = 0; j < Ninterac; j++) {
        const Long src_idx = pair_lst[j].first - elem_rank_offset;
        const Matrix<Real> M_(KDIM0_ * DensityBasis::Size(), KDIM1_, (Iterator<Real>)M[j * KDIM0_ * DensityBasis::Size()], false);
        Matrix<Real> U_(dof, KDIM1_, U_loc.begin() + j*dof*KDIM1_, false);
        Matrix<Real> F_(dof, KDIM0_ * DensityBasis::Size());
        for (Long i = 0; i < dof; i++) {
          for (Long k = 0; k < KDIM0_; k++) {
            for (Long l = 0; l < DensityBasis::Size(); l++) {
              F_[i][k * DensityBasis::Size() + l] = density[(src_idx * dof + i) * KDIM0_ + k][l];
            }
          }
        }
        Matrix<Real>::GEMM(U_, F_, M_);
      }

      if (U.Dim() != Ntrg_ * dof * KDIM1_) {
        U.ReInit(Ntrg_ * dof * KDIM1_);
        U = 0;
      }
      { // Set U
        Integer rank = comm.Rank();
        Integer np = comm.Size();

        Vector<Long> splitter_ranks;
        { // Set splitter_ranks
          Vector<Long> cnt(np);
          comm.Allgather(Ptr2ConstItr<Long>(&Ntrg_,1), 1, cnt.begin(), 1);
          scan(splitter_ranks, cnt);
        }

        Vector<Long> scatter_index, send_index, send_cnt(np), send_dsp(np);
        { // Set scatter_index, send_index, send_cnt, send_dsp
          { // Set scatter_index, send_index
            Vector<Pair<Long,Long>> scatter_pair(pair_lst.Dim());
            for (Long i = 0; i < pair_lst.Dim(); i++) {
              scatter_pair[i] = Pair<Long,Long>(pair_lst[i].second,i);
            }
            omp_par::merge_sort(scatter_pair.begin(), scatter_pair.end());

            send_index.ReInit(scatter_pair.Dim());
            scatter_index.ReInit(scatter_pair.Dim());
            for (Long i = 0; i < scatter_index.Dim(); i++) {
              send_index[i] = scatter_pair[i].first;
              scatter_index[i] = scatter_pair[i].second;
            }
          }
          for (Integer i = 0; i < np; i++) {
            send_dsp[i] = std::lower_bound(send_index.begin(), send_index.end(), splitter_ranks[i]) - send_index.begin();
          }
          for (Integer i = 0; i < np-1; i++) {
            send_cnt[i] = send_dsp[i+1] - send_dsp[i];
          }
          send_cnt[np-1] = send_index.Dim() - send_dsp[np-1];
        }

        Vector<Long> recv_index, recv_cnt(np), recv_dsp(np);
        { // Set recv_index, recv_cnt, recv_dsp
          comm.Alltoall(send_cnt.begin(), 1, recv_cnt.begin(), 1);
          scan(recv_dsp, recv_cnt);
          recv_index.ReInit(recv_cnt[np-1] + recv_dsp[np-1]);
          comm.Alltoallv(send_index.begin(), send_cnt.begin(), send_dsp.begin(), recv_index.begin(), recv_cnt.begin(), recv_dsp.begin());
        }

        Vector<Real> U_send(scatter_index.Dim() * dof * KDIM1_);
        for (Long i = 0; i < scatter_index.Dim(); i++) {
          Long idx = scatter_index[i]*dof*KDIM1_;
          for (Long k = 0; k < dof * KDIM1_; k++) {
            U_send[i*dof*KDIM1_ + k] = U_loc[idx + k];
          }
        }

        Vector<Real> U_recv(recv_index.Dim() * dof * KDIM1_);
        { // Set U_recv
          for (Long i = 0; i < np; i++) {
            send_cnt[i] *= dof * KDIM1_;
            send_dsp[i] *= dof * KDIM1_;
            recv_cnt[i] *= dof * KDIM1_;
            recv_dsp[i] *= dof * KDIM1_;
          }
          comm.Alltoallv(U_send.begin(), send_cnt.begin(), send_dsp.begin(), U_recv.begin(), recv_cnt.begin(), recv_dsp.begin());
        }

        for (Long i = 0; i < recv_index.Dim(); i++) { // Set U
          Long idx = (recv_index[i] - splitter_ranks[rank]) * dof * KDIM1_;
          for (Integer k = 0; k < dof * KDIM1_; k++) {
            U[idx + k] += U_recv[i*dof*KDIM1_ + k];
          }
        }
      }
    }



    template <class ElemList, class DensityBasis, class Kernel> static void Direct(Vector<Real>& U, const Vector<Real>& Xt, const ElemList& elem_lst, const Vector<DensityBasis>& density, const Kernel& kernel, Integer order_direct, const Comm& comm) {
      using CoordBasis = typename ElemList::CoordBasis;
      using CoordEvalOpType = typename CoordBasis::EvalOpType;
      using DensityEvalOpType = typename DensityBasis::EvalOpType;

      constexpr Integer CoordDim = ElemList::CoordDim();
      constexpr Integer ElemDim = ElemList::ElemDim();
      constexpr Integer KDIM0 = Kernel::SrcDim();
      constexpr Integer KDIM1 = Kernel::TrgDim();
      const Long Nelem = elem_lst.NElem();
      const Integer dof = density.Dim() / Nelem / KDIM0;
      SCTL_ASSERT(density.Dim() == Nelem * dof * KDIM0);

      Matrix<Real> quad_nds;
      Vector<Real> quad_wts;
      TensorProductGaussQuad<ElemDim>(quad_nds, quad_wts, order_direct);
      const CoordEvalOpType CoordEvalOp = CoordBasis::SetupEval(quad_nds);
      Integer Nnds = quad_wts.Dim();

      const Vector<CoordBasis>& X = elem_lst.ElemVector();
      Vector<CoordBasis> dX;
      CoordBasis::Grad(dX, X);

      Vector<Real> X_, dX_, Xa_, Xn_;
      eval_basis(X_, X, CoordDim, Nnds, CoordEvalOp);
      eval_basis(dX_, dX, CoordDim*ElemDim, Nnds, CoordEvalOp);
      if (CoordDim == 3 && ElemDim == 2) { // Compute Xa_, Xn_
        Long N = Nelem*Nnds;
        Xa_.ReInit(N);
        Xn_.ReInit(N*CoordDim);
        for (Long j = 0; j < N; j++) {
          StaticArray<Real,CoordDim> normal;
          normal[0] = dX_[j*6+2]*dX_[j*6+5] - dX_[j*6+4]*dX_[j*6+3];
          normal[1] = dX_[j*6+4]*dX_[j*6+1] - dX_[j*6+0]*dX_[j*6+5];
          normal[2] = dX_[j*6+0]*dX_[j*6+3] - dX_[j*6+2]*dX_[j*6+1];
          Xa_[j] = sctl::sqrt<Real>(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
          Real invXa = 1/Xa_[j];
          Xn_[j*3+0] = normal[0] * invXa;
          Xn_[j*3+1] = normal[1] * invXa;
          Xn_[j*3+2] = normal[2] * invXa;
        }
      }

      Vector<Real> Fa_;
      { // Set Fa_
        Vector<Real> F_;
        if (std::is_same<CoordBasis,DensityBasis>::value) {
          eval_basis(F_, density, dof * KDIM0, Nnds, CoordEvalOp);
        } else {
          const DensityEvalOpType EvalOp = DensityBasis::SetupEval(quad_nds);
          eval_basis(F_, density, dof * KDIM0, Nnds, EvalOp);
        }

        Fa_.ReInit(F_.Dim());
        const Integer DensityDOF = dof * KDIM0;
        SCTL_ASSERT(F_.Dim() == Nelem * Nnds * DensityDOF);
        for (Long j = 0; j < Nelem; j++) {
          for (Integer k = 0; k < Nnds; k++) {
            Long idx = j * Nnds + k;
            Real quad_wt = Xa_[idx] * quad_wts[k];
            for (Integer l = 0; l < DensityDOF; l++) {
              Fa_[idx * DensityDOF + l] = F_[idx * DensityDOF + l] * quad_wt;
            }
          }
        }
      }

      { // Evaluate potential
        const Long Ntrg = Xt.Dim() / CoordDim;
        SCTL_ASSERT(Xt.Dim() == Ntrg * CoordDim);
        if (U.Dim() != Ntrg * dof * KDIM1) {
          U.ReInit(Ntrg * dof * KDIM1);
          U = 0;
        }
        ParticleFMM<Real,CoordDim>::Eval(U, Xt, X_, Xn_, Fa_, kernel, comm);
      }
    }

  public:

    template <class DensityBasis, class ElemList, class Kernel> void Setup(const ElemList& elem_lst, const Vector<Real>& Xt, const Kernel& kernel, Integer order_singular, Integer order_direct, Real period_length, const Comm& comm) {
      order_direct_ = order_direct;
      period_length_ = period_length;
      comm_ = comm;

      Profile::Tic("Setup", &comm_);
      static_assert(std::is_same<Real,typename DensityBasis::ValueType>::value);
      static_assert(std::is_same<Real,typename ElemList::CoordType>::value);
      static_assert(DensityBasis::Dim() == ElemList::ElemDim());

      Xt_ = Xt;
      M_singular.ReInit(0,0);

      Profile::Tic("SetupNearSingular", &comm_);
      SetupNearSingular<DensityBasis>(M_near_singular, pair_lst, Xt_, Vector<Long>(), elem_lst, kernel, order_singular, order_direct_, period_length_, comm_);
      Profile::Toc();

      Profile::Toc();
    }

    template <class DensityBasis, class PotentialBasis, class ElemList, class Kernel> void Setup(const ElemList& elem_lst, const Kernel& kernel, Integer order_singular, Integer order_direct, Real period_length, const Comm& comm) {
      order_direct_ = order_direct;
      period_length_ = period_length;
      comm_ = comm;

      Profile::Tic("Setup", &comm_);
      static_assert(std::is_same<Real,typename PotentialBasis::ValueType>::value);
      static_assert(std::is_same<Real,typename DensityBasis::ValueType>::value);
      static_assert(std::is_same<Real,typename ElemList::CoordType>::value);
      static_assert(PotentialBasis::Dim() == ElemList::ElemDim());
      static_assert(DensityBasis::Dim() == ElemList::ElemDim());

      Vector<Long> trg_surf;
      { // Set Xt_
        using CoordBasis = typename ElemList::CoordBasis;
        Matrix<Real> trg_nds = PotentialBasis::Nodes();
        auto Meval = CoordBasis::SetupEval(trg_nds);
        eval_basis(Xt_, elem_lst.ElemVector(), ElemList::CoordDim(), trg_nds.Dim(1), Meval);

        { // Set trg_surf
          const Long Nelem = elem_lst.NElem();
          const Long Nnds  = trg_nds.Dim(1);
          Long elem_offset;
          { // Set elem_offset
            comm.Scan(Ptr2ConstItr<Long>(&Nelem,1), Ptr2Itr<Long>(&elem_offset,1), 1, Comm::CommOp::SUM);
            elem_offset -= Nelem;
          }
          trg_surf.ReInit(elem_lst.NElem() * trg_nds.Dim(1));
          for (Long i = 0; i < Nelem; i++) {
            for (Long j = 0; j < Nnds; j++) {
              trg_surf[i*Nnds+j] = elem_offset + i;
            }
          }
        }
      }

      Profile::Tic("SetupSingular", &comm_);
      SetupSingular<DensityBasis>(M_singular, PotentialBasis::Nodes(), elem_lst, kernel, order_singular, order_direct_);
      Profile::Toc();

      Profile::Tic("SetupNearSingular", &comm_);
      SetupNearSingular<DensityBasis>(M_near_singular, pair_lst, Xt_, trg_surf, elem_lst, kernel, order_singular, order_direct_, period_length_, comm_);
      Profile::Toc();

      Profile::Toc();
    }

    template <class DensityBasis, class PotentialBasis, class ElemList, class Kernel> void Eval(Vector<PotentialBasis>& U, const ElemList& elements, const Vector<DensityBasis>& F, const Kernel& kernel) {
      Profile::Tic("Eval", &comm_);
      Matrix<Real> U_singular;
      Vector<Real> U_direct, U_near_sing;

      Profile::Tic("EvalDirect", &comm_);
      Direct(U_direct, Xt_, elements, F, kernel, order_direct_, comm_);
      Profile::Toc();

      Profile::Tic("EvalSingular", &comm_);
      EvalSingular(U_singular, F, M_singular, kernel.SrcDim(), kernel.TrgDim());
      Profile::Toc();

      Profile::Tic("EvalNearSingular", &comm_);
      EvalNearSingular(U_near_sing, F, M_near_singular, pair_lst, elements.NElem(), Xt_.Dim() / ElemList::CoordDim(), kernel.SrcDim(), kernel.TrgDim(), comm_);
      SCTL_ASSERT(U_near_sing.Dim() == U_direct.Dim());
      Profile::Toc();

      if (U.Dim() != elements.NElem() * kernel.TrgDim()) {
        U.ReInit(elements.NElem() * kernel.TrgDim());
      }
      for (int i = 0; i < elements.NElem(); i++) {
        for (int j = 0; j < PotentialBasis::Size(); j++) {
          for (int k = 0; k < kernel.TrgDim(); k++) {
            Real& U_ = U[i*kernel.TrgDim()+k][j];
            U_ = 0;
            U_ += U_direct   [(i*PotentialBasis::Size()+j)*kernel.TrgDim()+k];
            U_ += U_near_sing[(i*PotentialBasis::Size()+j)*kernel.TrgDim()+k];
            U_ += U_singular[i*kernel.TrgDim()+k][j];
            U_ *= kernel.template ScaleFactor<Real>();
          }
        }
      }
      Profile::Toc();
    }

    template <class DensityBasis, class ElemList, class Kernel> void Eval(Vector<Real>& U, const ElemList& elements, const Vector<DensityBasis>& F, const Kernel& kernel) {
      Profile::Tic("Eval", &comm_);
      Matrix<Real> U_singular;
      Vector<Real> U_direct, U_near_sing;

      Profile::Tic("EvalDirect", &comm_);
      Direct(U_direct, Xt_, elements, F, kernel, order_direct_, comm_);
      Profile::Toc();

      Profile::Tic("EvalSingular", &comm_);
      EvalSingular(U_singular, F, M_singular, kernel.SrcDim(), kernel.TrgDim());
      Profile::Toc();

      Profile::Tic("EvalNearSingular", &comm_);
      EvalNearSingular(U_near_sing, F, M_near_singular, pair_lst, elements.NElem(), Xt_.Dim() / ElemList::CoordDim(), kernel.SrcDim(), kernel.TrgDim(), comm_);
      SCTL_ASSERT(U_near_sing.Dim() == U_direct.Dim());
      Profile::Toc();

      if (U.Dim() != U_direct.Dim()) {
        U.ReInit(U_direct.Dim());
      }
      for (int i = 0; i < U.Dim(); i++) {
        U[i] = (U_direct[i] + U_near_sing[i]) * kernel.template ScaleFactor<Real>();
      }
      if (U_singular.Dim(1)) {
        for (int i = 0; i < elements.NElem(); i++) {
          for (int j = 0; j < U_singular.Dim(1); j++) {
            for (int k = 0; k < kernel.TrgDim(); k++) {
              Real& U_ = U[(i*U_singular.Dim(1)+j)*kernel.TrgDim()+k];
              U_ += U_singular[i*kernel.TrgDim()+k][j] * kernel.template ScaleFactor<Real>();
            }
          }
        }
      }
      Profile::Toc();
    }

    template <Integer ORDER = 5> static void test(Integer order_singular = 10, Integer order_direct = 5, const Comm& comm = Comm::World()) {
      constexpr Integer COORD_DIM = 3;
      constexpr Integer ELEM_DIM = COORD_DIM-1;
      using ElemList = ElemList<COORD_DIM, Basis<Real, ELEM_DIM, ORDER>>;
      using DensityBasis = Basis<Real, ELEM_DIM, ORDER>;
      using PotentialBasis = Basis<Real, ELEM_DIM, ORDER>;

      int np = comm.Size();
      int rank = comm.Rank();
      auto build_torus = [rank,np](ElemList& elements, long Nt, long Np, Real Rmajor, Real Rminor){
        auto nodes = ElemList::CoordBasis::Nodes();
        auto torus = [](Real theta, Real phi, Real Rmajor, Real Rminor) {
          Real R = Rmajor + Rminor * cos<Real>(phi);
          Real X = R * cos<Real>(theta);
          Real Y = R * sin<Real>(theta);
          Real Z = Rminor * sin<Real>(phi);
          return std::make_tuple(X,Y,Z);
        };

        long start = Nt*Np*(rank+0)/np;
        long end   = Nt*Np*(rank+1)/np;
        elements.ReInit(end - start);
        for (long ii = start; ii < end; ii++) {
          long i = ii / Np;
          long j = ii % Np;
          for (int k = 0; k < nodes.Dim(1); k++) {
            Real X, Y, Z;
            Real theta = 2 * const_pi<Real>() * (i + nodes[0][k]) / Nt;
            Real phi   = 2 * const_pi<Real>() * (j + nodes[1][k]) / Np;
            std::tie(X,Y,Z) = torus(theta, phi, Rmajor, Rminor);
            elements(ii-start,0)[k] = X;
            elements(ii-start,1)[k] = Y;
            elements(ii-start,2)[k] = Z;
          }
        }
      };
      ElemList elements_src, elements_trg;
      build_torus(elements_src, 28, 16, 2, 1.0);
      build_torus(elements_trg, 29, 17, 2, 0.99);

      Vector<Real> Xt;
      Vector<PotentialBasis> U_onsurf, U_offsurf;
      Vector<DensityBasis> density_sl, density_dl;
      { // Set Xt, elements_src, elements_trg, density_sl, density_dl, U
        Real X0[COORD_DIM] = {3,2,1};
        std::function<void(Real*,Real*,Real*)> potential = [X0](Real* U, Real* X, Real* Xn) {
          Real dX[COORD_DIM] = {X[0]-X0[0],X[1]-X0[1],X[2]-X0[2]};
          Real Rinv = 1/sqrt(dX[0]*dX[0]+dX[1]*dX[1]+dX[2]*dX[2]);
          U[0] = Rinv;
        };
        std::function<void(Real*,Real*,Real*)> potential_normal_derivative = [X0](Real* U, Real* X, Real* Xn) {
          Real dX[COORD_DIM] = {X[0]-X0[0],X[1]-X0[1],X[2]-X0[2]};
          Real Rinv = 1/sqrt(dX[0]*dX[0]+dX[1]*dX[1]+dX[2]*dX[2]);
          Real RdotN = dX[0]*Xn[0]+dX[1]*Xn[1]+dX[2]*Xn[2];
          U[0] = -RdotN * Rinv*Rinv*Rinv;
        };

        DiscretizeSurfaceFn<COORD_DIM,1>(density_sl, elements_src, potential_normal_derivative);
        DiscretizeSurfaceFn<COORD_DIM,1>(density_dl, elements_src, potential);
        DiscretizeSurfaceFn<COORD_DIM,1>(U_onsurf  , elements_src, potential);
        DiscretizeSurfaceFn<COORD_DIM,1>(U_offsurf , elements_trg, potential);

        for (long i = 0; i < elements_trg.NElem(); i++) { // Set Xt
          for (long j = 0; j < PotentialBasis::Size(); j++) {
            for (int k = 0; k < COORD_DIM; k++) {
              Xt.PushBack(elements_trg(i,k)[j]);
            }
          }
        }
      }

      Laplace3D_DxU Laplace_DxU;
      Laplace3D_FxU Laplace_FxU;

      Profile::Enable(true);
      if (1) { // Greeen's identity test (Laplace, on-surface)
        Profile::Tic("OnSurface", &comm);
        Quadrature<Real> quadrature_DxU, quadrature_FxU;
        quadrature_FxU.Setup<DensityBasis, PotentialBasis>(elements_src, Laplace_FxU, order_singular, order_direct, -1.0, comm);
        quadrature_DxU.Setup<DensityBasis, PotentialBasis>(elements_src, Laplace_DxU, order_singular, order_direct, -1.0, comm);

        Vector<PotentialBasis> U_sl, U_dl;
        quadrature_FxU.Eval(U_sl, elements_src, density_sl, Laplace_FxU);
        quadrature_DxU.Eval(U_dl, elements_src, density_dl, Laplace_DxU);
        Profile::Toc();

        Real max_err = 0;
        Vector<PotentialBasis> err(U_onsurf.Dim());
        for (long i = 0; i < U_sl.Dim(); i++) {
          for (long j = 0; j < PotentialBasis::Size(); j++) {
            err[i][j] = 0.5*U_onsurf[i][j] - (U_sl[i][j] + U_dl[i][j]);
            max_err = std::max<Real>(max_err, fabs(err[i][j]));
          }
        }
        { // Print error
          Real glb_err;
          comm.Allreduce(Ptr2ConstItr<Real>(&max_err,1), Ptr2Itr<Real>(&glb_err,1), 1, Comm::CommOp::MAX);
          if (!comm.Rank()) std::cout<<"Error = "<<glb_err<<'\n';
        }
        { // Write VTK output
          VTUData vtu;
          vtu.AddElems(elements_src, err, ORDER);
          vtu.WriteVTK("err", comm);
        }
        { // Write VTK output
          VTUData vtu;
          vtu.AddElems(elements_src, U_onsurf, ORDER);
          vtu.WriteVTK("U", comm);
        }
      }
      if (1) { // Greeen's identity test (Laplace, off-surface)
        Profile::Tic("OffSurface", &comm);
        Quadrature<Real> quadrature_DxU, quadrature_FxU;
        quadrature_FxU.Setup<DensityBasis>(elements_src, Xt, Laplace_FxU, order_singular, order_direct, -1.0, comm);
        quadrature_DxU.Setup<DensityBasis>(elements_src, Xt, Laplace_DxU, order_singular, order_direct, -1.0, comm);

        Vector<Real> U_sl, U_dl;
        quadrature_FxU.Eval(U_sl, elements_src, density_sl, Laplace_FxU);
        quadrature_DxU.Eval(U_dl, elements_src, density_dl, Laplace_DxU);
        Profile::Toc();

        Real max_err = 0;
        Vector<PotentialBasis> err(elements_trg.NElem());
        for (long i = 0; i < elements_trg.NElem(); i++) {
          for (long j = 0; j < PotentialBasis::Size(); j++) {
            err[i][j] = U_offsurf[i][j] - (U_sl[i*PotentialBasis::Size()+j] + U_dl[i*PotentialBasis::Size()+j]);
            max_err = std::max<Real>(max_err, fabs(err[i][j]));
          }
        }
        { // Print error
          Real glb_err;
          comm.Allreduce(Ptr2ConstItr<Real>(&max_err,1), Ptr2Itr<Real>(&glb_err,1), 1, Comm::CommOp::MAX);
          if (!comm.Rank()) std::cout<<"Error = "<<glb_err<<'\n';
        }
        { // Write VTK output
          VTUData vtu;
          vtu.AddElems(elements_trg, err, ORDER);
          vtu.WriteVTK("err", comm);
        }
        { // Write VTK output
          VTUData vtu;
          vtu.AddElems(elements_trg, U_offsurf, ORDER);
          vtu.WriteVTK("U", comm);
        }
      }
      Profile::print(&comm);
    }

  private:

    static void scan(Vector<Long>& dsp, const Vector<Long>& cnt) {
      dsp.ReInit(cnt.Dim());
      if (cnt.Dim()) dsp[0] = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());
    }

    template <class Basis> static void eval_basis(Vector<Real>& value, const Vector<Basis> X, Integer dof, Integer Nnds, const typename Basis::EvalOpType& EvalOp) {
      Long Nelem = X.Dim() / dof;
      SCTL_ASSERT(X.Dim() == Nelem * dof);

      value.ReInit(Nelem*Nnds*dof);
      Matrix<Real> X_(Nelem*dof, Nnds, value.begin(),false);
      Basis::Eval(X_, X, EvalOp);
      for (Long j = 0; j < Nelem; j++) { // Rearrange data
        Matrix<Real> X(Nnds, dof, X_[j*dof], false);
        X = Matrix<Real>(dof, Nnds, X_[j*dof], false).Transpose();
      }
    }

    template <int CoordDim, int FnDim, class FnBasis, class ElemList> static void DiscretizeSurfaceFn(Vector<FnBasis>& U, const ElemList& elements, std::function<void(Real*,Real*,Real*)> fn) {
      using CoordBasis = typename ElemList::CoordBasis;
      const long Nelem = elements.NElem();
      U.ReInit(Nelem * FnDim);

      Matrix<Real> X, X_grad;
      { // Set X, X_grad
        Vector<CoordBasis> coord = elements.ElemVector();
        Vector<CoordBasis> coord_grad;
        CoordBasis::Grad(coord_grad, coord);

        const auto Meval = CoordBasis::SetupEval(FnBasis::Nodes());
        CoordBasis::Eval(X, coord, Meval);
        CoordBasis::Eval(X_grad, coord_grad, Meval);
      }

      for (long i = 0; i < Nelem; i++) {
        for (long j = 0; j < FnBasis::Size(); j++) {
          Real X_[CoordDim], Xn[CoordDim], U_[FnDim];
          for (long k = 0; k < CoordDim; k++) {
            X_[k] = X[i*CoordDim+k][j];
          }
          { // Set Xn
            Real Xu[CoordDim], Xv[CoordDim];
            for (long k = 0; k < CoordDim; k++) {
              Xu[k] = X_grad[(i*CoordDim+k)*2+0][j];
              Xv[k] = X_grad[(i*CoordDim+k)*2+1][j];
            }
            Real dA = 0;
            for (long k = 0; k < CoordDim; k++) {
              Xn[k] = Xu[(k+1)%CoordDim] * Xv[(k+2)%CoordDim];
              Xn[k] -= Xv[(k+1)%CoordDim] * Xu[(k+2)%CoordDim];
              dA += Xn[k] * Xn[k];
            }
            dA = sqrt(dA);
            for (long k = 0; k < CoordDim; k++) {
              Xn[k] /= dA;
            }
          }
          fn(U_, X_, Xn);
          for (long k = 0; k < FnDim; k++) {
            U[i*FnDim+k][j] = U_[k];
          }
        }
      }
    }

    Vector<Real> Xt_;
    Matrix<Real> M_singular;
    Matrix<Real> M_near_singular;
    Vector<Pair<Long,Long>> pair_lst;
    Integer order_direct_;

    Real period_length_;
    Comm comm_;
};

}  // end namespace

#endif  //_SCTL_BOUNDARY_QUADRATURE_HPP_
