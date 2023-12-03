#ifndef _SCTL_SLENDER_ELEMENT_HPP_
#define _SCTL_SLENDER_ELEMENT_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(mem_mgr.hpp)
#include SCTL_INCLUDE(vector.hpp)

namespace SCTL_NAMESPACE {

  class Comm;
  struct VTUData;
  template <class Real> class FFT;
  template <class ValueType> class Vector;
  template <class ValueType> class Matrix;
  template <class Real> class ElementListBase;
  template <class Real, class Kernel> class BoundaryIntegralOp;

  template <class Real> class LagrangeInterp {
    public:
      static void Interpolate(Vector<Real>& wts, const Vector<Real>& src_nds, const Vector<Real>& trg_nds);

      static void Derivative(Vector<Real>& df, const Vector<Real>& f, const Vector<Real>& nds);

      static void test();
  };

  template <class Real, Integer Nm = 12, Integer Nr = 20, Integer Nt = 16> class ToroidalGreensFn {
      static constexpr Integer COORD_DIM = 3;
      static constexpr Real min_dist = 0.0;
      static constexpr Real max_dist = 0.2;

    public:

      /**
       * Constructor
       */
      ToroidalGreensFn() {}

      /**
       * Precompute tables for modal Green's funcation
       */
      template <class Kernel> void Setup(const Kernel& ker, Real R0);

      /**
       * Build modal Green's function operator for a given target point
       * (x0,x1,x2).
       */
      template <class Kernel> void BuildOperatorModal(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const;

    private:

      /**
       * Basis functions in which to represent the potential.
       */
      template <class ValueType> class BasisFn { // p(x) log(x) + q(x) + 1/x
        public:
          static ValueType Eval(const Vector<ValueType>& coeff, ValueType x);
          static void EvalBasis(Vector<ValueType>& f, ValueType x);
          static const Vector<ValueType>& nds(Integer ORDER);
      };

      /**
       * Precompute tables for modal Green's funcation
       */
      template <class ValueType, class Kernel> void PrecompToroidalGreensFn(const Kernel& ker, ValueType R0);

      /**
       * Compute reference potential using adaptive integration.
       */
      template <class ValueType, class Kernel> static void ComputePotential(Vector<ValueType>& U, const Vector<ValueType>& Xtrg, ValueType R0, const Vector<ValueType>& F_, const Kernel& ker, ValueType tol = 1e-18);

      /**
       * Compute modal Green's function operator using trapezoidal quadrature
       * rule (for distant target points).
       */
      template <Integer Nnds, class Kernel> void BuildOperatorModalDirect(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const;

      Real R0_;
      FFT<Real> fft_Nm_R2C, fft_Nm_C2R;
      Matrix<Real> Mnds2coeff0, Mnds2coeff1;
      Vector<Real> U; // KDIM0*Nmm*KDIM1*Nr*Ntt
      Vector<Real> Ut; // Nr*Ntt*KDIM0*Nmm*KDIM1
  };

  /**
   * Implements the abstract class ElementListBase for slender boundary
   * elements with circular cross-section.
   *
   * @see ElementListBase
   */
  template <class Real> class SlenderElemList : public ElementListBase<Real> {
      static constexpr Integer FARFIELD_UPSAMPLE = 1;
      static constexpr Integer COORD_DIM = 3;

      static constexpr Integer ModalUpsample = 1; // toroidal quadrature order is FourierModes+ModalUpsample

    public:

      /**
       * Constructor
       */
      SlenderElemList() {}

      /**
       * Constructor
       */
      SlenderElemList(const Vector<Long>& cheb_order0, const Vector<Long>& fourier_order0, const Vector<Real>& coord0, const Vector<Real>& radius0, const Vector<Real>& orientation0 = Vector<Real>());

      /**
       * Initialize list of elements
       */
      void Init(const Vector<Long>& cheb_order0, const Vector<Long>& fourier_order0, const Vector<Real>& coord0, const Vector<Real>& radius0, const Vector<Real>& orientation0 = Vector<Real>());

      /**
       * Destructor
       */
      virtual ~SlenderElemList() {}

      /**
       * Return the number of elements in the list.
       */
      Long Size() const;

      /**
       * Returns the position and normals of the surface nodal points for each
       * element.
       *
       * @see ElementListBase::GetNodeCoord()
       */
      void GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const override;

      /**
       * Given an accuracy tolerance, returns the quadrature node positions,
       * the normals at the nodes, the weights and the cut-off distance from
       * the nodes for computing the far-field potential from the surface (at
       * target points beyond the cut-off distance).
       *
       * @see ElementListBase::GetFarFieldNodes()
       */
      void GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const override;

      /**
       * Interpolates the density from surface node points to far-field
       * quadrature node points.
       *
       * @see ElementListBase::GetFarFieldDensity()
       */
      void GetFarFieldDensity(Vector<Real>& Fout, const Vector<Real>& Fin) const override;

      /**
       * Apply the transpose of the GetFarFieldDensity() operator applied to
       * the column-vectors of Min and the result is returned in Mout.
       *
       * @see ElementListBase::FarFieldDensityOperatorTranspose()
       */
      void FarFieldDensityOperatorTranspose(Matrix<Real>& Mout, const Matrix<Real>& Min, const Long elem_idx) const override;

      /**
       * Compute self-interaction operator for each element.
       *
       * @see ElementListBase::SelfInterac()
       */
      template <class Kernel> static void SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);

      /**
       * Compute near-interaction operator for a given element-idx and each each target.
       *
       * @see ElementListBase::NearInterac()
       */
      template <class Kernel> static void NearInterac(Matrix<Real>& M, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);

      /**
       * Returns the Chebyshev node points for a given order.
       */
      static const Vector<Real>& CenterlineNodes(Integer Order);

      /**
       * Write elements to file.
       */
      void Write(const std::string& fname, const Comm& comm = Comm::Self()) const;

      /**
       * Read elements from file.
       */
      void Read(const std::string& fname, const Comm& comm = Comm::Self());

      /**
       * Get geometry data for an element.
       */
      void GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_ds, Vector<Real>* dX_dt, const Vector<Real>& s_param, const Vector<Real>& sin_theta_, const Vector<Real>& cos_theta_, const Long elem_idx) const;

      /**
       * Get the VTU (Visualization Toolkit for Unstructured grids) data for
       * one or all elements.
       */
      void GetVTUData(VTUData& vtu_data, const Vector<Real>& F = Vector<Real>(), const Long elem_idx = -1) const;

      /**
       * Write VTU data to file.
       */
      void WriteVTK(const std::string& fname, const Vector<Real>& F = Vector<Real>(), const Comm& comm = Comm::Self()) const;

      /**
       * Test example for Laplace double-layer kernel.
       */
      template <class Kernel> static void test(const Comm& comm = Comm::Self(), Real tol = 1e-10);

      /**
       * Test example for Green's identity with Laplace kernel.
       */
      static void test_greens_identity(const Comm& comm = Comm::Self(), Real tol = 1e-10);

      template <class ValueType> void Copy(SlenderElemList<ValueType>& elem_lst) const;

    private:

      template <class Kernel> Matrix<Real> SelfInteracHelper_(const Kernel& ker, const Long elem_idx, const Real tol) const; // constant radius
      template <Integer digits, bool trg_dot_prod, class Kernel> Matrix<Real> SelfInteracHelper(const Kernel& ker, const Long elem_idx) const;

      template <Integer digits, bool trg_dot_prod, class Kernel> void NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx) const;

      Vector<Real> radius, coord, e1;
      Vector<Long> cheb_order, fourier_order, elem_dsp;

      Vector<Real> dr, dx, d2x; // derived quantities
  };

}

#include SCTL_INCLUDE(slender_element.txx)

#endif //_SCTL_SLENDER_ELEMENT_HPP_
