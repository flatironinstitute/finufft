#ifndef _SCTL_BOUNDARY_INTEGRAL_HPP_
#define _SCTL_BOUNDARY_INTEGRAL_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)
#include SCTL_INCLUDE(vector.hpp)

#include <map>
#include <set>

namespace SCTL_NAMESPACE {

  template <class ValueType> class Matrix;
  template <class Real, Integer DIM> class ParticleFMM;

  /**
   * Abstract base class for an element-list. In addition to the functions
   * declared in this base class, the derived class must be copy-constructible.
   */
  template <class Real> class ElementListBase {

    public:

      virtual ~ElementListBase() {}

      /**
       * Returns the position and normals of the surface nodal points for each
       * element.
       *
       * @param[out] X the position of the node points in array-of-struct
       * order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       *
       * @param[out] Xn the normal vectors of the node points in
       * array-of-struct order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       *
       * @param[out] element_wise_node_cnt the number of node points
       * belonging to each element.
       */
      virtual void GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const = 0;

      /**
       * Given an accuracy tolerance, returns the quadrature node positions,
       * the normals at the nodes, the weights and the cut-off distance from
       * the nodes for computing the far-field potential from the surface (at
       * target points beyond the cut-off distance).
       *
       * @param[out] X the position of the quadrature node points in
       * array-of-struct order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       *
       * @param[out] Xn the normal vectors at the quadrature node points in
       * array-of-struct order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       *
       * @param[out] wts the weights corresponding to each quadrature node.
       *
       * @param[out] dist_far the cut-off distance from each quadrature node
       * such that quadrature rule will be accurate to the specified tolerance
       * for target points further away than this distance.
       *
       * @param[out] element_wise_node_cnt the number of quadrature node
       * points belonging to each element.
       *
       * @param[in] tol the accuracy tolerance.
       */
      virtual void GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const = 0;

      /**
       * Interpolates the density from surface node points to far-field
       * quadrature node points.
       *
       * @param[out] Fout the interpolated density at far-field quadrature
       * nodes in array-of-struct order.
       *
       * @param[in] Fin the input density at surface node points in
       * array-of-struct order.
       */
      virtual void GetFarFieldDensity(Vector<Real>& Fout, const Vector<Real>& Fin) const = 0;

      /**
       * Apply the transpose of the GetFarFieldDensity() operator for a single
       * element applied to the column-vectors of Min and the result is
       * returned in Mout.
       *
       * @param[out] Mout the output matrix where the column-vectors are the
       * result of the application of the transpose operator.
       *
       * @param[in] Min the input matrix whose column-vectors are
       * multiplied by the transpose operator.
       *
       * @param[in] elem_idx the index of the element.
       */
      virtual void FarFieldDensityOperatorTranspose(Matrix<Real>& Mout, const Matrix<Real>& Min, const Long elem_idx) const = 0;

      /**
       * Compute self-interaction operator for each element.
       *
       * @param[out] M_lst the vector of all self-interaction matrices
       * (in row-major format).
       *
       * @param[in] ker the kernel object.
       *
       * @param[in] tol the accuracy tolerance.
       *
       * @param[in] trg_dot_prod whether to compute dot product of the potential with the target-normal vector.
       *
       * @param[in] self pointer to element-list object.
       */
      template <class Kernel> static void SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self);

      /**
       * Compute near-interaction operator for a given element-idx and each each target.
       *
       * @param[out] M the near-interaction matrix (in row-major format).
       *
       * @param[in] Xt the position of the target points in array-of-structure
       * order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       *
       * @param[in] normal_trg the normal at the target points in array-of-structure
       * order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       *
       * @param[in] ker the kernel object.
       *
       * @param[in] tol the accuracy tolerance.
       *
       * @param[in] elem_idx the index of the source element.
       *
       * @param[in] self pointer to element-list object.
       */
      template <class Kernel> static void NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self);
  };

  /**
   * Implements parallel computation of boundary integrals.
   */
  template <class Real, class Kernel> class BoundaryIntegralOp {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1 = Kernel::TrgDim();
      static constexpr Integer COORD_DIM = 3;

    public:

      BoundaryIntegralOp() = delete;
      BoundaryIntegralOp(const BoundaryIntegralOp&) = delete;
      BoundaryIntegralOp& operator= (const BoundaryIntegralOp&) = delete;

      /**
       * Constructor
       *
       * @param[in] ker the kernel object.
       *
       * @param[in] comm the MPI communicator.
       */
      explicit BoundaryIntegralOp(const Kernel& ker, bool trg_normal_dot_prod = false, const Comm& comm = Comm::Self());

      /**
       * Destructor
       */
      ~BoundaryIntegralOp();

      /**
       * Specify accuracy tolerance.
       */
      void SetAccuracy(Real tol);

      /**
       * Set kernel functions for FMM translation operators
       */
      template <class KerS2M, class KerS2L, class KerS2T, class KerM2M, class KerM2L, class KerM2T, class KerL2L, class KerL2T> void SetFMMKer(const KerS2M& k_s2m, const KerS2L& k_s2l, const KerS2T& k_s2t, const KerM2M& k_m2m, const KerM2L& k_m2l, const KerM2T& k_m2t, const KerL2L& k_l2l, const KerL2T& k_l2t);

      /**
       * Add an element-list.
       */
      template <class ElemLstType> void AddElemList(const ElemLstType& elem_lst, const std::string& name = std::to_string(typeid(ElemLstType).hash_code()));

      /**
       * Get const reference to an element-list.
       */
      template <class ElemLstType> const ElemLstType& GetElemList(const std::string& name = std::to_string(typeid(ElemLstType).hash_code())) const;

      /**
       * Delete an element-list.
       */
      void DeleteElemList(const std::string& name);

      /**
       * Delete an element-list.
       */
      template <class ElemLstType> void DeleteElemList();

      /**
       * Set target point coordinates.
       *
       * @param[in] Xtrg the coordinates of target points in array-of-struct
       * order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
       */
      void SetTargetCoord(const Vector<Real>& Xtrg);

      /**
       * Set target point normals.
       *
       * @param[in] Xn_trg the coordinates of target points in array-of-struct
       * order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
       */
      void SetTargetNormal(const Vector<Real>& Xn_trg);

      /**
       * Get dimension of the boundary integral operator. Dim(0) is the input
       * dimension and Dim(1) is the output dimension.
       */
      Long Dim(Integer k) const;

      /**
       * Setup the boundary integral operator.
       */
      void Setup() const;

      /**
       * Clear setup data.
       */
      void ClearSetup() const;

      /**
       * Evaluate the boundary integral operator.
       */
      void ComputePotential(Vector<Real>& U, const Vector<Real>& F) const;

    private:

      void SetupBasic() const;
      void SetupFar() const;
      void SetupSelf() const;
      void SetupNear() const;

      void ComputeFarField(Vector<Real>& U, const Vector<Real>& F) const;
      void ComputeNearInterac(Vector<Real>& U, const Vector<Real>& F) const;

      struct ElemLstData {
        void (*SelfInterac)(Vector<Matrix<Real>>&, const Kernel&, Real, bool, const ElementListBase<Real>*);
        void (*NearInterac)(Matrix<Real>&, const Vector<Real>&, const Vector<Real>&, const Kernel&, Real, const Long, const ElementListBase<Real>*);
      };
      std::map<std::string,ElementListBase<Real>*> elem_lst_map;
      std::map<std::string,ElemLstData> elem_data_map;
      Vector<Real> Xt; // User specified position of target points
      Vector<Real> Xnt; // User specified normal at target points
      Real tol_;
      Kernel ker_;
      bool trg_normal_dot_prod_;
      Comm comm_;

      mutable bool setup_flag;
      mutable Vector<std::string> elem_lst_name; // name of each element-list (size=Nlst)
      mutable Vector<Long> elem_lst_cnt, elem_lst_dsp; // cnt and dsp of elements for each elem_lst (size=Nlst)
      mutable Vector<Long> elem_nds_cnt, elem_nds_dsp; // cnt and dsp of nodes for each element (size=Nelem)
      mutable Vector<Real> Xsurf; // Position of surface node points (target points for on-surface evaluation)
      mutable Vector<Real> Xn_surf; // Normal at surface node points (normal vector for on-surface evaluation)
      mutable Vector<Real> Xtrg; // Position of target points
      mutable Vector<Real> Xn_trg; // Normal vector at target points

      mutable bool setup_far_flag;
      mutable ParticleFMM<Real,COORD_DIM> fmm;
      mutable Vector<Long> elem_nds_cnt_far, elem_nds_dsp_far; // cnt and dsp of far-nodes for each element (size=Nelem)
      mutable Vector<Real> X_far, Xn_far, wts_far; // position, normal and weights for far-field quadrature
      mutable Vector<Real> dist_far; // minimum distance of target points for far-field evaluation
      mutable Vector<Real> F_far; // pre-allocated memory for density in far-field evaluation

      mutable bool setup_near_flag;
      mutable Vector<Real> Xtrg_near; // position of near-interaction target points sorted by element (size=Nnear*COORD_DIM)
      mutable Vector<Real> Xn_trg_near; // normal at near-interaction target points sorted by element (size=Nnear*COORD_DIM)
      mutable Vector<Long> near_scatter_index; // prmutation vector that takes near-interactions sorted by elem-idx to sorted by trg-idx (size=Nnear)
      mutable Vector<Long> near_trg_cnt, near_trg_dsp; // cnt and dsp of near-interactions for each target (size=Ntrg)
      mutable Vector<Long> near_elem_cnt, near_elem_dsp; // cnt and dsp of near-interaction for each element (size=Nelem)
      mutable Vector<Long> K_near_cnt, K_near_dsp; // cnt and dsp of element wise near-interaction matrix (size=Nelem)
      mutable Vector<Real> K_near;

      mutable bool setup_self_flag;
      mutable Vector<Matrix<Real>> K_self;
  };

}

#include SCTL_INCLUDE(boundary_integral.txx)

#endif //_SCTL_BOUNDARY_INTEGRAL_HPP_
