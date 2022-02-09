#ifndef _SCTL_CHEB_UTILS_HPP_
#define _SCTL_CHEB_UTILS_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(math_utils.hpp)
#include SCTL_INCLUDE(legendre_rule.hpp)

#include <type_traits>
#include <functional>
#include <algorithm>

namespace SCTL_NAMESPACE {

template <class ValueType, class Derived> class BasisInterface {

 public:
  template <Integer DIM> static void Nodes(Integer order, Vector<ValueType>& nodes) {
    if (DIM == 1) {
      Derived::Nodes1D(order, nodes);
      return;
    }

    Vector<ValueType> nodes1d;
    Derived::Nodes1D(order, nodes1d);

    Integer order_DIM = pow<Integer>(order, DIM);
    if (nodes.Dim() != order_DIM * DIM) nodes.ReInit(order_DIM * DIM);

    StaticArray<Integer, DIM> idx;
    for (Integer i = 0; i < DIM; i++) idx[i] = 0;
    Integer itr = 0;
    for (Integer j = 0; j < order_DIM; j++) {
      for (Integer i = 0; i < DIM; i++) {
        if (idx[i] == order) idx[i] = 0;
        nodes[itr + i] = nodes1d[idx[i]];
      }
      itr += DIM;
      idx[0]++;
      for (Integer i = 1; i < DIM; i++) {
        if (idx[i - 1] == order) idx[i]++;
      }
    }
  }

  /**
   * \brief Computes approximation from function values at node points.
   * \param[in] fn_v Function values at node points (dof x order^DIM).
   * \param[out] coeff Coefficient values (dof x Ncoeff).
   */
  template <Integer DIM> static void Approx(Integer order, const Vector<ValueType>& fn_v, Vector<ValueType>& coeff) {
    Matrix<ValueType> Mp;
    {  // Precompute
      static Vector<Matrix<ValueType>> precomp(1000);
      SCTL_ASSERT(order < precomp.Dim());
      if (precomp[order].Dim(0) * precomp[order].Dim(1) == 0) {
        #pragma omp critical(SCTL_BASIS_APPROX)
        if (precomp[order].Dim(0) * precomp[order].Dim(1) == 0) {
          Vector<ValueType> x, p;
          Derived::Nodes1D(order, x);
          Derived::EvalBasis1D(order, x, p);
          Matrix<ValueType> Mp1(order, order, p.begin(), false);
          Mp1.pinv().Swap(precomp[order]);
        }
      }
      Mp.ReInit(precomp[order].Dim(0), precomp[order].Dim(1), precomp[order].begin(), false);
    }

    Integer order_DIM = pow<Integer>(order, DIM);
    Integer order_DIM_ = pow<Integer>(order, DIM - 1);
    Long dof = fn_v.Dim() / order_DIM;
    SCTL_ASSERT(fn_v.Dim() == dof * order_DIM);

    // Create work buffers
    Long buff_size = dof * order_DIM;
    Vector<ValueType> buff(2 * buff_size);
    Iterator<ValueType> buff1 = buff.begin() + buff_size * 0;
    Iterator<ValueType> buff2 = buff.begin() + buff_size * 1;

    Vector<ValueType> fn(order_DIM * dof, (Iterator<ValueType>)fn_v.begin(), false);
    for (Integer k = 0; k < DIM; k++) {  // Apply Mp along k-dimension
      Matrix<ValueType> Mi(dof * order_DIM_, order, fn.begin(), false);
      Matrix<ValueType> Mo(dof * order_DIM_, order, buff2, false);
      Matrix<ValueType>::GEMM(Mo, Mi, Mp);

      Matrix<ValueType> Mo_t(order, dof * order_DIM_, buff1, false);
      for (Long i = 0; i < Mo.Dim(0); i++) {
        for (Long j = 0; j < Mo.Dim(1); j++) {
          Mo_t[j][i] = Mo[i][j];
        }
      }
      fn.ReInit(order_DIM * dof, buff1, false);
    }

    {  // Rearrange and write to coeff
      Vector<ValueType> tensor(order_DIM * dof, buff1, false);
      tensor2coeff<DIM>(order, tensor, coeff);
    }
  }

  template <Integer DIM> static void Approx_(Integer order, const Vector<ValueType>& fn_v, Vector<ValueType>& coeff, ValueType scale) {
    Matrix<ValueType> Mp;
    {  // Precompute
      static Vector<Matrix<ValueType>> precomp(1000);
      SCTL_ASSERT(order < precomp.Dim());
      if (precomp[order].Dim(0) * precomp[order].Dim(1) == 0) {
        #pragma omp critical(SCTL_BASIS_APPROX)
        if (precomp[order].Dim(0) * precomp[order].Dim(1) == 0) {
          Vector<ValueType> x, p;
          Derived::Nodes1D(order, x);
          for (Integer i = 0; i < order; i++) x[i] = (x[i] - 0.5) * scale + 0.5;
          Derived::EvalBasis1D(order, x, p);
          Matrix<ValueType> Mp1(order, order, p.begin(), false);
          Mp1.pinv().Swap(precomp[order]);
        }
      }
      Mp.ReInit(precomp[order].Dim(0), precomp[order].Dim(1), precomp[order].begin(), false);
    }

    Integer order_DIM = pow<Integer>(order, DIM);
    Integer order_DIM_ = pow<Integer>(order, DIM - 1);
    Long dof = fn_v.Dim() / order_DIM;
    SCTL_ASSERT(fn_v.Dim() == dof * order_DIM);

    // Create work buffers
    Long buff_size = dof * order_DIM;
    Vector<ValueType> buff(2 * buff_size);
    Iterator<ValueType> buff1 = buff.begin() + buff_size * 0;
    Iterator<ValueType> buff2 = buff.begin() + buff_size * 1;

    Vector<ValueType> fn(order_DIM * dof, (Iterator<ValueType>)fn_v.begin(), false);
    for (Integer k = 0; k < DIM; k++) {  // Apply Mp along k-dimension
      Matrix<ValueType> Mi(dof * order_DIM_, order, fn.begin(), false);
      Matrix<ValueType> Mo(dof * order_DIM_, order, buff2, false);
      Matrix<ValueType>::GEMM(Mo, Mi, Mp);

      Matrix<ValueType> Mo_t(order, dof * order_DIM_, buff1, false);
      for (Long i = 0; i < Mo.Dim(0); i++) {
        for (Long j = 0; j < Mo.Dim(1); j++) {
          Mo_t[j][i] = Mo[i][j];
        }
      }
      fn.ReInit(order_DIM * dof, buff1, false);
    }

    {  // Rearrange and write to coeff
      Vector<ValueType> tensor(order_DIM * dof, buff1, false);
      tensor2coeff<DIM>(order, tensor, coeff);
    }
  }

  /**
   * \brief Evaluates values from input coefficients at points on a regular
   * grid defined by in_x, in_y, in_z the values in the input vector.
   * \param[in] coeff Coefficient values (dof x Ncoeff).
   * \param[out] out Values at node points (in_x[DIM-1].Dim() x ... x in_x[0].Dim() x dof).
   */
  template <Integer DIM> static void Eval(Integer order, const Vector<ValueType>& coeff, ConstIterator<Vector<ValueType>> in_x, Vector<ValueType>& out) {
    Integer Ncoeff = 1;
    for (Integer i = 0; i < DIM; i++) {
      Ncoeff = (Ncoeff * (order + i)) / (i + 1);
    }
    Long dof = coeff.Dim() / Ncoeff;
    SCTL_ASSERT(coeff.Dim() == Ncoeff * dof);

    // Precomputation
    Long buff_size = dof;
    StaticArray<Matrix<ValueType>, DIM> Mp;
    for (Integer i = 0; i < DIM; i++) {
      Integer n = in_x[i].Dim();
      if (!n) return;
      Mp[i].ReInit(order, n);
      Vector<ValueType> p(order * n, Mp[i].begin(), false);
      Derived::EvalBasis1D(order, in_x[i], p);
      buff_size *= std::max(order, n);
    }

    // Create work buffers
    Vector<ValueType> buff(2 * buff_size);
    Iterator<ValueType> buff1 = buff.begin() + buff_size * 0;
    Iterator<ValueType> buff2 = buff.begin() + buff_size * 1;

    {  // Rearrange coefficients into a tensor.
      Vector<ValueType> tensor(dof * pow<Integer>(order, DIM), buff1, false);
      coeff2tensor<DIM>(order, coeff, tensor);
    }

    {  // ReInit out
      Long len = dof;
      for (Integer i = 0; i < DIM; i++) len *= in_x[i].Dim();
      if (out.Dim() != len) out.ReInit(len);
    }

    for (Integer k = 0; k < DIM; k++) {  // Apply Mp along k-dimension
      Integer order_DIM = pow<Integer>(order, DIM - k - 1);
      for (Integer i = 0; i < k; i++) order_DIM *= in_x[i].Dim();

      Matrix<ValueType> Mi(dof * order_DIM, order, buff1, false);
      Matrix<ValueType> Mo(dof * order_DIM, in_x[k].Dim(), buff2, false);
      Matrix<ValueType>::GEMM(Mo, Mi, Mp[k]);

      Matrix<ValueType> Mo_t(in_x[k].Dim(), dof * order_DIM, buff1, false);
      if (k == DIM - 1) Mo_t.ReInit(in_x[k].Dim(), dof * order_DIM, out.begin(), false);
      for (Long i = 0; i < Mo.Dim(0); i++) {
        for (Long j = 0; j < Mo.Dim(1); j++) {
          Mo_t[j][i] = Mo[i][j];
        }
      }
    }
  }

  /**
   * \brief Returns the sum of the absolute value of coefficients of the
   * highest order terms as an estimate of truncation error.
   * \param[in] coeff Coefficient values (dof x Ncoeff).
   */
  template <Integer DIM> static ValueType TruncErr(Integer order, const Vector<ValueType>& coeff) {
    Integer Ncoeff = 1;
    {  // Set Ncoeff
      for (Integer i = 0; i < DIM; i++) Ncoeff = (Ncoeff * (order + i)) / (i + 1);
    }
    Long dof = coeff.Dim() / Ncoeff;
    SCTL_ASSERT(coeff.Dim() == Ncoeff * dof);

    ValueType err = 0;
    for (Long l = 0; l < dof; l++) {  // TODO: optimize this
      Long offset0 = l * Ncoeff;

      Integer indx0 = 0;
      Integer indx1 = 0;
      StaticArray<Integer, DIM + 1> i0;
      for (Integer i = 0; i <= DIM; i++) i0[i] = 0;

      Integer sum = 0;
      while (1) {
        if (sum < order) {
          if (sum == order - 1) err += fabs<ValueType>(coeff[offset0 + indx0]);
          indx0++;
        }
        indx1++;
        sum++;

        i0[0]++;
        for (Integer j = 0; j < DIM && i0[j] == order; j++) {
          i0[j] = 0;
          i0[j + 1]++;
          sum = sum + 1 - order;
        }
        if (i0[DIM]) break;
      }
    }

    return err;
  }

  /**
   * \brief Compute gradient.
   * \param[in] coeff_in Input coefficients (dof x Ncoeff)
   * \param[out] coeff_out Output coefficients (dof x DIM x Ncoeff)
   */
  template <Integer DIM> static void Grad(Integer order, const Vector<ValueType>& coeff_in, Vector<ValueType>* coeff_out) {
    Integer Ncoeff = 1;
    for (Integer i = 0; i < DIM; i++) {
      Ncoeff = (Ncoeff * (order + i)) / (i + 1);
    }
    Long dof = coeff_in.Dim() / Ncoeff;
    SCTL_ASSERT(coeff_in.Dim() == Ncoeff * dof);

    Matrix<ValueType> Mdiff;
    {  // Precompute
      static Vector<Matrix<ValueType>> precomp(1000);
      SCTL_ASSERT(order < precomp.Dim());
      if (precomp[order].Dim(0) * precomp[order].Dim(1) == 0) {
        #pragma omp critical(SCTL_BASIS_GRAD)
        if (precomp[order].Dim(0) * precomp[order].Dim(1) == 0) {
          Matrix<ValueType> M;
          diff_1d(order, &M);
          M.Swap(precomp[order]);
        }
      }
      Mdiff.ReInit(precomp[order].Dim(0), precomp[order].Dim(1), precomp[order].begin(), false);
    }

    // Create work buffers
    Long buff_size = dof * pow<Integer>(order, DIM);
    Vector<ValueType> buff((3 + DIM) * buff_size);
    Vector<ValueType> buff0(buff_size * 1, buff.begin() + buff_size * 0, false);
    Vector<ValueType> buff1(buff_size * 1, buff.begin() + buff_size * 1, false);
    Vector<ValueType> buff2(buff_size * 1, buff.begin() + buff_size * 2, false);
    Vector<ValueType> buff3(buff_size * DIM, buff.begin() + buff_size * 3, false);

    {  // buff0 <-- coeff2tensor(coeff_in);
      coeff2tensor<DIM>(order, coeff_in, buff0);
    }

    for (Integer k = 0; k < DIM; k++) {  // buff2 <-- Grad(buff0)
      Long N0 = pow<Integer>(order, k);
      Long N1 = order;
      Long N2 = pow<Integer>(order, DIM - k - 1);

      for (Long i3 = 0; i3 < dof; i3++) {  // buff1 <-- transpose(buff0)
        for (Long i2 = 0; i2 < N2; i2++) {
          for (Long i1 = 0; i1 < N1; i1++) {
            for (Long i0 = 0; i0 < N0; i0++) {
              buff1[((i3 * N2 + i2) * N0 + i0) * N1 + i1] = buff0[((i3 * N2 + i2) * N1 + i1) * N0 + i0];
            }
          }
        }
      }

      {  // buff2 <-- buff1 * Mdiff
        Matrix<ValueType> Mi(dof * N0 * N2, N1, buff1.begin(), false);
        Matrix<ValueType> Mo(dof * N0 * N2, N1, buff2.begin(), false);
        Matrix<ValueType>::GEMM(Mo, Mi, Mdiff);
      }

      for (Long i3 = 0; i3 < dof; i3++) {  // buff3 <-- transpose(buff2)
        for (Long i2 = 0; i2 < N2; i2++) {
          for (Long i1 = 0; i1 < N1; i1++) {
            for (Long i0 = 0; i0 < N0; i0++) {
              buff3[(((i2 * N1 + i1) * N0 + i0) * dof + i3) * DIM + k] = buff2[((i3 * N2 + i2) * N0 + i0) * N1 + i1];
            }
          }
        }
      }
    }

    {  // coeff_out <-- tensor2coeff(buff2);
      tensor2coeff<DIM>(order, buff3, *coeff_out);
    }
  }

  template <Integer DIM, Integer SUBDIM, class Kernel> static void Integ(Matrix<ValueType>& Mcoeff, Integer order, ConstIterator<ValueType> trg_, ValueType side, Integer src_face, const Kernel& ker, ValueType tol = -1, Integer Nq = 0) {
    if (!Nq) Nq = order;
    Integ_<DIM, SUBDIM>(Mcoeff, order, trg_, side, src_face, ker, Nq);
    if (tol < 0) tol = 1e-10; //machine_eps() * 256;
    ValueType err = tol + 1;
    Matrix<ValueType> Mtmp;
    while (err > tol) {
      err = 0;
      ValueType max_val = pow<SUBDIM>(side);
      Nq = std::max((Integer)(Nq * 1.26), Nq + 1);
      Integ_<DIM, SUBDIM>(Mtmp, order, trg_, side, src_face, ker, Nq);
      for (Integer i = 0; i < Mtmp.Dim(0) * Mtmp.Dim(1); i++) {
        err = std::max(err, fabs<ValueType>(Mtmp[0][i] - Mcoeff[0][i]));
        max_val = std::max(max_val, fabs<ValueType>(Mtmp[0][i]));
      }
      err /= max_val;
      Mcoeff = Mtmp;
      if (Nq>200) {
        SCTL_WARN("Failed to converge, error = "<<err);
        break;
      }
    }
    Mcoeff = Mcoeff.Transpose();
  }

  template <Integer DIM> static void tensor2coeff(Integer order, const Vector<ValueType>& tensor, Vector<ValueType>& coeff) {
    Integer Ncoeff = 1, Ntensor = pow<Integer>(order, DIM);
    for (Integer i = 0; i < DIM; i++) Ncoeff = (Ncoeff * (order + i)) / (i + 1);
    Long dof = tensor.Dim() / Ntensor;
    SCTL_ASSERT(tensor.Dim() == Ntensor * dof);
    if (coeff.Dim() != Ncoeff * dof) coeff.ReInit(Ncoeff * dof);

    for (Long l = 0; l < dof; l++) {  // TODO: optimize this
      Long offset0 = l * Ncoeff;

      Integer indx0 = 0;
      Integer indx1 = 0;
      StaticArray<Integer, DIM + 1> i0;
      for (Integer i = 0; i <= DIM; i++) i0[i] = 0;

      Integer sum = 0;
      while (1) {
        if (sum < order) {
          coeff[offset0 + indx0] = tensor[l + indx1 * dof];
          indx0++;
        }
        indx1++;
        sum++;

        i0[0]++;
        for (Integer j = 0; j < DIM && i0[j] == order; j++) {
          i0[j] = 0;
          i0[j + 1]++;
          sum = sum + 1 - order;
        }
        if (i0[DIM]) break;
      }
    }
  }

  template <Integer DIM> static void coeff2tensor(Integer order, const Vector<ValueType>& coeff, Vector<ValueType>& tensor) {
    Integer Ncoeff = 1, Ntensor = pow<Integer>(order, DIM);
    for (Integer i = 0; i < DIM; i++) Ncoeff = (Ncoeff * (order + i)) / (i + 1);
    Long dof = coeff.Dim() / Ncoeff;
    SCTL_ASSERT(coeff.Dim() == Ncoeff * dof);
    if (tensor.Dim() != Ntensor * dof) tensor.ReInit(Ntensor * dof);

    for (Long l = 0; l < dof; l++) {  // TODO: optimize this
      Long offset0 = l * Ncoeff;
      Long offset1 = l * Ntensor;

      Integer indx0 = 0;
      Integer indx1 = 0;
      StaticArray<Integer, DIM + 1> i0;
      for (Integer i = 0; i <= DIM; i++) i0[i] = 0;

      Integer sum = 0;
      while (1) {
        if (sum < order) {
          tensor[offset1 + indx1] = coeff[offset0 + indx0];
          indx0++;
        } else {
          tensor[offset1 + indx1] = 0;
        }
        indx1++;
        sum++;

        i0[0]++;
        for (Integer j = 0; j < DIM && i0[j] == order; j++) {
          i0[j] = 0;
          i0[j + 1]++;
          sum = sum + 1 - order;
        }
        if (i0[DIM]) break;
      }
    }
  }

  template <Integer DIM> static void Truncate(Vector<ValueType> &coeff0, Integer order0, Integer order1) {
    SCTL_ASSERT(order1 <= order0);
    Integer Ncoeff0 = 1, Ncoeff1 = 1;
    for (Integer i = 0; i < DIM; i++) Ncoeff0 = (Ncoeff0 * (order0 + i)) / (i + 1);
    for (Integer i = 0; i < DIM; i++) Ncoeff1 = (Ncoeff1 * (order1 + i)) / (i + 1);

    Long dof = coeff0.Dim() / Ncoeff0;
    SCTL_ASSERT(coeff0.Dim() == Ncoeff0 * dof);
    Vector<ValueType> coeff1(dof * Ncoeff1);
    coeff1.SetZero();

    for (Long l = 0; l < dof; l++) {  // TODO: optimize this
      Long offset0 = l * Ncoeff0;
      Long offset1 = l * Ncoeff1;

      Integer indx0 = 0;
      Integer indx1 = 0;
      StaticArray<Integer, DIM + 1> i0;
      for (Integer i = 0; i <= DIM; i++) i0[i] = 0;

      Integer sum = 0;
      while (1) {
        if (sum < order1) coeff1[offset1 + indx1] = coeff0[offset0 + indx0];
        if (sum < order0) indx0++;
        if (sum < order1) indx1++;
        sum++;

        i0[0]++;
        for (Integer j = 0; j < DIM && i0[j] == order0; j++) {
          i0[j] = 0;
          i0[j + 1]++;
          sum = sum + 1 - order0;
        }
        if (i0[DIM]) break;
      }
    }
    coeff0 = coeff1;
  }

  template <Integer DIM> static void Reflect(Vector<ValueType> &coeff, Integer order, Integer dir) {
    SCTL_ASSERT(dir < DIM);
    Integer Ncoeff = 1;
    for (Integer i = 0; i < DIM; i++) Ncoeff = (Ncoeff * (order + i)) / (i + 1);

    Long dof = coeff.Dim() / Ncoeff;
    SCTL_ASSERT(coeff.Dim() == Ncoeff * dof);

    for (Long l = 0; l < dof; l++) {  // TODO: optimize this
      Long offset = l * Ncoeff;

      Integer indx = 0;
      StaticArray<Integer, DIM + 1> i0;
      for (Integer i = 0; i <= DIM; i++) i0[i] = 0;

      Integer sum = 0;
      while (1) {
        if (sum < order) coeff[offset + indx] = coeff[offset + indx] * (i0[dir] % 2 ? -1 : 1) * (1);
        if (sum < order) indx++;
        sum++;

        i0[0]++;
        for (Integer j = 0; j < DIM && i0[j] == order; j++) {
          i0[j] = 0;
          i0[j + 1]++;
          sum = sum + 1 - order;
        }
        if (i0[DIM]) break;
      }
    }
  }

  template <Integer DIM, Integer CONTINUITY> static void MakeContinuous(Vector<ValueType> &coeff0, Vector<ValueType> &coeff1, Integer order, Integer dir0, Integer dir1) {
    if (dir0>=2*DIM || dir1>=2*DIM) return;
    Integer Ncoeff = 1;
    for (Integer i = 0; i < DIM; i++) Ncoeff = (Ncoeff * (order + i)) / (i + 1);

    Long dof = coeff0.Dim() / Ncoeff;
    SCTL_ASSERT(coeff0.Dim() == Ncoeff * dof);
    SCTL_ASSERT(coeff1.Dim() == Ncoeff * dof);

    static Matrix<Matrix<ValueType>> M(2*DIM, 2*DIM);
    if (M[dir0][dir1].Dim(0) != 2 * Ncoeff) {
      Integer Ngrid = pow<Integer>(order, DIM - 1);
      Vector<ValueType> nodes;
      Nodes<1>(order, nodes);

      Matrix<ValueType> M_diff(2*Ncoeff, Ngrid);
      { // Set M_diff
        M_diff.SetZero();
        StaticArray<Vector<ValueType>, DIM> nodes_;
        for (Integer i = 0; i < DIM; i++) { // Set nodes_
          nodes_[i].ReInit(nodes.Dim(), nodes.begin(), false);
        }
        Vector<ValueType> nodes0, nodes1;
        nodes0.PushBack(0);
        nodes1.PushBack(1);

        Vector<ValueType> value;
        Vector<ValueType> coeff(Ncoeff);
        coeff.SetZero();
        for (Integer i = 0; i < Ncoeff; i++) {
          coeff[i]=0.5;
          value.ReInit(Ngrid, M_diff[i + Ncoeff * 0], false);
          nodes_[dir0/2].ReInit(1, (dir0 & 1 ? nodes1.begin() : nodes0.begin()), false);
          Eval<DIM>(order, coeff, nodes_, value);
          nodes_[dir0/2].ReInit(nodes.Dim(), nodes.begin(), false);

          coeff[i]=-0.5;
          value.ReInit(Ngrid, M_diff[i + Ncoeff * 1], false);
          nodes_[dir1/2].ReInit(1, (dir1 & 1 ? nodes1.begin() : nodes0.begin()), false);
          Eval<DIM>(order, coeff, nodes_, value);
          nodes_[dir1/2].ReInit(nodes.Dim(), nodes.begin(), false);

          coeff[i]=0;
        }
      }

      Matrix<ValueType> M_grad(2 * Ncoeff, 2 * Ncoeff);
      { // Set M_grad
        M_grad.SetZero();
        Vector<ValueType> coeff(Ncoeff * Ncoeff), coeff_grad;
        coeff.SetZero();
        for(Integer i = 0; i < Ncoeff; i++) coeff[i * Ncoeff + i] = 1;
        Grad<DIM>(order, coeff, &coeff_grad);
        for (Integer i = 0; i < Ncoeff; i++){
          for (Integer j = 0; j < Ncoeff; j++){
            M_grad[i + Ncoeff * 0][j + Ncoeff * 0] = coeff_grad[j + (i * DIM + dir0/2) * Ncoeff];
            M_grad[i + Ncoeff * 1][j + Ncoeff * 1] = coeff_grad[j + (i * DIM + dir1/2) * Ncoeff];
          }
        }
      }

      auto fn_perturb = [&](std::function<ValueType(ValueType)> fn, bool even) { // Set M0
        Matrix<ValueType> M0(Ngrid, 2 * Ncoeff);
        M0.SetZero();
        { // dir0
          Integer N0=pow<Integer>(order, dir0/2);
          Integer N1=pow<Integer>(order, 1);
          Integer N2=pow<Integer>(order, DIM - dir0/2 - 1);
          SCTL_ASSERT(N0 * N2 == Ngrid);
          Vector<ValueType> val(Ngrid * order), coeff;
          val.SetZero();
          for (Integer i0=0;i0<N0;i0++){
            for (Integer i2=0;i2<N2;i2++){
              for (Integer i1=0;i1<N1;i1++){
                val[(i2*N1+i1)*N0+i0] = (dir0 & 1 ? fn(nodes[i1]) : fn(1.0 - nodes[i1])) * (even ? 1.0 : -1.0);
              }
              coeff.ReInit(Ncoeff, M0[i2 * N0 + i0] + Ncoeff * 0, false);
              Approx<DIM>(order, val, coeff);
              for (Integer i1=0;i1<N1;i1++){
                val[(i2*N1+i1)*N0+i0] = 0;
              }
            }
          }
        }
        { // dir1
          Integer N0=pow<Integer>(order, dir1/2);
          Integer N1=pow<Integer>(order, 1);
          Integer N2=pow<Integer>(order, DIM - dir1/2 - 1);
          SCTL_ASSERT(N0 * N2 == Ngrid);
          Vector<ValueType> val(Ngrid * order), coeff;
          val.SetZero();
          for (Integer i0=0;i0<N0;i0++){
            for (Integer i2=0;i2<N2;i2++){
              for (Integer i1=0;i1<N1;i1++){
                val[(i2*N1+i1)*N0+i0] = (dir1 & 1 ? fn(nodes[i1]) : fn(1.0 - nodes[i1]));
              }
              coeff.ReInit(Ncoeff, M0[i2 * N0 + i0] + Ncoeff * 1, false);
              Approx<DIM>(order, val, coeff);
              for (Integer i1=0;i1<N1;i1++){
                val[(i2*N1+i1)*N0+i0] = 0;
              }
            }
          }
        }
        return M0;
      };

      if (CONTINUITY == 0) {
        auto fn0 = [](ValueType x) {return x;};
        Matrix<ValueType> M0 = fn_perturb(fn0, 0);
        M[dir0][dir1] = M_diff * M0;
      } else if (CONTINUITY == 1) {
        auto fn0 = [](ValueType x) {return (-2*x + 3) * x * x;};
        auto fn1 = [](ValueType x) {return (1 - x) * x * x;};
        Matrix<ValueType> M0 = fn_perturb(fn0, 0);
        Matrix<ValueType> M1 = fn_perturb(fn1, 1);
        M[dir0][dir1] = M_diff * M0 + M_grad * M_diff * M1;
      } else if (CONTINUITY == 2) {
        auto fn0 = [](ValueType x) {return x*x*x*(6*x*x-15*x+10);};
        auto fn1 = [](ValueType x) {return x*x*x*(-3*x*x+7*x-4);};
        auto fn2 = [](ValueType x) {return x*x*x*(0.5*x*x-1*x+0.5);};
        Matrix<ValueType> M0 = fn_perturb(fn0, 0);
        Matrix<ValueType> M1 = fn_perturb(fn1, 1);
        Matrix<ValueType> M2 = fn_perturb(fn2, 0);
        M[dir0][dir1] = M_diff * M0 + M_grad * M_diff * M1 + M_grad * M_grad * M_diff * M2;
      }

      for (Integer i=0;i<2*Ncoeff;i++){
        M[dir0][dir1][i][i]+=1.0;
      }

      if(0){ //// Alternate approach // DOESN'T WORK
      //Matrix<ValueType> Mgrid2coeff;
      //{ // Set Mgrid2coeff
      //  Integer Ngrid = pow<Integer>(order, DIM);
      //  Matrix<ValueType> M(Ngrid, Ncoeff);
      //  Vector<ValueType> val(Ngrid);
      //  val.SetZero();
      //  for (Integer i=0;i<Ngrid;i++) {
      //    val[i]=1.0;
      //    Vector<ValueType> coeff(Ncoeff, M[i], false);
      //    Approx<DIM>(order, val, coeff);
      //    val[i]=0.0;
      //  }

      //  Mgrid2coeff.ReInit(2*Ngrid, 2*Ncoeff);
      //  Mgrid2coeff.SetZero();
      //  for(Integer i=0;i<Ngrid;i++){
      //    for(Integer j=0;j<Ncoeff;j++){
      //      Mgrid2coeff[i+Ngrid*0][j+Ncoeff*0]=M[i][j];
      //      Mgrid2coeff[i+Ngrid*1][j+Ncoeff*1]=M[i][j];
      //    }
      //  }
      //}

      //Matrix<ValueType> Mcoeff2grid;
      //{ // Set Mgrid2coeff
      //  StaticArray<Vector<ValueType>, DIM> nodes_;
      //  for (Integer i = 0; i < DIM; i++) { // Set nodes_
      //    nodes_[i].ReInit(nodes.Dim(), nodes.begin(), false);
      //  }

      //  Integer Ngrid = pow<Integer>(order, DIM);
      //  Matrix<ValueType> M(Ncoeff, Ngrid);
      //  Vector<ValueType> coeff(Ncoeff);
      //  coeff.SetZero();
      //  for (Integer i=0;i<Ncoeff;i++) {
      //    coeff[i]=1.0;
      //    Vector<ValueType> val(Ngrid, M[i], false);
      //    Eval<DIM>(order, coeff, nodes_, val);
      //    coeff[i]=0.0;
      //  }

      //  Mcoeff2grid.ReInit(2*Ncoeff, 2*Ngrid);
      //  Mcoeff2grid.SetZero();
      //  for(Integer i=0;i<Ncoeff;i++){
      //    for(Integer j=0;j<Ngrid;j++){
      //      Mcoeff2grid[i+Ncoeff*0][j+Ngrid*0]=M[i][j];
      //      Mcoeff2grid[i+Ncoeff*1][j+Ngrid*1]=M[i][j];
      //    }
      //  }
      //}

      //if(0){
      //  Integer Ngrid0 = Ngrid*order;
      //  Matrix<ValueType> MM(2*Ngrid0 + 2*Ngrid, 2*Ngrid0);
      //  MM.SetZero();
      //  for (Integer i=0;i<2*Ngrid0;i++) MM[i][i]=1;
      //  Matrix<ValueType> M0_(Ngrid, 2 * Ngrid0, MM[2 * Ngrid0 + Ngrid * 0], false); M0_ = (Mgrid2coeff * M_diff).Transpose();
      //  Matrix<ValueType> M1_(Ngrid, 2 * Ngrid0, MM[2 * Ngrid0 + Ngrid * 1], false); M1_ = (Mgrid2coeff * M_grad * M_diff).Transpose();
      //  for (Long i=0;i<2*Ngrid*2*Ngrid0;i++) MM[0][2*Ngrid0*2*Ngrid0 +i] *= 10000;
      //  MM = MM.Transpose().pinv();
      //  M[dir].ReInit(2 * Ngrid0, 2 * Ngrid0, MM.begin());
      //  M[dir] = Mcoeff2grid * M[dir] * Mgrid2coeff;
      //} else {
      //  SCTL_ASSERT(DIM==2);
      //  Vector<ValueType> coeff_weight;
      //  for (Integer i=0;i<order;i++) {
      //    for (Integer j=0;j<order;j++) {
      //      if(i+j<order) coeff_weight.PushBack(pow<ValueType>(1.5, i+j)*1e-4);
      //    }
      //  }
      //  SCTL_ASSERT(coeff_weight.Dim()==Ncoeff);

      //  auto M0_ = M_diff.Transpose();
      //  auto M1_ = (M_grad * M_diff).Transpose();

      //  Matrix<ValueType> MM(2*Ncoeff + 6*Ngrid, 2*Ncoeff);
      //  MM.SetZero();
      //  for (Integer i=0;i<Ncoeff;i++) {
      //    MM[i+Ncoeff*0][i+Ncoeff*0]=coeff_weight[i];
      //    MM[i+Ncoeff*1][i+Ncoeff*1]=coeff_weight[i];
      //  }
      //  for (Integer i=0;i<Ngrid;i++) {
      //    for (Integer j=0;j<Ncoeff;j++) {
      //      MM[2 * Ncoeff + 0 * Ngrid +i][0 * Ncoeff + j] = M0_[0 * Ngrid + i][0 * Ncoeff + j];
      //      MM[2 * Ncoeff + 0 * Ngrid +i][1 * Ncoeff + j] = M0_[0 * Ngrid + i][1 * Ncoeff + j];

      //      MM[2 * Ncoeff + 1 * Ngrid +i][0 * Ncoeff + j] = M1_[0 * Ngrid + i][0 * Ncoeff + j];
      //      MM[2 * Ncoeff + 1 * Ngrid +i][1 * Ncoeff + j] = M1_[0 * Ngrid + i][1 * Ncoeff + j];

      //      MM[2 * Ncoeff + 2 * Ngrid +i][0 * Ncoeff + j] = M0_[0 * Ngrid + i][1 * Ncoeff + j];
      //      MM[2 * Ncoeff + 3 * Ngrid +i][1 * Ncoeff + j] = M0_[0 * Ngrid + i][0 * Ncoeff + j];

      //      MM[2 * Ncoeff + 4 * Ngrid +i][0 * Ncoeff + j] = M1_[0 * Ngrid + i][1 * Ncoeff + j];
      //      MM[2 * Ncoeff + 5 * Ngrid +i][1 * Ncoeff + j] = M1_[0 * Ngrid + i][0 * Ncoeff + j];
      //    }
      //  }

      //  Matrix<ValueType> MMM(2*Ncoeff + 6*Ngrid, 2*Ncoeff);
      //  MMM.SetZero();
      //  for (Integer i=0;i<Ncoeff;i++) {
      //    MMM[i+Ncoeff*0][i+Ncoeff*0]=coeff_weight[i];
      //    MMM[i+Ncoeff*1][i+Ncoeff*1]=coeff_weight[i];
      //  }
      //  for (Integer i=0;i<Ngrid;i++) {
      //    for (Integer j=0;j<Ncoeff;j++) {
      //      // MMM[2 * Ncoeff + 0 * Ngrid +i][0 * Ncoeff + j] = M0_[0 * Ngrid + i][0 * Ncoeff + j];
      //      // MMM[2 * Ncoeff + 0 * Ngrid +i][1 * Ncoeff + j] = M0_[0 * Ngrid + i][1 * Ncoeff + j];

      //      // MMM[2 * Ncoeff + 1 * Ngrid +i][0 * Ncoeff + j] = M1_[0 * Ngrid + i][0 * Ncoeff + j];
      //      // MMM[2 * Ncoeff + 1 * Ngrid +i][1 * Ncoeff + j] = M1_[0 * Ngrid + i][1 * Ncoeff + j];

      //      MMM[2 * Ncoeff + 2 * Ngrid +i][0 * Ncoeff + j] = M0_[0 * Ngrid + i][1 * Ncoeff + j];
      //      MMM[2 * Ncoeff + 3 * Ngrid +i][1 * Ncoeff + j] = M0_[0 * Ngrid + i][0 * Ncoeff + j];

      //      MMM[2 * Ncoeff + 4 * Ngrid +i][0 * Ncoeff + j] = M1_[0 * Ngrid + i][1 * Ncoeff + j];
      //      MMM[2 * Ncoeff + 5 * Ngrid +i][1 * Ncoeff + j] = M1_[0 * Ngrid + i][0 * Ncoeff + j];
      //    }
      //  }


      //  M[dir] = (MM.pinv(1e-10) * MMM).Transpose();
      //}
      //M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      ////M[dir]=M[dir]*M[dir];
      }
    }

    Matrix<ValueType> x(dof, 2 * Ncoeff), y(dof, 2 * Ncoeff);
    for (Long i = 0; i < dof; i++) {
      for (Integer j = 0; j < Ncoeff; j++) {
        x[i][Ncoeff * 0 + j] = coeff0[i * Ncoeff + j];
        x[i][Ncoeff * 1 + j] = coeff1[i * Ncoeff + j];
      }
    }
    Matrix<ValueType>::GEMM(y, x, M[dir0][dir1]);
    for (Long i = 0; i < dof; i++) {
      for (Integer j = 0; j < Ncoeff; j++) {
        coeff0[i * Ncoeff + j] = y[i][Ncoeff * 0 + j];
        coeff1[i * Ncoeff + j] = y[i][Ncoeff * 1 + j];
      }
    }
  }

  template <Integer DIM, Integer CONTINUITY> static void MakeContinuousEdge(Vector<ValueType> &coeff0, Vector<ValueType> &coeff1, Integer order, Integer dir0, Integer dir1, Integer norm0, Integer norm1) {
    SCTL_ASSERT(DIM==2);
    if (dir0>=2*DIM || dir1>=2*DIM) return;
    Integer Ncoeff = 1;
    for (Integer i = 0; i < DIM; i++) Ncoeff = (Ncoeff * (order + i)) / (i + 1);

    Long dof = coeff0.Dim() / Ncoeff;
    SCTL_ASSERT(coeff0.Dim() == Ncoeff * dof);
    SCTL_ASSERT(coeff1.Dim() == Ncoeff * dof);

    static Matrix<Matrix<ValueType>> M(2*DIM, 2*DIM);
    static Matrix<Matrix<ValueType>> MM(2*DIM, 2*DIM);
    if (M[dir0][dir1].Dim(0) != 2 * Ncoeff) {
      Integer Ngrid = pow<Integer>(order, DIM - 1);
      Vector<ValueType> nodes;
      Nodes<1>(order, nodes);

      Matrix<ValueType> Mtrunc(2*Ncoeff, 2*Ncoeff);
      { // Set Mtrunc
        Vector<ValueType> w;
        w.SetZero();
        for (Integer i=0;i<order;i++){
          for (Integer j=0;j<order;j++){
            if (i+j<order) {
              w.PushBack(i<order-CONTINUITY*2-1 && j<order-CONTINUITY*2-1);
            }
          }
        }
        Mtrunc.SetZero();
        for (Integer i=0;i<Ncoeff;i++){
          Mtrunc[i + Ncoeff * 0][i + Ncoeff * 0] = w[i];
          Mtrunc[i + Ncoeff * 1][i + Ncoeff * 1] = w[i];
        }
      }

      Matrix<ValueType> M_diff(2*Ncoeff, Ngrid);
      { // Set M_diff
        M_diff.SetZero();
        StaticArray<Vector<ValueType>, DIM> nodes_;
        for (Integer i = 0; i < DIM; i++) { // Set nodes_
          nodes_[i].ReInit(nodes.Dim(), nodes.begin(), false);
        }
        Vector<ValueType> nodes0, nodes1;
        nodes0.PushBack(0);
        nodes1.PushBack(1);

        Vector<ValueType> value;
        Vector<ValueType> coeff(Ncoeff);
        coeff.SetZero();
        for (Integer i = 0; i < Ncoeff; i++) {
          coeff[i]=0.5;
          value.ReInit(Ngrid, M_diff[i + Ncoeff * 0], false);
          nodes_[dir0/2].ReInit(1, (dir0 & 1 ? nodes1.begin() : nodes0.begin()), false);
          Eval<DIM>(order, coeff, nodes_, value);
          nodes_[dir0/2].ReInit(nodes.Dim(), nodes.begin(), false);

          coeff[i]=-0.5;
          value.ReInit(Ngrid, M_diff[i + Ncoeff * 1], false);
          nodes_[dir1/2].ReInit(1, (dir1 & 1 ? nodes1.begin() : nodes0.begin()), false);
          Eval<DIM>(order, coeff, nodes_, value);
          nodes_[dir1/2].ReInit(nodes.Dim(), nodes.begin(), false);

          coeff[i]=0;
        }
      }

      Matrix<ValueType> M_grad(2 * Ncoeff, 2 * Ncoeff);
      { // Set M_grad
        M_grad.SetZero();
        Vector<ValueType> coeff(Ncoeff * Ncoeff), coeff_grad;
        coeff.SetZero();
        for(Integer i = 0; i < Ncoeff; i++) coeff[i * Ncoeff + i] = 1;
        Grad<DIM>(order, coeff, &coeff_grad);
        for (Integer i = 0; i < Ncoeff; i++){
          for (Integer j = 0; j < Ncoeff; j++){
            M_grad[i + Ncoeff * 0][j + Ncoeff * 0] = coeff_grad[j + (i * DIM + dir0/2) * Ncoeff];
            M_grad[i + Ncoeff * 1][j + Ncoeff * 1] = coeff_grad[j + (i * DIM + dir1/2) * Ncoeff];
          }
        }
      }

      auto fn_perturb = [&](std::function<ValueType(ValueType)> fn, bool even) { // Set M0
        Matrix<ValueType> M0(Ngrid, 2 * Ncoeff);
        M0.SetZero();
        { // dir0
          Integer N0=pow<Integer>(order, dir0/2);
          Integer N1=pow<Integer>(order, 1);
          Integer N2=pow<Integer>(order, DIM - dir0/2 - 1);
          SCTL_ASSERT(N0 * N2 == Ngrid);
          Vector<ValueType> val(Ngrid * order), coeff;
          val.SetZero();
          for (Integer i0=0;i0<N0;i0++){
            for (Integer i2=0;i2<N2;i2++){
              for (Integer i1=0;i1<N1;i1++){
                val[(i2*N1+i1)*N0+i0] = (dir0 & 1 ? fn(nodes[i1]) : fn(1.0 - nodes[i1])) * (even ? 1.0 : -1.0);
              }
              coeff.ReInit(Ncoeff, M0[i2 * N0 + i0] + Ncoeff * 0, false);
              Approx<DIM>(order, val, coeff);
              for (Integer i1=0;i1<N1;i1++){
                val[(i2*N1+i1)*N0+i0] = 0;
              }
            }
          }
        }
        { // dir1
          Integer N0=pow<Integer>(order, dir1/2);
          Integer N1=pow<Integer>(order, 1);
          Integer N2=pow<Integer>(order, DIM - dir1/2 - 1);
          SCTL_ASSERT(N0 * N2 == Ngrid);
          Vector<ValueType> val(Ngrid * order), coeff;
          val.SetZero();
          for (Integer i0=0;i0<N0;i0++){
            for (Integer i2=0;i2<N2;i2++){
              for (Integer i1=0;i1<N1;i1++){
                val[(i2*N1+i1)*N0+i0] = (dir1 & 1 ? fn(nodes[i1]) : fn(1.0 - nodes[i1]));
              }
              coeff.ReInit(Ncoeff, M0[i2 * N0 + i0] + Ncoeff * 1, false);
              Approx<DIM>(order, val, coeff);
              for (Integer i1=0;i1<N1;i1++){
                val[(i2*N1+i1)*N0+i0] = 0;
              }
            }
          }
        }
        return M0;
      };

      Matrix<ValueType> Mfilter[2];
      { // Set Mfilter
        Mfilter[0].ReInit(2*Ncoeff, 2*Ncoeff);
        Mfilter[1].ReInit(2*Ncoeff, 2*Ncoeff);
        Mfilter[0].SetZero();
        Mfilter[1].SetZero();
        for (Integer i=0;i<Ncoeff;i++) {
          Mfilter[0][i + Ncoeff * 0][i + Ncoeff * 0] = 1;
          Mfilter[1][i + Ncoeff * 1][i + Ncoeff * 1] = 1;
        }
      }

      if (CONTINUITY == 0) {
        auto fn0 = [](ValueType x) {return x;};
        Matrix<ValueType> M0 = fn_perturb(fn0, 0);
        M[dir0][dir1] = M_diff * M0;
      } else if (CONTINUITY == 1) {
        auto fn0 = [](ValueType x) {return (-2*x + 3) * x * x;};
        auto fn1 = [](ValueType x) {return (1 - x) * x * x;};
        Matrix<ValueType> M0 = fn_perturb(fn0, 0);
        Matrix<ValueType> M1 = fn_perturb(fn1, 1);
        M[dir0][dir1] = M_diff * M0;
        if (dir0 & 1) M[dir0][dir1] += (M_grad+M_grad) * (Mfilter[0] * M_diff * M1 * Mfilter[0]);
        else          M[dir0][dir1] -= (M_grad+M_grad) * (Mfilter[0] * M_diff * M1 * Mfilter[0]);
        if (dir1 & 1) M[dir0][dir1] -= (M_grad+M_grad) * (Mfilter[1] * M_diff * M1 * Mfilter[1]);
        else          M[dir0][dir1] += (M_grad+M_grad) * (Mfilter[1] * M_diff * M1 * Mfilter[1]);
      }
      if (CONTINUITY == 1) {
        auto fn1 = [](ValueType x) {return (1 - x) * x * x;};
        Matrix<ValueType> M1 = fn_perturb(fn1, 1);
        MM[dir0][dir1] = M_grad * M_diff * M1;

        for (Integer i=0;i<Ncoeff;i++){
          for (Integer j=0;j<2*Ncoeff;j++){
            MM[dir0][dir1][i + Ncoeff*1][j]*=-1;
          }
        }

        for (Integer i=0;i<2*Ncoeff;i++){
          for (Integer j=0;j<Ncoeff;j++){
            MM[dir0][dir1][i][j + Ncoeff*0]*=(dir0 & 1 ? 1.0 : -1.0);
            MM[dir0][dir1][i][j + Ncoeff*1]*=(dir1 & 1 ? 1.0 : -1.0);
          }
        }
      }

      for (Integer i=0;i<2*Ncoeff;i++){
        M[dir0][dir1][i][i]+=1.0;
        MM[dir0][dir1][i][i]+=1.0;
      }
      M[dir0][dir1] = Mtrunc * M[dir0][dir1] * Mtrunc;
      MM[dir0][dir1] = Mtrunc * MM[dir0][dir1] * Mtrunc;

      for (Integer i=0;i<10;i++) {
        M[dir0][dir1]=M[dir0][dir1]*M[dir0][dir1];
      }
    }

    Matrix<ValueType> x(dof, 2 * Ncoeff), y(dof, 2 * Ncoeff);
    for (Long i = 0; i < dof; i++) {
      for (Integer j = 0; j < Ncoeff; j++) {
        x[i][Ncoeff * 0 + j] = coeff0[i * Ncoeff + j];
        x[i][Ncoeff * 1 + j] = coeff1[i * Ncoeff + j];
      }
    }
    Matrix<ValueType>::GEMM(y, x, M[dir0][dir1]);
    { ////
      Matrix<ValueType> xx(1, 2*Ncoeff), yy(1, 2*Ncoeff);
      for (Integer j = 0; j < Ncoeff; j++) {
        xx[0][Ncoeff * 0 + j] = coeff0[norm0 * Ncoeff + j];
        xx[0][Ncoeff * 1 + j] = coeff1[norm1 * Ncoeff + j];
      }
      Matrix<ValueType>::GEMM(yy, xx, MM[dir0][dir1]);
      for (Integer j = 0; j < Ncoeff; j++) {
        y[norm0][Ncoeff * 0 + j] = yy[0][Ncoeff * 0 + j];
        y[norm1][Ncoeff * 1 + j] = yy[0][Ncoeff * 1 + j];
      }
    }
    for (Long i = 0; i < dof; i++) {
      for (Integer j = 0; j < Ncoeff; j++) {
        coeff0[i * Ncoeff + j] = y[i][Ncoeff * 0 + j];
        coeff1[i * Ncoeff + j] = y[i][Ncoeff * 1 + j];
      }
    }
  }

  static void quad_rule(Integer order, Vector<ValueType>& x, Vector<ValueType>& w) {
    static Vector<Vector<ValueType>> x_lst(10000);
    static Vector<Vector<ValueType>> w_lst(x_lst.Dim());
    SCTL_ASSERT(order < x_lst.Dim());

    if (x.Dim() != order) x.ReInit(order);
    if (w.Dim() != order) w.ReInit(order);
    if (!order) return;

    bool done = false;
    #pragma omp critical(SCTL_QUAD_RULE)
    if (x_lst[order].Dim()) {
      Vector<ValueType>& x_ = x_lst[order];
      Vector<ValueType>& w_ = w_lst[order];
      for (Integer i = 0; i < order; i++) {
        x[i] = x_[i];
        w[i] = w_[i];
      }
      done = true;
    }
    if (done) return;

    Vector<ValueType> x_(order);
    Vector<ValueType> w_(order);
    if (std::is_same<ValueType, double>::value || std::is_same<ValueType, float>::value) {  // Gauss-Legendre quadrature nodes and weights
      Vector<double> xd(order);
      Vector<double> wd(order);
      int kind = 1;
      double alpha = 0.0, beta = 0.0, a = -1.0, b = 1.0;
      cgqf(order, kind, (double)alpha, (double)beta, (double)a, (double)b, &xd[0], &wd[0]);
      for (Integer i = 0; i < order; i++) {
        x_[i] = (ValueType)(0.5 * xd[i] + 0.5);
        w_[i] = (ValueType)(0.5 * wd[i]);
      }
    } else {  // Chebyshev quadrature nodes and weights
      cheb_nodes_1d(order, x_);
      for (Integer i = 0; i < order; i++) w_[i] = 0;

      Vector<ValueType> V_cheb(order * order);
      cheb_basis_1d(order, x_, V_cheb);
      for (Integer i = 0; i < order; i++) V_cheb[i] /= 2.0;
      Matrix<ValueType> M(order, order, V_cheb.begin());

      Vector<ValueType> w_sample(order);
      w_sample.SetZero();
      for (Integer i = 0; i < order; i += 2) w_sample[i] = -((ValueType)2.0 / (i + 1) / (i - 1));

      for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
          w_[j] += M[i][j] * w_sample[i] / order;
        }
      }
    }
    #pragma omp critical(SCTL_QUAD_RULE)
    if (!x_lst[order].Dim()) {  // Set x_lst, w_lst
      x_lst[order].Swap(x_);
      w_lst[order].Swap(w_);
    }
    quad_rule(order, x, w);
  }

 private:
  BasisInterface() {
    void (*EvalBasis1D)(Integer, const Vector<ValueType>&, Vector<ValueType>&) = Derived::EvalBasis1D;
    void (*Nodes1D)(Integer, Vector<ValueType>&) = Derived::Nodes1D;
  }

  static void cheb_nodes_1d(Integer order, Vector<ValueType>& nodes) {
    if (nodes.Dim() != order) nodes.ReInit(order);
    for (Integer i = 0; i < order; i++) {
      nodes[i] = -cos<ValueType>((i + (ValueType)0.5) * const_pi<ValueType>() / order) * (ValueType)0.5 + (ValueType)0.5;
    }
  }

  static void cheb_basis_1d(Integer order, const Vector<ValueType>& x, Vector<ValueType>& y) {
    Integer n = x.Dim();
    if (y.Dim() != order * n) y.ReInit(order * n);

    if (order > 0) {
      for (Long i = 0; i < n; i++) {
        y[i] = 1.0;
      }
    }
    if (order > 1) {
      for (Long i = 0; i < n; i++) {
        y[i + n] = x[i] * 2 - 1;
      }
    }
    for (Integer i = 2; i < order; i++) {
      for (Long j = 0; j < n; j++) {
        y[i * n + j] = 2 * y[n + j] * y[i * n - 1 * n + j] - y[i * n - 2 * n + j];
      }
    }
  }

  static ValueType machine_eps() {
    ValueType eps = 1.0;
    while (eps + (ValueType)1.0 > 1.0) eps *= 0.5;
    return eps;
  }

  template <Integer DIM, Integer SUBDIM, class Kernel> static void Integ_(Matrix<ValueType>& Mcoeff, Integer order, ConstIterator<ValueType> trg_, ValueType side, Integer src_face, const Kernel& ker, Integer Nq = 0) {
    static const ValueType eps = machine_eps() * 64;
    ValueType side_inv = 1.0 / side;
    if (!Nq) Nq = order;

    Vector<ValueType> qp, qw;
    quad_rule(Nq, qp, qw);

    Integer Ncoeff;
    {  // Set Ncoeff
      Ncoeff = 1;
      for (Integer i = 0; i < SUBDIM; i++) Ncoeff = (Ncoeff * (order + i)) / (i + 1);
    }
    StaticArray<Integer, 2> kdim;
    kdim[0] = ker.Dim(0);
    kdim[1] = ker.Dim(1);

    StaticArray<Integer, DIM> perm0;
    StaticArray<ValueType, DIM> trg;  // target after rotation
    {                                 // Set perm0
      SCTL_ASSERT(0 <= src_face && src_face < 2 * DIM);
      if (SUBDIM == DIM - 1) {
        for (Integer i = 0; i < DIM; i++) {
          perm0[i] = (i + (src_face >> 1) + 1) % DIM;
        }
      } else {
        for (Integer i = 0; i < DIM; i++) {
          perm0[i] = i;
        }
      }
      for (Integer i = 0; i < DIM; i++) trg[i] = trg_[perm0[i]];
      if (SUBDIM == DIM - 1) trg[DIM - 1] -= side * (src_face & 1);
    }

    Vector<ValueType> r;
    {  // Set r
      Vector<ValueType> r_;
      r_.PushBack(0);
      for (Integer i = 0; i < SUBDIM; i++) {
        r_.PushBack(fabs(trg[i] - 0.0));
        r_.PushBack(fabs(trg[i] - side));
      }
      std::sort(r_.begin(), r_.begin() + r_.Dim());

      ValueType r0, r1 = r_[r_.Dim() - 1];
      r0 = (r1 > side ? r1 - side : 0.0);
      for (Integer i = SUBDIM; i < DIM; i++) r0 = std::max(r0, fabs(trg[i]));
      if (r0 > eps) r.PushBack(-r0);
      r.PushBack(r0);

      for (Integer i = 0; i < r_.Dim(); i++) {
        if (r_[i] > r0) {
          while (r[r.Dim() - 1] > 0.0 && 3.0 * r[r.Dim() - 1] < r_[i]) r.PushBack(3.0 * r[r.Dim() - 1]);
          r.PushBack(r_[i]);
        }
      }
    }

    // Work vectors
    StaticArray<Vector<ValueType>, SUBDIM> eval_mesh;
    StaticArray<Vector<ValueType>, SUBDIM> eval_poly;
    Vector<ValueType> eval_coord_tmp;
    Vector<ValueType> eval_coord;
    Vector<ValueType> kern_value;

    // Temporary vectors
    Vector<ValueType> r_src, n_src, v_src;
    {  // Init r_src, n_src, v_src
      r_src.ReInit(DIM);
      for (Integer k = 0; k < DIM; k++) r_src[k] = 0.0;
      if (SUBDIM == DIM - 1) {
        n_src.ReInit(DIM);
        for (Integer k = 0; k < DIM; k++) n_src[k] = 0.0;
        n_src[src_face >> 1] = (src_face & 1 ? -1.0 : 1.0);
      }
      v_src.ReInit(kdim[0]);
    }
    Vector<ValueType> v0;
    Vector<ValueType> v1;

    Matrix<ValueType> Mtensor(kdim[1] * kdim[0], pow<Integer>(order, SUBDIM));
    Mtensor.SetZero();

    for (Integer i0 = 0; i0 < r.Dim() - 1; i0++) {   // for each layer
      for (Integer i1 = 0; i1 < 2 * SUBDIM; i1++) {  // for each direction
        StaticArray<ValueType, 2 * SUBDIM> range0;
        StaticArray<ValueType, 2 * SUBDIM> range1;
        {  // Set range0, range1
          for (Integer k = 0; k < SUBDIM; k++) {
            if (i1 >> 1 == k) {
              ValueType s = (i1 & 1 ? 1.0 : -1.0);
              range0[k * 2 + 0] = trg[k] + s * r[i0 + 0];
              range0[k * 2 + 1] = trg[k] + s * r[i0 + 0];
              range1[k * 2 + 0] = trg[k] + s * r[i0 + 1];
              range1[k * 2 + 1] = trg[k] + s * r[i0 + 1];
            } else {
              range0[k * 2 + 0] = trg[k] - fabs(r[i0 + 0]);
              range0[k * 2 + 1] = trg[k] + fabs(r[i0 + 0]);
              range1[k * 2 + 0] = trg[k] - fabs(r[i0 + 1]);
              range1[k * 2 + 1] = trg[k] + fabs(r[i0 + 1]);
            }
          }
          for (Integer k = 0; k < 2 * SUBDIM; k++) {
            if (range0[k] > side) range0[k] = side;
            if (range0[k] < 0.0) range0[k] = 0.0;
            if (range1[k] > side) range1[k] = side;
            if (range1[k] < 0.0) range1[k] = 0.0;
          }

          bool continue_flag = false;
          for (Integer k = 0; k < SUBDIM; k++) {  // continue if volume if 0
            if (i1 >> 1 == k) {
              if (fabs(range0[2 * k + 0] - range1[2 * k + 0]) < eps && fabs(range0[2 * k + 1] - range1[2 * k + 1]) < eps) {
                continue_flag = true;
                break;
              }
            } else {
              if (fabs(range0[2 * k + 0] - range0[2 * k + 1]) < eps && fabs(range1[2 * k + 0] - range1[2 * k + 1]) < eps) {
                continue_flag = true;
                break;
              }
            }
          }
          if (continue_flag) continue;
        }
        for (Integer i2 = 0; i2 < Nq; i2++) {  // for each quadrature point
          StaticArray<ValueType, 2 * SUBDIM> range;
          for (Integer k = 0; k < 2 * SUBDIM; k++) {  // Set range
            range[k] = range0[k] + (range1[k] - range0[k]) * qp[i2];
          }
          for (Integer k = 0; k < SUBDIM; k++) {  // Set eval_mesh
            if (k == (i1 >> 1)) {
              eval_mesh[k].ReInit(1);
              eval_mesh[k][0] = range[2 * k];
            } else {
              eval_mesh[k].ReInit(Nq);
              for (Integer l = 0; l < Nq; l++) eval_mesh[k][l] = range[2 * k + 0] + (range[2 * k + 1] - range[2 * k + 0]) * qp[l];
            }
          }
          {  // Set eval_coord
            Integer N = 1;
            eval_coord.ReInit(0);
            for (Integer k = 0; k < SUBDIM; k++) {
              Integer Nk = eval_mesh[k].Dim();
              eval_coord_tmp.Swap(eval_coord);
              eval_coord.ReInit(Nk * N * DIM);
              for (Integer l0 = 0; l0 < Nk; l0++) {
                for (Integer l1 = 0; l1 < N; l1++) {
                  for (Integer l2 = 0; l2 < k; l2++) {
                    eval_coord[DIM * (N * l0 + l1) + l2] = eval_coord_tmp[DIM * l1 + l2];
                  }
                  eval_coord[DIM * (N * l0 + l1) + k] = trg[k] - eval_mesh[k][l0];
                }
              }
              N *= Nk;
            }
            StaticArray<ValueType, DIM> c;
            for (Integer k = 0; k < N; k++) {  // Rotate
              for (Integer l = 0; l < SUBDIM; l++) c[l] = eval_coord[k * DIM + l];
              for (Integer l = SUBDIM; l < DIM; l++) c[l] = trg[l];
              for (Integer l = 0; l < DIM; l++) eval_coord[k * DIM + perm0[l]] = c[l];
            }
          }
          for (Integer k = 0; k < SUBDIM; k++) {  // Set eval_poly
            Integer N = eval_mesh[k].Dim();
            for (Integer l = 0; l < eval_mesh[k].Dim(); l++) {  // Scale eval_mesh to [0, 1]
              eval_mesh[k][l] *= side_inv;
            }
            Derived::EvalBasis1D(order, eval_mesh[k], eval_poly[k]);
            if (k == (i1 >> 1)) {
              assert(N == 1);
              ValueType qscal = fabs(range1[i1] - range0[i1]);
              for (Integer l0 = 0; l0 < order; l0++) {
                eval_poly[k][l0] *= qscal * qw[i2];
              }
            } else {
              assert(N == Nq);
              ValueType qscal = (range[2 * k + 1] - range[2 * k + 0]);
              for (Integer l0 = 0; l0 < order; l0++) {
                for (Integer l1 = 0; l1 < N; l1++) {
                  eval_poly[k][N * l0 + l1] *= qscal * qw[l1];
                }
              }
            }
          }
          {  // Set kern_value
            Integer N = eval_coord.Dim() / DIM;
            kern_value.ReInit(kdim[0] * N * kdim[1]);
            kern_value.SetZero();
            for (Integer j = 0; j < kdim[0]; j++) {  // Evaluate ker
              for (Integer k = 0; k < kdim[0]; k++) v_src[k] = 0.0;
              v_src[j] = 1.0;
              Vector<ValueType> ker_value(N * kdim[1], kern_value.begin() + j * N * kdim[1], false);
              ker(r_src, n_src, v_src, eval_coord, ker_value);
            }
            {  // Transpose
              v0.ReInit(kern_value.Dim());
              for (Integer k = 0; k < v0.Dim(); k++) v0[k] = kern_value[k];
              Matrix<ValueType> M0(kdim[0], N * kdim[1], v0.begin(), false);
              Matrix<ValueType> M1(N * kdim[1], kdim[0], kern_value.begin(), false);
              for (Integer l0 = 0; l0 < M1.Dim(0); l0++) {  // Transpose
                for (Integer l1 = 0; l1 < M1.Dim(1); l1++) {
                  M1[l0][l1] = M0[l1][l0];
                }
              }
            }
          }
          {  // Set Update M
            Matrix<ValueType> Mkern(eval_mesh[SUBDIM - 1].Dim(), kern_value.Dim() / eval_mesh[SUBDIM - 1].Dim(), kern_value.begin(), false);
            for (Integer k = SUBDIM - 1; k >= 0; k--) {  // Compute v0
              Matrix<ValueType> Mpoly(order, eval_mesh[k].Dim(), eval_poly[k].begin(), false);

              v1.ReInit(Mpoly.Dim(0) * Mkern.Dim(1));
              Matrix<ValueType> Mcoef(Mpoly.Dim(0), Mkern.Dim(1), v1.begin(), false);
              Matrix<ValueType>::GEMM(Mcoef, Mpoly, Mkern);

              v0.ReInit(v1.Dim());
              Matrix<ValueType> Mt(Mkern.Dim(1), Mpoly.Dim(0), v0.begin(), false);
              for (Integer l0 = 0; l0 < Mt.Dim(0); l0++) {  // Transpose
                for (Integer l1 = 0; l1 < Mt.Dim(1); l1++) {
                  Mt[l0][l1] = Mcoef[l1][l0];
                }
              }

              if (k > 0) {  // Reinit Mkern
                Mkern.ReInit(eval_mesh[k - 1].Dim(), v0.Dim() / eval_mesh[k - 1].Dim(), v0.begin(), false);
              }
            }

            assert(v0.Dim() == Mtensor.Dim(0) * Mtensor.Dim(1));
            for (Integer k = 0; k < v0.Dim(); k++) {  // Update M
              Mtensor[0][k] += v0[k];
            }
          }
        }
        if (r[i0] < 0.0) break;
      }
    }

    Mtensor = Mtensor.Transpose();
    {  // Set Mcoeff
      if (Mcoeff.Dim(0) != kdim[1] || Mcoeff.Dim(1) != kdim[0] * Ncoeff) {
        Mcoeff.ReInit(kdim[1], kdim[0] * Ncoeff);
      }
      Vector<ValueType> Mtensor_(Mtensor.Dim(0) * Mtensor.Dim(1), Mtensor.begin(), false);
      Vector<ValueType> Mcoeff_(Mcoeff.Dim(0) * Mcoeff.Dim(1), Mcoeff.begin(), false);
      tensor2coeff<SUBDIM>(order, Mtensor_, Mcoeff_);
    }
  }

  static void diff_1d(Integer order, Matrix<ValueType>* M) {
    Vector<ValueType> nodes;
    Nodes<1>(order, nodes);
    Integer N = nodes.Dim();

    Matrix<ValueType> M0(N, N);
    for (Integer i = 0; i < N; i++) {
      for (Integer j = 0; j < N; j++) {
        M0[i][j] = 0;
        for (Integer l = 0; l < N; l++) {
          if (l != i) {
            ValueType Mij = 1;
            for (Integer k = 0; k < N; k++) {
              if (k != i) {
                if (l == k) {
                  Mij *= 1 / (nodes[i] - nodes[k]);
                } else {
                  Mij *= (nodes[j] - nodes[k]) / (nodes[i] - nodes[k]);
                }
              }
            }
            M0[i][j] += Mij;
          }
        }
      }
    }

    Vector<ValueType> p;
    Derived::EvalBasis1D(order, nodes, p);
    Matrix<ValueType> Mp(order, N, p.begin(), false);
    M0 = Mp * M0;

    Vector<ValueType> coeff;
    Approx<1>(order, Vector<ValueType>(M0.Dim(0) * M0.Dim(1), M0.begin(), false), coeff);
    (*M) = Matrix<ValueType>(M0.Dim(0), coeff.Dim() / M0.Dim(0), coeff.begin(), false);
  }

  friend Derived;
};

template <class ValueType> class ChebBasis : public BasisInterface<ValueType, ChebBasis<ValueType>> {

 private:
  ChebBasis();

  static void Nodes1D(Integer order, Vector<ValueType>& nodes) { BasisInterface<ValueType, ChebBasis<ValueType>>::cheb_nodes_1d(order, nodes); }

  /**
   * \brief Returns the values of all Chebyshev polynomials up to degree d,
   * evaluated at points in the input vector. Output format:
   * { T0[x[0]], ..., T0[x[n-1]], T1[x[0]], ..., Td[x[n-1]] }
   */
  static void EvalBasis1D(Integer order, const Vector<ValueType>& x, Vector<ValueType>& y) { BasisInterface<ValueType, ChebBasis<ValueType>>::cheb_basis_1d(order, x, y); }

  friend BasisInterface<ValueType, ChebBasis<ValueType>>;
};

}  // end namespace

#endif  //_SCTL_CHEB_UTILS_HPP_
