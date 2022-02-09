#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(math_utils.hpp)

#if defined(SCTL_HAVE_CUDA)
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#if defined(SCTL_HAVE_BLAS)
#include SCTL_INCLUDE(blas.h)
#endif
#if defined(SCTL_HAVE_LAPACK)
#include SCTL_INCLUDE(lapack.h)
#endif

#include <omp.h>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>

namespace SCTL_NAMESPACE {
namespace mat {

template <class ValueType> inline void gemm(char TransA, char TransB, int M, int N, int K, ValueType alpha, Iterator<ValueType> A, int lda, Iterator<ValueType> B, int ldb, ValueType beta, Iterator<ValueType> C, int ldc) {
  if ((TransA == 'N' || TransA == 'n') && (TransB == 'N' || TransB == 'n')) {
    for (Long n = 0; n < N; n++) {    // Columns of C
      for (Long m = 0; m < M; m++) {  // Rows of C
        ValueType AxB = 0;
        for (Long k = 0; k < K; k++) {
          AxB += A[m + lda * k] * B[k + ldb * n];
        }
        C[m + ldc * n] = alpha * AxB + (beta == 0 ? 0 : beta * C[m + ldc * n]);
      }
    }
  } else if (TransA == 'N' || TransA == 'n') {
    for (Long n = 0; n < N; n++) {    // Columns of C
      for (Long m = 0; m < M; m++) {  // Rows of C
        ValueType AxB = 0;
        for (Long k = 0; k < K; k++) {
          AxB += A[m + lda * k] * B[n + ldb * k];
        }
        C[m + ldc * n] = alpha * AxB + (beta == 0 ? 0 : beta * C[m + ldc * n]);
      }
    }
  } else if (TransB == 'N' || TransB == 'n') {
    for (Long n = 0; n < N; n++) {    // Columns of C
      for (Long m = 0; m < M; m++) {  // Rows of C
        ValueType AxB = 0;
        for (Long k = 0; k < K; k++) {
          AxB += A[k + lda * m] * B[k + ldb * n];
        }
        C[m + ldc * n] = alpha * AxB + (beta == 0 ? 0 : beta * C[m + ldc * n]);
      }
    }
  } else {
    for (Long n = 0; n < N; n++) {    // Columns of C
      for (Long m = 0; m < M; m++) {  // Rows of C
        ValueType AxB = 0;
        for (Long k = 0; k < K; k++) {
          AxB += A[k + lda * m] * B[n + ldb * k];
        }
        C[m + ldc * n] = alpha * AxB + (beta == 0 ? 0 : beta * C[m + ldc * n]);
      }
    }
  }
}

#if defined(SCTL_HAVE_BLAS)
template <> inline void gemm<float>(char TransA, char TransB, int M, int N, int K, float alpha, Iterator<float> A, int lda, Iterator<float> B, int ldb, float beta, Iterator<float> C, int ldc) { sgemm_(&TransA, &TransB, &M, &N, &K, &alpha, &A[0], &lda, &B[0], &ldb, &beta, &C[0], &ldc); }

template <> inline void gemm<double>(char TransA, char TransB, int M, int N, int K, double alpha, Iterator<double> A, int lda, Iterator<double> B, int ldb, double beta, Iterator<double> C, int ldc) { dgemm_(&TransA, &TransB, &M, &N, &K, &alpha, &A[0], &lda, &B[0], &ldb, &beta, &C[0], &ldc); }
#endif

#if defined(SCTL_HAVE_CUDA)
template <> inline void cublasgemm<float>(char TransA, char TransB, int M, int N, int K, float alpha, Iterator<float> A, int lda, Iterator<float> B, int ldb, float beta, Iterator<float> C, int ldc) {
  cublasOperation_t cublasTransA, cublasTransB;
  cublasHandle_t *handle = CUDA_Lock::acquire_handle();
  if (TransA == 'T' || TransA == 't')
    cublasTransA = CUBLAS_OP_T;
  else if (TransA == 'N' || TransA == 'n')
    cublasTransA = CUBLAS_OP_N;
  if (TransB == 'T' || TransB == 't')
    cublasTransB = CUBLAS_OP_T;
  else if (TransB == 'N' || TransB == 'n')
    cublasTransB = CUBLAS_OP_N;
  cublasStatus_t status = cublasSgemm(*handle, cublasTransA, cublasTransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <> inline void cublasgemm<double>(char TransA, char TransB, int M, int N, int K, double alpha, Iterator<double> A, int lda, Iterator<double> B, int ldb, double beta, Iterator<double> C, int ldc) {
  cublasOperation_t cublasTransA, cublasTransB;
  cublasHandle_t *handle = CUDA_Lock::acquire_handle();
  if (TransA == 'T' || TransA == 't')
    cublasTransA = CUBLAS_OP_T;
  else if (TransA == 'N' || TransA == 'n')
    cublasTransA = CUBLAS_OP_N;
  if (TransB == 'T' || TransB == 't')
    cublasTransB = CUBLAS_OP_T;
  else if (TransB == 'N' || TransB == 'n')
    cublasTransB = CUBLAS_OP_N;
  cublasStatus_t status = cublasDgemm(*handle, cublasTransA, cublasTransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
#endif

//#define SCTL_SVD_DEBUG

template <class ValueType> static inline void GivensL(Iterator<ValueType> S_, const StaticArray<Long, 2> &dim, Long m, ValueType a, ValueType b) {
  auto S = [S_,dim](Long i, Long j) -> ValueType& { return S_[(i) * dim[1] + (j)]; };

  ValueType r = sqrt<ValueType>(a * a + b * b);
  if (r == 0) return;
  ValueType c = a / r;
  ValueType s = -b / r;

#pragma omp parallel for
  for (Long i = 0; i < dim[1]; i++) {
    ValueType S0 = S(m + 0, i);
    ValueType S1 = S(m + 1, i);
    S(m, i) += S0 * (c - 1);
    S(m, i) += S1 * (-s);

    S(m + 1, i) += S0 * (s);
    S(m + 1, i) += S1 * (c - 1);
  }
}

template <class ValueType> static inline void GivensR(Iterator<ValueType> S_, const StaticArray<Long, 2> &dim, Long m, ValueType a, ValueType b) {
  auto S = [S_,dim](Long i, Long j) -> ValueType& { return S_[(i) * dim[1] + (j)]; };

  ValueType r = sqrt<ValueType>(a * a + b * b);
  if (r == 0) return;
  ValueType c = a / r;
  ValueType s = -b / r;

#pragma omp parallel for
  for (Long i = 0; i < dim[0]; i++) {
    ValueType S0 = S(i, m + 0);
    ValueType S1 = S(i, m + 1);
    S(i, m) += S0 * (c - 1);
    S(i, m) += S1 * (-s);

    S(i, m + 1) += S0 * (s);
    S(i, m + 1) += S1 * (c - 1);
  }
}

template <class ValueType> static inline void SVD(const StaticArray<Long, 2> &dim, Iterator<ValueType> U_, Iterator<ValueType> S_, Iterator<ValueType> V_, ValueType eps = -1) {
  auto U = [U_,dim](Long i, Long j) -> ValueType& { return U_[(i) * dim[0] + (j)]; };
  auto S = [S_,dim](Long i, Long j) -> ValueType& { return S_[(i) * dim[1] + (j)]; };
  auto V = [V_,dim](Long i, Long j) -> ValueType& { return V_[(i) * dim[1] + (j)]; };

  assert(dim[0] >= dim[1]);
#ifdef SCTL_SVD_DEBUG
  Matrix<ValueType> M0(dim[0], dim[1], S_);
#endif

  {  // Bi-diagonalization
    Long n = std::min(dim[0], dim[1]);
    std::vector<ValueType> house_vec(std::max(dim[0], dim[1]));
    for (Long i = 0; i < n; i++) {
      // Column Householder
      {
        ValueType x1 = S(i, i);
        if (x1 < 0) x1 = -x1;

        ValueType x_inv_norm = 0;
        for (Long j = i; j < dim[0]; j++) {
          x_inv_norm += S(j, i) * S(j, i);
        }
        if (x_inv_norm > 0) x_inv_norm = 1 / sqrt<ValueType>(x_inv_norm);

        ValueType alpha = sqrt<ValueType>(1 + x1 * x_inv_norm);
        ValueType beta = x_inv_norm / alpha;
        if (x_inv_norm == 0) alpha = 0; // nothing to do

        house_vec[i] = -alpha;
        for (Long j = i + 1; j < dim[0]; j++) {
          house_vec[j] = -beta * S(j, i);
        }
        if (S(i, i) < 0)
          for (Long j = i + 1; j < dim[0]; j++) {
            house_vec[j] = -house_vec[j];
          }
      }
#pragma omp parallel for
      for (Long k = i; k < dim[1]; k++) {
        ValueType dot_prod = 0;
        for (Long j = i; j < dim[0]; j++) {
          dot_prod += S(j, k) * house_vec[j];
        }
        for (Long j = i; j < dim[0]; j++) {
          S(j, k) -= dot_prod * house_vec[j];
        }
      }
#pragma omp parallel for
      for (Long k = 0; k < dim[0]; k++) {
        ValueType dot_prod = 0;
        for (Long j = i; j < dim[0]; j++) {
          dot_prod += U(k, j) * house_vec[j];
        }
        for (Long j = i; j < dim[0]; j++) {
          U(k, j) -= dot_prod * house_vec[j];
        }
      }

      // Row Householder
      if (i >= n - 1) continue;
      {
        ValueType x1 = S(i, i + 1);
        if (x1 < 0) x1 = -x1;

        ValueType x_inv_norm = 0;
        for (Long j = i + 1; j < dim[1]; j++) {
          x_inv_norm += S(i, j) * S(i, j);
        }
        if (x_inv_norm > 0) x_inv_norm = 1 / sqrt<ValueType>(x_inv_norm);

        ValueType alpha = sqrt<ValueType>(1 + x1 * x_inv_norm);
        ValueType beta = x_inv_norm / alpha;
        if (x_inv_norm == 0) alpha = 0; // nothing to do

        house_vec[i + 1] = -alpha;
        for (Long j = i + 2; j < dim[1]; j++) {
          house_vec[j] = -beta * S(i, j);
        }
        if (S(i, i + 1) < 0)
          for (Long j = i + 2; j < dim[1]; j++) {
            house_vec[j] = -house_vec[j];
          }
      }
#pragma omp parallel for
      for (Long k = i; k < dim[0]; k++) {
        ValueType dot_prod = 0;
        for (Long j = i + 1; j < dim[1]; j++) {
          dot_prod += S(k, j) * house_vec[j];
        }
        for (Long j = i + 1; j < dim[1]; j++) {
          S(k, j) -= dot_prod * house_vec[j];
        }
      }
#pragma omp parallel for
      for (Long k = 0; k < dim[1]; k++) {
        ValueType dot_prod = 0;
        for (Long j = i + 1; j < dim[1]; j++) {
          dot_prod += V(j, k) * house_vec[j];
        }
        for (Long j = i + 1; j < dim[1]; j++) {
          V(j, k) -= dot_prod * house_vec[j];
        }
      }
    }
  }

  Long k0 = 0;
  Long iter = 0;
  if (eps < 0) {
    eps = 1.0;
    while (eps + (ValueType)1.0 > 1.0) eps *= 0.5;
    eps *= 64.0;
  }
  while (k0 < dim[1] - 1) {  // Diagonalization
    iter++;

    ValueType S_max = 0.0;
    for (Long i = 0; i < dim[1]; i++) S_max = (S_max > fabs<ValueType>(S(i, i)) ? S_max : fabs<ValueType>(S(i, i)));
    for (Long i = 0; i < dim[1] - 1; i++) S_max = (S_max > fabs<ValueType>(S(i, i + 1)) ? S_max : fabs<ValueType>(S(i, i + 1)));

    // while(k0<dim[1]-1 && fabs<ValueType>(S(k0,k0+1))<=eps*(fabs<ValueType>(S(k0,k0))+fabs<ValueType>(S(k0+1,k0+1)))) k0++;
    while (k0 < dim[1] - 1 && fabs<ValueType>(S(k0, k0 + 1)) <= eps * S_max) k0++;
    if (k0 == dim[1] - 1) continue;

    Long n = k0 + 2;
    // while(n<dim[1] && fabs<ValueType>(S(n-1,n))>eps*(fabs<ValueType>(S(n-1,n-1))+fabs<ValueType>(S(n,n)))) n++;
    while (n < dim[1] && fabs<ValueType>(S(n - 1, n)) > eps * S_max) n++;

    ValueType alpha = 0;
    ValueType beta = 0;
    if (n - k0 == 2 && fabs<ValueType>(S(k0, k0)) < eps * S_max && fabs<ValueType>(S(k0 + 1, k0 + 1)) < eps * S_max) {  // Compute mu
      alpha=0;
      beta=1;
    } else {
      StaticArray<ValueType, 2 * 2> C;
      C[0 * 2 + 0] = S(n - 2, n - 2) * S(n - 2, n - 2);
      if (n - k0 > 2) C[0 * 2 + 0] += S(n - 3, n - 2) * S(n - 3, n - 2);
      C[0 * 2 + 1] = S(n - 2, n - 2) * S(n - 2, n - 1);
      C[1 * 2 + 0] = S(n - 2, n - 2) * S(n - 2, n - 1);
      C[1 * 2 + 1] = S(n - 1, n - 1) * S(n - 1, n - 1) + S(n - 2, n - 1) * S(n - 2, n - 1);

      ValueType b = -(C[0 * 2 + 0] + C[1 * 2 + 1]) / 2;
      ValueType c = C[0 * 2 + 0] * C[1 * 2 + 1] - C[0 * 2 + 1] * C[1 * 2 + 0];
      ValueType d = 0;
      if (fabs(b * b - c) > eps*b*b)
        d = sqrt<ValueType>(b * b - c);
      else {
        ValueType b = (C[0 * 2 + 0] - C[1 * 2 + 1]) / 2;
        ValueType c = -C[0 * 2 + 1] * C[1 * 2 + 0];
        if (b * b - c > 0) d = sqrt<ValueType>(b * b - c);
      }

      ValueType lambda1 = -b + d;
      ValueType lambda2 = -b - d;

      ValueType d1 = lambda1 - C[1 * 2 + 1];
      d1 = (d1 < 0 ? -d1 : d1);
      ValueType d2 = lambda2 - C[1 * 2 + 1];
      d2 = (d2 < 0 ? -d2 : d2);
      ValueType mu = (d1 < d2 ? lambda1 : lambda2);

      alpha = S(k0, k0) * S(k0, k0) - mu;
      beta = S(k0, k0) * S(k0, k0 + 1);
    }

    for (Long k = k0; k < n - 1; k++) {
      StaticArray<Long, 2> dimU;
      dimU[0] = dim[0];
      dimU[1] = dim[0];
      StaticArray<Long, 2> dimV;
      dimV[0] = dim[1];
      dimV[1] = dim[1];
      GivensR(S_, dim, k, alpha, beta);
      GivensL(V_, dimV, k, alpha, beta);

      alpha = S(k, k);
      beta = S(k + 1, k);
      GivensL(S_, dim, k, alpha, beta);
      GivensR(U_, dimU, k, alpha, beta);

      alpha = S(k, k + 1);
      beta = S(k, k + 2);
    }

    {  // Make S bi-diagonal again
      for (Long i0 = k0; i0 < n - 1; i0++) {
        for (Long i1 = 0; i1 < dim[1]; i1++) {
          if (i0 > i1 || i0 + 1 < i1) S(i0, i1) = 0;
        }
      }
      for (Long i0 = 0; i0 < dim[0]; i0++) {
        for (Long i1 = k0; i1 < n - 1; i1++) {
          if (i0 > i1 || i0 + 1 < i1) S(i0, i1) = 0;
        }
      }
      for (Long i = 0; i < dim[1] - 1; i++) {
        if (fabs<ValueType>(S(i, i + 1)) <= eps * S_max) {
          S(i, i + 1) = 0;
        }
      }
    }
    // std::cout<<iter<<' '<<k0<<' '<<n<<'\n';
  }

  {  // Check Error
#ifdef SCTL_SVD_DEBUG
    Matrix<ValueType> U0(dim[0], dim[0], U_);
    Matrix<ValueType> S0(dim[0], dim[1], S_);
    Matrix<ValueType> V0(dim[1], dim[1], V_);
    Matrix<ValueType> E = M0 - U0 * S0 * V0;
    ValueType max_err = 0;
    ValueType max_nondiag0 = 0;
    ValueType max_nondiag1 = 0;
    for (Long i = 0; i < E.Dim(0); i++)
      for (Long j = 0; j < E.Dim(1); j++) {
        if (max_err < fabs<ValueType>(E[i][j])) max_err = fabs<ValueType>(E[i][j]);
        if ((i > j + 0 || i + 0 < j) && max_nondiag0 < fabs<ValueType>(S0[i][j])) max_nondiag0 = fabs<ValueType>(S0[i][j]);
        if ((i > j + 1 || i + 1 < j) && max_nondiag1 < fabs<ValueType>(S0[i][j])) max_nondiag1 = fabs<ValueType>(S0[i][j]);
      }
    std::cout << max_err << '\n';
    std::cout << max_nondiag0 << '\n';
    std::cout << max_nondiag1 << '\n';
#endif
  }
}

#undef SCTL_SVD_DEBUG

template <class ValueType> inline void svd(char *JOBU, char *JOBVT, int *M, int *N, Iterator<ValueType> A, int *LDA, Iterator<ValueType> S, Iterator<ValueType> U, int *LDU, Iterator<ValueType> VT, int *LDVT, Iterator<ValueType> WORK, int *LWORK, int *INFO) {
  StaticArray<Long, 2> dim;
  dim[0] = std::max(*N, *M);
  dim[1] = std::min(*N, *M);
  Iterator<ValueType> U_ = aligned_new<ValueType>(dim[0] * dim[0]);
  memset(U_, 0, dim[0] * dim[0]);
  Iterator<ValueType> V_ = aligned_new<ValueType>(dim[1] * dim[1]);
  memset(V_, 0, dim[1] * dim[1]);
  Iterator<ValueType> S_ = aligned_new<ValueType>(dim[0] * dim[1]);

  const Long lda = *LDA;
  const Long ldu = *LDU;
  const Long ldv = *LDVT;

  if (dim[1] == *M) {
    for (Long i = 0; i < dim[0]; i++)
      for (Long j = 0; j < dim[1]; j++) {
        S_[i * dim[1] + j] = A[i * lda + j];
      }
  } else {
    for (Long i = 0; i < dim[0]; i++)
      for (Long j = 0; j < dim[1]; j++) {
        S_[i * dim[1] + j] = A[j * lda + i];
      }
  }
  for (Long i = 0; i < dim[0]; i++) {
    U_[i * dim[0] + i] = 1;
  }
  for (Long i = 0; i < dim[1]; i++) {
    V_[i * dim[1] + i] = 1;
  }

  SVD<ValueType>(dim, U_, S_, V_, (ValueType) - 1);

  for (Long i = 0; i < dim[1]; i++) {  // Set S
    S[i] = S_[i * dim[1] + i];
  }
  if (dim[1] == *M) {  // Set U
    for (Long i = 0; i < dim[1]; i++)
      for (Long j = 0; j < *M; j++) {
        U[j + ldu * i] = V_[j + i * dim[1]] * (S[i] < 0.0 ? -1.0 : 1.0);
      }
  } else {
    for (Long i = 0; i < dim[1]; i++)
      for (Long j = 0; j < *M; j++) {
        U[j + ldu * i] = U_[i + j * dim[0]] * (S[i] < 0.0 ? -1.0 : 1.0);
      }
  }
  if (dim[0] == *N) {  // Set V
    for (Long i = 0; i < *N; i++)
      for (Long j = 0; j < dim[1]; j++) {
        VT[j + ldv * i] = U_[j + i * dim[0]];
      }
  } else {
    for (Long i = 0; i < *N; i++)
      for (Long j = 0; j < dim[1]; j++) {
        VT[j + ldv * i] = V_[i + j * dim[1]];
      }
  }
  for (Long i = 0; i < dim[1]; i++) {
    S[i] = S[i] * (S[i] < 0.0 ? -1.0 : 1.0);
  }

  aligned_delete<ValueType>(U_);
  aligned_delete<ValueType>(S_);
  aligned_delete<ValueType>(V_);

  if (0) {  // Verify
    StaticArray<Long, 2> dim;
    dim[0] = std::max(*N, *M);
    dim[1] = std::min(*N, *M);
    const Long lda = *LDA;
    const Long ldu = *LDU;
    const Long ldv = *LDVT;

    Matrix<ValueType> A1(*M, *N);
    Matrix<ValueType> S1(dim[1], dim[1]);
    Matrix<ValueType> U1(*M, dim[1]);
    Matrix<ValueType> V1(dim[1], *N);
    for (Long i = 0; i < *N; i++)
      for (Long j = 0; j < *M; j++) {
        A1[j][i] = A[j + i * lda];
      }
    S1.SetZero();
    for (Long i = 0; i < dim[1]; i++) {  // Set S
      S1[i][i] = S[i];
    }
    for (Long i = 0; i < dim[1]; i++)
      for (Long j = 0; j < *M; j++) {
        U1[j][i] = U[j + ldu * i];
      }
    for (Long i = 0; i < *N; i++)
      for (Long j = 0; j < dim[1]; j++) {
        V1[j][i] = VT[j + ldv * i];
      }
    std::cout << U1 *S1 *V1 - A1 << '\n';
  }
}

#if defined(SCTL_HAVE_LAPACK)
template <> inline void svd<float>(char *JOBU, char *JOBVT, int *M, int *N, Iterator<float> A, int *LDA, Iterator<float> S, Iterator<float> U, int *LDU, Iterator<float> VT, int *LDVT, Iterator<float> WORK, int *LWORK, int *INFO) { sgesvd_(JOBU, JOBVT, M, N, &A[0], LDA, &S[0], &U[0], LDU, &VT[0], LDVT, &WORK[0], LWORK, INFO); }

template <> inline void svd<double>(char *JOBU, char *JOBVT, int *M, int *N, Iterator<double> A, int *LDA, Iterator<double> S, Iterator<double> U, int *LDU, Iterator<double> VT, int *LDVT, Iterator<double> WORK, int *LWORK, int *INFO) { dgesvd_(JOBU, JOBVT, M, N, &A[0], LDA, &S[0], &U[0], LDU, &VT[0], LDVT, &WORK[0], LWORK, INFO); }
#endif

/**
 * \brief Computes the pseudo inverse of matrix M(n1xn2) (in row major form)
 * and returns the output M_(n2xn1). Original contents of M are destroyed.
 */
template <class ValueType> inline void pinv(Iterator<ValueType> M, int n1, int n2, ValueType eps, Iterator<ValueType> M_) {
  if (n1 * n2 == 0) return;
  int m = n2;
  int n = n1;
  int k = (m < n ? m : n);

  Iterator<ValueType> tU = aligned_new<ValueType>(m * k);
  Iterator<ValueType> tS = aligned_new<ValueType>(k);
  Iterator<ValueType> tVT = aligned_new<ValueType>(k * n);

  // SVD
  int INFO = 0;
  char JOBU = 'S';
  char JOBVT = 'S';

  // int wssize = max(3*min(m,n)+max(m,n), 5*min(m,n));
  int wssize = 3 * (m < n ? m : n) + (m > n ? m : n);
  int wssize1 = 5 * (m < n ? m : n);
  wssize = (wssize > wssize1 ? wssize : wssize1);

  Iterator<ValueType> wsbuf = aligned_new<ValueType>(wssize);

  svd(&JOBU, &JOBVT, &m, &n, M, &m, tS, tU, &m, tVT, &k, wsbuf, &wssize, &INFO);
  if (INFO != 0) std::cout << INFO << '\n';
  assert(INFO == 0);
  aligned_delete<ValueType>(wsbuf);

  ValueType eps_ = tS[0] * eps;
  for (int i = 0; i < k; i++)
    if (tS[i] < eps_)
      tS[i] = 0;
    else
      tS[i] = 1 / tS[i];

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      tU[i + j * m] *= tS[j];
    }
  }

  gemm<ValueType>('T', 'T', n, m, k, 1.0, tVT, k, tU, m, 0.0, M_, n);
  aligned_delete<ValueType>(tU);
  aligned_delete<ValueType>(tS);
  aligned_delete<ValueType>(tVT);
}

}  // end namespace mat
}  // end namespace SCTL_NAMESPACE
