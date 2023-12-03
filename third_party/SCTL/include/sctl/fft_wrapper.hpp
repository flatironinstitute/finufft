#ifndef _SCTL_FFT_WRAPPER_
#define _SCTL_FFT_WRAPPER_

#include <cmath>
#include <cassert>
#include <cstdlib>
#include <vector>
#if defined(SCTL_HAVE_FFTW) || defined(SCTL_HAVE_FFTWF)
#include <fftw3.h>
#ifdef SCTL_FFTW3_MKL
#include <fftw3_mkl.h>
#endif
#endif

#include <sctl/common.hpp>
#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(math_utils.hpp)

namespace SCTL_NAMESPACE {

template <class ValueType> class Complex {
  public:
    Complex<ValueType>(ValueType r=0, ValueType i=0) : real(r), imag(i) {}

    Complex<ValueType> operator-() const {
      Complex<ValueType> z;
      z.real = -real;
      z.imag = -imag;
      return z;
    }

    Complex<ValueType> conj() const {
      Complex<ValueType> z;
      z.real = real;
      z.imag = -imag;
      return z;
    }


    bool operator==(const Complex<ValueType>& x) const {
      return real == x.real && imag == x.imag;
    }

    bool operator!=(const Complex<ValueType>& x) const {
      return !((*this) == x);;
    }


    template <class ScalarType> void operator+=(const Complex<ScalarType>& x) {
      (*this) = (*this) + x;
    }

    template <class ScalarType> void operator-=(const Complex<ScalarType>& x) {
      (*this) = (*this) - x;
    }

    template <class ScalarType> void operator*=(const Complex<ScalarType>& x) {
      (*this) = (*this) * x;
    }

    template <class ScalarType> void operator/=(const Complex<ScalarType>& x) {
      (*this) = (*this) / x;
    }


    template <class ScalarType> Complex<ValueType> operator+(const ScalarType& x) const {
      Complex<ValueType> z;
      z.real = real + x;
      z.imag = imag;
      return z;
    }

    template <class ScalarType> Complex<ValueType> operator-(const ScalarType& x) const {
      Complex<ValueType> z;
      z.real = real - x;
      z.imag = imag;
      return z;
    }

    template <class ScalarType> Complex<ValueType> operator*(const ScalarType& x) const {
      Complex<ValueType> z;
      z.real = real * x;
      z.imag = imag * x;
      return z;
    }

    template <class ScalarType> Complex<ValueType> operator/(const ScalarType& y) const {
      Complex<ValueType> z;
      z.real = real / y;
      z.imag = imag / y;
      return z;
    }


    Complex<ValueType> operator+(const Complex<ValueType>& x) const {
      Complex<ValueType> z;
      z.real = real + x.real;
      z.imag = imag + x.imag;
      return z;
    }

    Complex<ValueType> operator-(const Complex<ValueType>& x) const {
      Complex<ValueType> z;
      z.real = real - x.real;
      z.imag = imag - x.imag;
      return z;
    }

    Complex<ValueType> operator*(const Complex<ValueType>& x) const {
      Complex<ValueType> z;
      z.real = real * x.real - imag * x.imag;
      z.imag = imag * x.real + real * x.imag;
      return z;
    }

    Complex<ValueType> operator/(const Complex<ValueType>& y) const {
      Complex<ValueType> z;
      ValueType y_inv = 1 / (y.real * y.real + y.imag * y.imag);
      z.real = (y.real * real + y.imag * imag) * y_inv;
      z.imag = (y.real * imag - y.imag * real) * y_inv;
      return z;
    }

    ValueType real;
    ValueType imag;
};

template <class ScalarType, class ValueType> Complex<ValueType> operator*(const ScalarType& x, const Complex<ValueType>& y) {
  Complex<ValueType> z;
  z.real = y.real * x;
  z.imag = y.imag * x;
  return z;
}

template <class ScalarType, class ValueType> Complex<ValueType> operator+(const ScalarType& x, const Complex<ValueType>& y) {
  Complex<ValueType> z;
  z.real = y.real + x;
  z.imag = y.imag;
  return z;
}

template <class ScalarType, class ValueType> Complex<ValueType> operator-(const ScalarType& x, const Complex<ValueType>& y) {
  Complex<ValueType> z;
  z.real = y.real - x;
  z.imag = y.imag;
  return z;
}

template <class ScalarType, class ValueType> Complex<ValueType> operator/(const ScalarType& x, const Complex<ValueType>& y) {
  Complex<ValueType> z;
  ValueType y_inv = 1 / (y.real * y.real + y.imag * y.imag);
  z.real =  (y.real * x) * y_inv;
  z.imag = -(y.imag * x) * y_inv;
  return z;
}

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Complex<ValueType>& V) {
  output << "(" << V.real <<"," << V.imag << ")";
  return output;
}


enum class FFT_Type {R2C, C2C, C2C_INV, C2R};

template <class ValueType, class FFT_Derived> class FFT_Base {

 public:

  FFT_Base() : dim{0,0}, fft_type(FFT_Type::R2C), howmany(0) {}

  Long Dim(Integer i) const {
    return dim[i];
  }

  static void test() {
    Vector<Long> fft_dim;
    fft_dim.PushBack(2);
    fft_dim.PushBack(5);
    fft_dim.PushBack(3);
    Long howmany = 3;

    if (1){ // R2C, C2R
      FFT_Derived myfft0, myfft1;
      myfft0.Setup(FFT_Type::R2C, howmany, fft_dim);
      myfft1.Setup(FFT_Type::C2R, howmany, fft_dim);
      Vector<ValueType> v0(myfft0.Dim(0)), v1, v2;
      for (int i = 0; i < v0.Dim(); i++) v0[i] = 1 + i;
      myfft0.Execute(v0, v1);
      myfft1.Execute(v1, v2);
      { // Print error
        ValueType err = 0;
        SCTL_ASSERT(v0.Dim() == v2.Dim());
        for (Long i = 0; i < v0.Dim(); i++) err = std::max(err, fabs(v0[i] - v2[i]));
        std::cout<<"Error : "<<err<<'\n';
      }
    }
    std::cout<<'\n';
    { // C2C, C2C_INV
      FFT_Derived myfft0, myfft1;
      myfft0.Setup(FFT_Type::C2C, howmany, fft_dim);
      myfft1.Setup(FFT_Type::C2C_INV, howmany, fft_dim);
      Vector<ValueType> v0(myfft0.Dim(0)), v1, v2;
      for (int i = 0; i < v0.Dim(); i++) v0[i] = 1 + i;
      myfft0.Execute(v0, v1);
      myfft1.Execute(v1, v2);
      { // Print error
        ValueType err = 0;
        SCTL_ASSERT(v0.Dim() == v2.Dim());
        for (Long i = 0; i < v0.Dim(); i++) err = std::max(err, fabs(v0[i] - v2[i]));
        std::cout<<"Error : "<<err<<'\n';
      }
    }
    std::cout<<'\n';
  }

 protected:

  static void check_align(const Vector<ValueType>& in, const Vector<ValueType>& out) {
    //SCTL_ASSERT_MSG((((uintptr_t)& in[0]) & ((uintptr_t)(SCTL_MEM_ALIGN - 1))) == 0, "sctl::FFT: Input vector not aligned to " <<SCTL_MEM_ALIGN<<" bytes!");
    //SCTL_ASSERT_MSG((((uintptr_t)&out[0]) & ((uintptr_t)(SCTL_MEM_ALIGN - 1))) == 0, "sctl::FFT: Output vector not aligned to "<<SCTL_MEM_ALIGN<<" bytes!");
    // TODO: copy to auxiliary array if unaligned
  }

  StaticArray<Long,2> dim;
  FFT_Type fft_type;
  Long howmany;
};

template <class ValueType> class FFT : public FFT_Base<ValueType, FFT<ValueType>> {

  typedef Complex<ValueType> ComplexType;

  struct FFTPlan {
    std::vector<Matrix<ValueType>> M;
  };

 public:

  FFT() = default;
  FFT (const FFT&) = delete;
  FFT& operator= (const FFT&) = delete;

  void Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads = 1) {
    Long rank = dim_vec.Dim();
    this->fft_type = fft_type_;
    this->howmany = howmany_;
    plan.M.resize(0);

    if (this->fft_type == FFT_Type::R2C) {
      plan.M.push_back(fft_r2c(dim_vec[rank - 1]));
      for (Long i = rank - 2; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]));
    } else if (this->fft_type == FFT_Type::C2C) {
      for (Long i = rank - 1; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]));
    } else if (this->fft_type == FFT_Type::C2C_INV) {
      for (Long i = rank - 1; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]).Transpose());
    } else if (this->fft_type == FFT_Type::C2R) {
      for (Long i = rank - 2; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]).Transpose());
      plan.M.push_back(fft_c2r(dim_vec[rank - 1]));
    }

    Long N0 = this->howmany * 2;
    Long N1 = this->howmany * 2;
    for (const auto M : plan.M) {
      N0 = N0 * M.Dim(0) / 2;
      N1 = N1 * M.Dim(1) / 2;
    }
    this->dim[0] = N0;
    this->dim[1] = N1;
  }

  void Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const {
    Long N0 = this->Dim(0);
    Long N1 = this->Dim(1);
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);
    this->check_align(in, out);

    Vector<ValueType> buff0(N0 + N1);
    Vector<ValueType> buff1(N0 + N1);
    Long rank = plan.M.size();
    if (rank <= 0) return;
    Long N = N0;

    if (this->fft_type == FFT_Type::C2R) {
      const Matrix<ValueType>& M = plan.M[rank - 1];
      transpose<ComplexType>(buff0.begin(), in.begin(), N / M.Dim(0), M.Dim(0) / 2);

      for (Long i = 0; i < rank - 1; i++) {
        const Matrix<ValueType>& M = plan.M[i];
        Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff0.begin(), false);
        Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), buff1.begin(), false);
        Matrix<ValueType>::GEMM(vo, vi, M);
        N = N * M.Dim(1) / M.Dim(0);
        transpose<ComplexType>(buff0.begin(), buff1.begin(), N / M.Dim(1), M.Dim(1) / 2);
      }
      transpose<ComplexType>(buff1.begin(), buff0.begin(), N / this->howmany / 2, this->howmany);

      Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff1.begin(), false);
      Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), out.begin(), false);
      Matrix<ValueType>::GEMM(vo, vi, M);
    } else {
      memcopy(buff0.begin(), in.begin(), in.Dim());
      for (Long i = 0; i < rank; i++) {
        const Matrix<ValueType>& M = plan.M[i];
        Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff0.begin(), false);
        Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), buff1.begin(), false);
        Matrix<ValueType>::GEMM(vo, vi, M);
        N = N * M.Dim(1) / M.Dim(0);
        transpose<ComplexType>(buff0.begin(), buff1.begin(), N / M.Dim(1), M.Dim(1) / 2);
      }
      transpose<ComplexType>(out.begin(), buff0.begin(), N / this->howmany / 2, this->howmany);
    }
  }

 private:

  static Matrix<ValueType> fft_r2c(Long N0) {
    ValueType s = 1 / sqrt<ValueType>(N0);
    Long N1 = (N0 / 2 + 1);
    Matrix<ValueType> M(N0, 2 * N1);
    for (Long j = 0; j < N0; j++)
      for (Long i = 0; i < N1; i++) {
        M[j][2 * i + 0] =  cos<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
        M[j][2 * i + 1] = -sin<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
      }
    return M;
  }

  static Matrix<ValueType> fft_c2c(Long N0) {
    ValueType s = 1 / sqrt<ValueType>(N0);
    Matrix<ValueType> M(2 * N0, 2 * N0);
    for (Long i = 0; i < N0; i++)
      for (Long j = 0; j < N0; j++) {
        M[2 * i + 0][2 * j + 0] =  cos<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
        M[2 * i + 1][2 * j + 0] =  sin<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
        M[2 * i + 0][2 * j + 1] = -sin<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
        M[2 * i + 1][2 * j + 1] =  cos<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
      }
    return M;
  }

  static Matrix<ValueType> fft_c2r(Long N0) {
    ValueType s = 1 / sqrt<ValueType>(N0);
    Long N1 = (N0 / 2 + 1);
    Matrix<ValueType> M(2 * N1, N0);
    for (Long i = 0; i < N1; i++) {
      for (Long j = 0; j < N0; j++) {
        M[2 * i + 0][j] =  2 * cos<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
        M[2 * i + 1][j] = -2 * sin<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
      }
    }
    if (N1 > 0) {
      for (Long j = 0; j < N0; j++) {
        M[0][j] = M[0][j] * (ValueType)0.5;
        M[1][j] = M[1][j] * (ValueType)0.5;
      }
    }
    if (N0 % 2 == 0) {
      for (Long j = 0; j < N0; j++) {
        M[2 * N1 - 2][j] = M[2 * N1 - 2][j] * (ValueType)0.5;
        M[2 * N1 - 1][j] = M[2 * N1 - 1][j] * (ValueType)0.5;
      }
    }
    return M;
  }

  template <class T> static void transpose(Iterator<ValueType> out, ConstIterator<ValueType> in, Long N0, Long N1) {
    const Matrix<T> M0(N0, N1, (Iterator<T>)in, false);
    Matrix<T> M1(N1, N0, (Iterator<T>)out, false);
    M1 = M0.Transpose();
  }

  FFTPlan plan;
};

static inline void FFTWInitThreads(Integer Nthreads) {
#ifdef SCTL_FFTW_THREADS
  static bool first_time = true;
  #pragma omp critical(SCTL_FFTW_INIT_THREADS)
  if (first_time) {
    fftw_init_threads();
    first_time = false;
  }
  fftw_plan_with_nthreads(Nthreads);
#endif
}

#ifdef SCTL_HAVE_FFTW
template <> class FFT<double> : public FFT_Base<double, FFT<double>> {

  typedef double ValueType;

 public:

  FFT() = default;
  FFT(const FFT&) = delete;
  FFT& operator=(const FFT&) = delete;

  ~FFT() { if (Dim(0) && Dim(1)) fftw_destroy_plan(plan); }

  void Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads = 1) {
    FFTWInitThreads(Nthreads);
    if (Dim(0) && Dim(1)) fftw_destroy_plan(plan);
    fft_type = fft_type_;
    howmany = howmany_;
    copy_input = false;
    plan = NULL;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) {
      dim_vec_[i] = dim_vec[i];
    }

    Long N0 = 0, N1 = 0;
    { // Set N0, N1
      Long N = howmany;
      for (auto ni : dim_vec) N *= ni;
      if (fft_type == FFT_Type::R2C) {
        N0 = N;
        N1 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
      } else if (fft_type == FFT_Type::C2C) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2C_INV) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2R) {
        N0 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
        N1 = N;
      } else {
        N0 = 0;
        N1 = 0;
      }
      dim[0] = N0;
      dim[1] = N1;
    }
    if (!N0 || !N1) return;
    Vector<ValueType> in(N0), out(N1);

    if (fft_type == FFT_Type::R2C) {
      plan = fftw_plan_many_dft_r2c(rank, &dim_vec_[0], howmany, &in[0], NULL, 1, N0 / howmany, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan = fftw_plan_many_dft(rank, &dim_vec_[0], howmany, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan = fftw_plan_many_dft(rank, &dim_vec_[0], howmany, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan = fftw_plan_many_dft_c2r(rank, &dim_vec_[0], howmany, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / howmany, &out[0], NULL, 1, N1 / howmany, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan = fftw_plan_many_dft_r2c(rank, &dim_vec_[0], howmany, &in[0], NULL, 1, N0 / howmany, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan = fftw_plan_many_dft(rank, &dim_vec_[0], howmany, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan = fftw_plan_many_dft(rank, &dim_vec_[0], howmany, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan = fftw_plan_many_dft_c2r(rank, &dim_vec_[0], howmany, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / howmany, &out[0], NULL, 1, N1 / howmany, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    SCTL_ASSERT(plan);
  }

  void Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const {
    Long N0 = Dim(0);
    Long N1 = Dim(1);
    if (!N0 || !N1) return;
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);
    check_align(in, out);

    ValueType s = 0;
    Vector<ValueType> tmp;
    auto in_ptr = in.begin();
    if (copy_input) { // Save input
      tmp.ReInit(N0);
      in_ptr = tmp.begin();
      tmp = in;
    }
    if (fft_type == FFT_Type::R2C) {
      s = 1 / sqrt<ValueType>(N0 / howmany);
      fftw_execute_dft_r2c(plan, (double*)&in_ptr[0], (fftw_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C) {
      s = 1 / sqrt<ValueType>(N0 / howmany * (ValueType)0.5);
      fftw_execute_dft(plan, (fftw_complex*)&in_ptr[0], (fftw_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C_INV) {
      s = 1 / sqrt<ValueType>(N1 / howmany * (ValueType)0.5);
      fftw_execute_dft(plan, (fftw_complex*)&in_ptr[0], (fftw_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2R) {
      s = 1 / sqrt<ValueType>(N1 / howmany);
      fftw_execute_dft_c2r(plan, (fftw_complex*)&in_ptr[0], (double*)&out[0]);
    }
    for (auto& x : out) x *= s;
  }

 private:

  bool copy_input;
  fftw_plan plan;
};
#endif

#ifdef SCTL_HAVE_FFTWF
template <> class FFT<float> : public FFT_Base<float, FFT<float>> {

  typedef float ValueType;

 public:

  FFT() = default;
  FFT(const FFT&) = delete;
  FFT& operator=(const FFT&) = delete;

  ~FFT() { if (Dim(0) && Dim(1)) fftwf_destroy_plan(plan); }

  void Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads = 1) {
    FFTWInitThreads(Nthreads);
    if (Dim(0) && Dim(1)) fftwf_destroy_plan(plan);
    fft_type = fft_type_;
    howmany = howmany_;
    copy_input = false;
    plan = NULL;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) {
      dim_vec_[i] = dim_vec[i];
    }

    Long N0, N1;
    { // Set N0, N1
      Long N = howmany;
      for (auto ni : dim_vec) N *= ni;
      if (fft_type == FFT_Type::R2C) {
        N0 = N;
        N1 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
      } else if (fft_type == FFT_Type::C2C) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2C_INV) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2R) {
        N0 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
        N1 = N;
      } else {
        N0 = 0;
        N1 = 0;
      }
      dim[0] = N0;
      dim[1] = N1;
    }
    if (!N0 || !N1) return;
    Vector<ValueType> in (N0), out(N1);

    if (fft_type == FFT_Type::R2C) {
      plan = fftwf_plan_many_dft_r2c(rank, &dim_vec_[0], howmany, &in[0], NULL, 1, N0 / howmany, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan = fftwf_plan_many_dft(rank, &dim_vec_[0], howmany, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan = fftwf_plan_many_dft(rank, &dim_vec_[0], howmany, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan = fftwf_plan_many_dft_c2r(rank, &dim_vec_[0], howmany, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / howmany, &out[0], NULL, 1, N1 / howmany, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan = fftwf_plan_many_dft_r2c(rank, &dim_vec_[0], howmany, &in[0], NULL, 1, N0 / howmany, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan = fftwf_plan_many_dft(rank, &dim_vec_[0], howmany, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan = fftwf_plan_many_dft(rank, &dim_vec_[0], howmany, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan = fftwf_plan_many_dft_c2r(rank, &dim_vec_[0], howmany, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / howmany, &out[0], NULL, 1, N1 / howmany, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    SCTL_ASSERT(plan);
  }

  void Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const {
    Long N0 = Dim(0);
    Long N1 = Dim(1);
    if (!N0 || !N1) return;
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);
    check_align(in, out);

    ValueType s = 0;
    Vector<ValueType> tmp;
    auto in_ptr = in.begin();
    if (copy_input) { // Save input
      tmp.ReInit(N0);
      in_ptr = tmp.begin();
      tmp = in;
    }
    if (fft_type == FFT_Type::R2C) {
      s = 1 / sqrt<ValueType>(N0 / howmany);
      fftwf_execute_dft_r2c(plan, (float*)&in_ptr[0], (fftwf_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C) {
      s = 1 / sqrt<ValueType>(N0 / howmany * (ValueType)0.5);
      fftwf_execute_dft(plan, (fftwf_complex*)&in_ptr[0], (fftwf_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C_INV) {
      s = 1 / sqrt<ValueType>(N1 / howmany * (ValueType)0.5);
      fftwf_execute_dft(plan, (fftwf_complex*)&in_ptr[0], (fftwf_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2R) {
      s = 1 / sqrt<ValueType>(N1 / howmany);
      fftwf_execute_dft_c2r(plan, (fftwf_complex*)&in_ptr[0], (float*)&out[0]);
    }
    for (auto& x : out) x *= s;
  }

 private:

  bool copy_input;
  fftwf_plan plan;
};
#endif

#ifdef SCTL_HAVE_FFTWL
template <> class FFT<long double> : public FFT_Base<long double, FFT<long double>> {

  typedef long double ValueType;

 public:

  FFT() = default;
  FFT(const FFT&) = delete;
  FFT& operator=(const FFT&) = delete;

  ~FFT() { if (Dim(0) && Dim(1)) fftwl_destroy_plan(plan); }

  void Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads = 1) {
    FFTWInitThreads(Nthreads);
    if (Dim(0) && Dim(1)) fftwl_destroy_plan(plan);
    fft_type = fft_type_;
    howmany = howmany_;
    copy_input = false;
    plan = NULL;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) dim_vec_[i] = dim_vec[i];

    Long N0, N1;
    { // Set N0, N1
      Long N = howmany;
      for (auto ni : dim_vec) N *= ni;
      if (fft_type == FFT_Type::R2C) {
        N0 = N;
        N1 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
      } else if (fft_type == FFT_Type::C2C) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2C_INV) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2R) {
        N0 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
        N1 = N;
      } else {
        N0 = 0;
        N1 = 0;
      }
      dim[0] = N0;
      dim[1] = N1;
    }
    if (!N0 || !N1) return;
    Vector<ValueType> in (N0), out(N1);

    if (fft_type == FFT_Type::R2C) {
      plan = fftwl_plan_many_dft_r2c(rank, &dim_vec_[0], howmany, &in[0], NULL, 1, N0 / howmany, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan = fftwl_plan_many_dft(rank, &dim_vec_[0], howmany, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan = fftwl_plan_many_dft(rank, &dim_vec_[0], howmany, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan = fftwl_plan_many_dft_c2r(rank, &dim_vec_[0], howmany, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / howmany, &out[0], NULL, 1, N1 / howmany, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan = fftwl_plan_many_dft_r2c(rank, &dim_vec_[0], howmany, &in[0], NULL, 1, N0 / howmany, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan = fftwl_plan_many_dft(rank, &dim_vec_[0], howmany, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan = fftwl_plan_many_dft(rank, &dim_vec_[0], howmany, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / howmany, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / howmany, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan = fftwl_plan_many_dft_c2r(rank, &dim_vec_[0], howmany, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / howmany, &out[0], NULL, 1, N1 / howmany, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    SCTL_ASSERT(plan);
  }

  void Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const {
    Long N0 = Dim(0);
    Long N1 = Dim(1);
    if (!N0 || !N1) return;
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);
    check_align(in, out);

    ValueType s = 0;
    Vector<ValueType> tmp;
    auto in_ptr = in.begin();
    if (copy_input) { // Save input
      tmp.ReInit(N0);
      in_ptr = tmp.begin();
      tmp = in;
    }
    if (fft_type == FFT_Type::R2C) {
      s = 1 / sqrt<ValueType>(N0 / howmany);
      fftwl_execute_dft_r2c(plan, (long double*)&in_ptr[0], (fftwl_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C) {
      s = 1 / sqrt<ValueType>(N0 / howmany * (ValueType)0.5);
      fftwl_execute_dft(plan, (fftwl_complex*)&in_ptr[0], (fftwl_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C_INV) {
      s = 1 / sqrt<ValueType>(N1 / howmany * (ValueType)0.5);
      fftwl_execute_dft(plan, (fftwl_complex*)&in_ptr[0], (fftwl_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2R) {
      s = 1 / sqrt<ValueType>(N1 / howmany);
      fftwl_execute_dft_c2r(plan, (fftwl_complex*)&in_ptr[0], (long double*)&out[0]);
    }
    for (auto& x : out) x *= s;
  }

 private:

  bool copy_input;
  fftwl_plan plan;
};
#endif

}  // end namespace

#endif  //_SCTL_FFT_WRAPPER_
