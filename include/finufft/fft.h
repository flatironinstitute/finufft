#ifndef FINUFFT_INCLUDE_FINUFFT_FFT_H
#define FINUFFT_INCLUDE_FINUFFT_FFT_H

#include <vector>

#ifndef FINUFFT_USE_DUCC0

//clang-format off
#include <complex>
#include <fftw3.h> // (after complex) needed so can typedef FFTW_CPX
//clang-format on
#include <mutex>

template<typename T> class Finufft_FFTW_plan {};

template<> struct Finufft_FFTW_plan<float> {
private:
  static std::mutex &mut() {
    static std::mutex mut_;
    return mut_;
  }
  fftwf_plan plan_;

public:
  Finufft_FFTW_plan() : plan_(nullptr) {
    std::lock_guard<std::mutex> lock(mut());
#ifdef _OPENMP
    static bool initialized = false;
    if (!initialized) {
      fftwf_init_threads();
      initialized = true;
    }
#endif
  }
  ~Finufft_FFTW_plan() {
    std::lock_guard<std::mutex> lock(mut());
    fftwf_destroy_plan(plan_);
  }

  void plan(const std::vector<int> &dims, size_t batchSize, std::complex<float> *ptr,
            int sign, int options, int nthreads) {
    uint64_t nf = 1;
    for (auto i : dims) nf *= i;
    std::lock_guard<std::mutex> lock(mut());
#ifdef _OPENMP
    fftwf_plan_with_nthreads(nthreads);
#endif
    plan_ = fftwf_plan_many_dft(dims.size(), dims.data(), batchSize,
                                reinterpret_cast<fftwf_complex *>(ptr), nullptr, 1, nf,
                                reinterpret_cast<fftwf_complex *>(ptr), nullptr, 1, nf,
                                sign, options);
  }
  static std::complex<float> *alloc_complex(size_t N) {
    return reinterpret_cast<std::complex<float> *>(fftwf_alloc_complex(N));
  }
  static void free(std::complex<float> *ptr) {
    fftwf_free(reinterpret_cast<fftwf_complex *>(ptr));
  }
  void execute() { fftwf_execute(plan_); }

  static void forget_wisdom() {
    std::lock_guard<std::mutex> lock(mut());
    fftwf_forget_wisdom();
  }
  static void cleanup() {
    std::lock_guard<std::mutex> lock(mut());
    fftwf_cleanup();
  }
  static void cleanup_threads() {
#ifdef _OPENMP
    std::lock_guard<std::mutex> lock(mut());
    fftwf_cleanup_threads();
#endif
  }
};

template<> struct Finufft_FFTW_plan<double> {
private:
  static std::mutex &mut() {
    static std::mutex mut_;
    return mut_;
  }
  fftw_plan plan_;

public:
  Finufft_FFTW_plan() : plan_(nullptr) {
    std::lock_guard<std::mutex> lock(mut());
#ifdef _OPENMP
    static bool initialized = false;
    if (!initialized) {
      fftw_init_threads();
      initialized = true;
    }
#endif
  }
  ~Finufft_FFTW_plan() {
    std::lock_guard<std::mutex> lock(mut());
    fftw_destroy_plan(plan_);
  }

  void plan(const std::vector<int> &dims, size_t batchSize, std::complex<double> *ptr,
            int sign, int options, int nthreads) {
    uint64_t nf = 1;
    for (auto i : dims) nf *= i;
    std::lock_guard<std::mutex> lock(mut());
#ifdef _OPENMP
    fftw_plan_with_nthreads(nthreads);
#endif
    plan_ = fftw_plan_many_dft(dims.size(), dims.data(), batchSize,
                               reinterpret_cast<fftw_complex *>(ptr), nullptr, 1, nf,
                               reinterpret_cast<fftw_complex *>(ptr), nullptr, 1, nf,
                               sign, options);
  }
  static std::complex<double> *alloc_complex(size_t N) {
    return reinterpret_cast<std::complex<double> *>(fftw_alloc_complex(N));
  }
  static void free(std::complex<double> *ptr) {
    fftw_free(reinterpret_cast<fftw_complex *>(ptr));
  }
  void execute() { fftw_execute(plan_); }

  static void forget_wisdom() {
    std::lock_guard<std::mutex> lock(mut());
    fftw_forget_wisdom();
  }
  static void cleanup() {
    std::lock_guard<std::mutex> lock(mut());
    fftw_cleanup();
  }
  static void cleanup_threads() {
#ifdef _OPENMP
    std::lock_guard<std::mutex> lock(mut());
    fftw_cleanup_threads();
#endif
  }
};

#endif

#include <finufft/defs.h>

#ifdef FINUFFT_USE_DUCC0
static inline void finufft_fft_forget_wisdom() {}
static inline void finufft_fft_cleanup() {}
static inline void finufft_fft_cleanup_threads() {}
#else
static inline void finufft_fft_forget_wisdom() {
  Finufft_FFTW_plan<FLT>::forget_wisdom();
}
static inline void finufft_fft_cleanup() { Finufft_FFTW_plan<FLT>::cleanup(); }
static inline void finufft_fft_cleanup_threads() {
  Finufft_FFTW_plan<FLT>::cleanup_threads();
}
#endif

std::vector<int> gridsize_for_fft(FINUFFT_PLAN p);
void do_fft(FINUFFT_PLAN p);

#endif // FINUFFT_INCLUDE_FINUFFT_FFT_H
