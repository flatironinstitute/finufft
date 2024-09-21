#ifndef FINUFFT_INCLUDE_FINUFFT_FFT_H
#define FINUFFT_INCLUDE_FINUFFT_FFT_H

#include <vector>

#ifdef FINUFFT_USE_DUCC0
#include <complex>

template<typename T> class Finufft_FFT_plan {
public:
  Finufft_FFT_plan(void (*)(void *) = nullptr, void (*)(void *) = nullptr,
                   void * = nullptr) {}
  void plan(const std::vector<int> & /*dims*/, size_t /*batchSize*/,
            std::complex<T> * /*ptr*/, int /*sign*/, int /*options*/, int /*nthreads*/) {}
  static std::complex<T> *alloc_complex(size_t N) { return new std::complex<T>[N]; }
  static void free(std::complex<T> *ptr) { delete[] ptr; }

  static void forget_wisdom() {}
  static void cleanup() {}
  static void cleanup_threads() {}
};

#else

//clang-format off
#include <complex>
#include <fftw3.h> // (after complex) needed so can typedef FFTW_CPX
//clang-format on
#include <mutex>

template<typename T> class Finufft_FFT_plan {};

template<> struct Finufft_FFT_plan<float> {
private:
  static std::mutex &mut() {
    static std::mutex mut_;
    return mut_;
  }
  fftwf_plan plan_;

  void (*fftw_lock_fun)(void *);   // Function ptr that locks the FFTW planner
  void (*fftw_unlock_fun)(void *); // Function ptr that unlocks the FFTW planner
  void *lock_data;
  void lock() { fftw_lock_fun ? fftw_lock_fun(lock_data) : mut().lock(); }
  void unlock() { fftw_lock_fun ? fftw_unlock_fun(lock_data) : mut().unlock(); }

public:
  Finufft_FFT_plan(void (*fftw_lock_fun_)(void *)   = nullptr,
                   void (*fftw_unlock_fun_)(void *) = nullptr,
                   void *lock_data_                 = nullptr)
      : plan_(nullptr), fftw_lock_fun(fftw_lock_fun_), fftw_unlock_fun(fftw_unlock_fun_),
        lock_data(lock_data_) {
    lock();
#ifdef _OPENMP
    static bool initialized = false;
    if (!initialized) {
      fftwf_init_threads();
      initialized = true;
    }
#endif
    unlock();
  }
  ~Finufft_FFT_plan() {
    lock();
    fftwf_destroy_plan(plan_);
    unlock();
  }

  void plan(const std::vector<int> &dims, size_t batchSize, std::complex<float> *ptr,
            int sign, int options, int nthreads) {
    uint64_t nf = 1;
    for (auto i : dims) nf *= i;
    lock();
#ifdef _OPENMP
    fftwf_plan_with_nthreads(nthreads);
#endif
    plan_ = fftwf_plan_many_dft(int(dims.size()), dims.data(), int(batchSize),
                                reinterpret_cast<fftwf_complex *>(ptr), nullptr, 1,
                                int(nf), reinterpret_cast<fftwf_complex *>(ptr), nullptr,
                                1, int(nf), sign, unsigned(options));
    unlock();
  }
  static std::complex<float> *alloc_complex(size_t N) {
    return reinterpret_cast<std::complex<float> *>(fftwf_alloc_complex(N));
  }
  static void free(std::complex<float> *ptr) {
    if (ptr) fftwf_free(reinterpret_cast<fftwf_complex *>(ptr));
  }
  void execute() { fftwf_execute(plan_); }

  static void forget_wisdom() {
    //    lock();
    fftwf_forget_wisdom();
    //    unlock();
  }
  static void cleanup() {
    //    lock();
    fftwf_cleanup();
    //    unlock();
  }
  static void cleanup_threads() {
#ifdef _OPENMP
    //    lock();
    fftwf_cleanup_threads();
//    unlock();
#endif
  }
};

template<> struct Finufft_FFT_plan<double> {
private:
  static std::mutex &mut() {
    static std::mutex mut_;
    return mut_;
  }
  fftw_plan plan_;

  void (*fftw_lock_fun)(void *);   // Function ptr that locks the FFTW planner
  void (*fftw_unlock_fun)(void *); // Function ptr that unlocks the FFTW planner
  void *lock_data;
  void lock() { fftw_lock_fun ? fftw_lock_fun(lock_data) : mut().lock(); }
  void unlock() { fftw_lock_fun ? fftw_unlock_fun(lock_data) : mut().unlock(); }

public:
  Finufft_FFT_plan(void (*fftw_lock_fun_)(void *)   = nullptr,
                   void (*fftw_unlock_fun_)(void *) = nullptr,
                   void *lock_data_                 = nullptr)
      : plan_(nullptr), fftw_lock_fun(fftw_lock_fun_), fftw_unlock_fun(fftw_unlock_fun_),
        lock_data(lock_data_) {
    lock();
#ifdef _OPENMP
    static bool initialized = false;
    if (!initialized) {
      fftw_init_threads();
      initialized = true;
    }
#endif
    unlock();
  }
  ~Finufft_FFT_plan() {
    lock();
    fftw_destroy_plan(plan_);
    unlock();
  }

  void plan(const std::vector<int> &dims, size_t batchSize, std::complex<double> *ptr,
            int sign, int options, int nthreads) {
    uint64_t nf = 1;
    for (auto i : dims) nf *= i;
    lock();
#ifdef _OPENMP
    fftw_plan_with_nthreads(nthreads);
#endif
    plan_ = fftw_plan_many_dft(int(dims.size()), dims.data(), int(batchSize),
                               reinterpret_cast<fftw_complex *>(ptr), nullptr, 1, int(nf),
                               reinterpret_cast<fftw_complex *>(ptr), nullptr, 1, int(nf),
                               sign, unsigned(options));
    unlock();
  }
  static std::complex<double> *alloc_complex(size_t N) {
    return reinterpret_cast<std::complex<double> *>(fftw_alloc_complex(N));
  }
  static void free(std::complex<double> *ptr) {
    fftw_free(reinterpret_cast<fftw_complex *>(ptr));
  }
  void execute() { fftw_execute(plan_); }

  static void forget_wisdom() {
    //    lock();
    fftw_forget_wisdom();
    //    unlock();
  }
  static void cleanup() {
    //    lock();
    fftw_cleanup();
    //    unlock();
  }
  static void cleanup_threads() {
#ifdef _OPENMP
    //    lock();
    fftw_cleanup_threads();
//    unlock();
#endif
  }
};

#endif

#include <finufft/defs.h>

static inline void finufft_fft_forget_wisdom [[maybe_unused]] () {
  Finufft_FFT_plan<FLT>::forget_wisdom();
}
static inline void finufft_fft_cleanup [[maybe_unused]] () {
  Finufft_FFT_plan<FLT>::cleanup();
}
static inline void finufft_fft_cleanup_threads [[maybe_unused]] () {
  Finufft_FFT_plan<FLT>::cleanup_threads();
}

std::vector<int> gridsize_for_fft(FINUFFT_PLAN p);
void do_fft(FINUFFT_PLAN p);

#endif // FINUFFT_INCLUDE_FINUFFT_FFT_H
