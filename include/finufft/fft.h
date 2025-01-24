#ifndef FINUFFT_INCLUDE_FINUFFT_FFT_H
#define FINUFFT_INCLUDE_FINUFFT_FFT_H

#include <vector>

#ifdef FINUFFT_USE_DUCC0
#include <complex>

template<typename T> class Finufft_FFT_plan {
public:
  [[maybe_unused]] Finufft_FFT_plan(void (*)(void *) = nullptr,
                                    void (*)(void *) = nullptr, void * = nullptr) {}
  // deleting these operations to be consistent with the FFTW plans (seel below)
  Finufft_FFT_plan(const Finufft_FFT_plan &)            = delete;
  Finufft_FFT_plan &operator=(const Finufft_FFT_plan &) = delete;
  [[maybe_unused]] void plan(const std::vector<int> & /*dims*/, size_t /*batchSize*/,
                             std::complex<T> * /*ptr*/, int /*sign*/, int /*options*/,
                             int /*nthreads*/) {}

  [[maybe_unused]] static void forget_wisdom() {}
  [[maybe_unused]] static void cleanup() {}
  [[maybe_unused]] static void cleanup_threads() {}
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
  [[maybe_unused]] Finufft_FFT_plan(void (*fftw_lock_fun_)(void *)   = nullptr,
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
  // we have raw pointers in the object (the FFTW plan).
  // If we allow copying those, we end up destroying the plans multiple times.
  Finufft_FFT_plan(const Finufft_FFT_plan &) = delete;
  [[maybe_unused]] ~Finufft_FFT_plan() {
    lock();
    fftwf_destroy_plan(plan_);
    unlock();
  }
  Finufft_FFT_plan &operator=(const Finufft_FFT_plan &) = delete;

  void plan
      [[maybe_unused]] (const std::vector<int> &dims, size_t batchSize,
                        std::complex<float> *ptr, int sign, int options, int nthreads) {
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
  void execute [[maybe_unused]] () { fftwf_execute(plan_); }

  static void forget_wisdom [[maybe_unused]] () { fftwf_forget_wisdom(); }
  static void cleanup [[maybe_unused]] () { fftwf_cleanup(); }
  static void cleanup_threads [[maybe_unused]] () {
#ifdef _OPENMP
    fftwf_cleanup_threads();
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
  [[maybe_unused]] Finufft_FFT_plan(void (*fftw_lock_fun_)(void *)   = nullptr,
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
  Finufft_FFT_plan(const Finufft_FFT_plan &) = delete;
  [[maybe_unused]] ~Finufft_FFT_plan() {
    lock();
    fftw_destroy_plan(plan_);
    unlock();
  }
  Finufft_FFT_plan &operator=(const Finufft_FFT_plan &) = delete;

  void plan
      [[maybe_unused]] (const std::vector<int> &dims, size_t batchSize,
                        std::complex<double> *ptr, int sign, int options, int nthreads) {
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
  void execute [[maybe_unused]] () { fftw_execute(plan_); }

  static void forget_wisdom [[maybe_unused]] () { fftw_forget_wisdom(); }
  static void cleanup [[maybe_unused]] () { fftw_cleanup(); }
  static void cleanup_threads [[maybe_unused]] () {
#ifdef _OPENMP
    fftw_cleanup_threads();
#endif
  }
};

#endif

#include <finufft/finufft_core.h>

static inline void finufft_fft_forget_wisdom [[maybe_unused]] () {
  Finufft_FFT_plan<float>::forget_wisdom();
  Finufft_FFT_plan<double>::forget_wisdom();
}
static inline void finufft_fft_cleanup [[maybe_unused]] () {
  Finufft_FFT_plan<float>::cleanup();
  Finufft_FFT_plan<double>::cleanup();
}
static inline void finufft_fft_cleanup_threads [[maybe_unused]] () {
  Finufft_FFT_plan<float>::cleanup_threads();
  Finufft_FFT_plan<double>::cleanup_threads();
}
template<typename TF> struct FINUFFT_PLAN_T;
template<typename TF> std::vector<int> gridsize_for_fft(FINUFFT_PLAN_T<TF> *p);
template<typename TF> void do_fft(FINUFFT_PLAN_T<TF> *p);

#endif // FINUFFT_INCLUDE_FINUFFT_FFT_H
