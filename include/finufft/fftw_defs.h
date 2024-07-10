// all FFTW-related private FINUFFT headers

#ifndef FFTW_DEFS_H
#define FFTW_DEFS_H

// Here we define typedefs and MACROS to switch between single and double
// precision library compilation, which need different FFTW command symbols.
// Barnett simplified via FFTWIFY, 6/7/22.

#include <complex>
#include <fftw3.h> // (after complex.h) needed so can typedef FFTW_CPX
#include <mutex>

template<typename T> class My_fftw_plan {};

template<> struct My_fftw_plan<float> {
private:
  static std::mutex fftw_lock;
  fftwf_plan plan_;

public:
  My_fftw_plan() : plan_(nullptr) {
    // Now place FFTW initialization in a lock, courtesy of OMP. Makes FINUFFT
    // thread-safe (can be called inside OMP)
    static bool did_fftw_init = false; // the only global state of FINUFFT
    std::lock_guard<std::mutex> lock(fftw_lock);
    if (!did_fftw_init) {
      init();               // setup FFTW global state; should only do once
      did_fftw_init = true; // ensure other FINUFFT threads don't clash
    }
  }
  ~My_fftw_plan() {
    std::lock_guard<std::mutex> lock(fftw_lock);
    fftwf_destroy_plan(plan_);
  }

  void plan(const std::vector<int> &dims, size_t batchSize, std::complex<float> *ptr,
            int sign, int options, int nthreads) {
    uint64_t nf = 1;
    for (auto i : dims) nf *= i;
    std::lock_guard<std::mutex> lock(fftw_lock);
    plan_with_nthreads(nthreads);
    plan_ = fftwf_plan_many_dft(dims.size(), dims.data(), batchSize, (fftwf_complex *)ptr,
                                nullptr, 1, nf, (fftwf_complex *)ptr, nullptr, 1, nf,
                                sign, options);
  }
  static std::complex<float> *alloc_complex(size_t N) {
    return reinterpret_cast<std::complex<float> *>(fftwf_alloc_complex(N));
  }
  static void free(std::complex<float> *ptr) { fftwf_free(ptr); }
  void execute() { fftwf_execute(plan_); }

  static void forget_wisdom() { fftwf_forget_wisdom(); }
  static void cleanup() { fftwf_cleanup(); }
#ifdef _OPENMP
  static void init() { fftwf_init_threads(); }
  static void plan_with_nthreads(int nthreads) { fftwf_plan_with_nthreads(nthreads); }
  static void cleanup_threads() { fftwf_cleanup_threads(); }
#else
  static void init() {}
  static void plan_with_nthreads(int /*nthreads*/) {}
  static void cleanup_threads() {}
#endif
};

template<> struct My_fftw_plan<double> {
private:
  static std::mutex fftw_lock;
  fftw_plan plan_;

public:
  My_fftw_plan() : plan_(nullptr) {
    // Now place FFTW initialization in a lock, courtesy of OMP. Makes FINUFFT
    // thread-safe (can be called inside OMP)
    static bool did_fftw_init = false; // the only global state of FINUFFT
    std::lock_guard<std::mutex> lock(fftw_lock);
    if (!did_fftw_init) {
      init();               // setup FFTW global state; should only do once
      did_fftw_init = true; // ensure other FINUFFT threads don't clash
    }
  }
  ~My_fftw_plan() {
    std::lock_guard<std::mutex> lock(fftw_lock);
    fftw_destroy_plan(plan_);
  }

  void plan(const std::vector<int> &dims, size_t batchSize, std::complex<double> *ptr,
            int sign, int options, int nthreads) {
    uint64_t nf = 1;
    for (auto i : dims) nf *= i;
    std::lock_guard<std::mutex> lock(fftw_lock);
    plan_with_nthreads(nthreads);
    plan_ = fftw_plan_many_dft(dims.size(), dims.data(), batchSize, (fftw_complex *)ptr,
                               nullptr, 1, nf, (fftw_complex *)ptr, nullptr, 1, nf, sign,
                               options);
  }
  static std::complex<double> *alloc_complex(size_t N) {
    return reinterpret_cast<std::complex<double> *>(fftw_alloc_complex(N));
  }
  static void free(std::complex<double> *ptr) { fftw_free(ptr); }
  void execute() { fftw_execute(plan_); }

  static void forget_wisdom() { fftw_forget_wisdom(); }
  static void cleanup() { fftw_cleanup(); }
#ifdef _OPENMP
  static void init() { fftw_init_threads(); }
  static void plan_with_nthreads(int nthreads) { fftw_plan_with_nthreads(nthreads); }
  static void cleanup_threads() { fftw_cleanup_threads(); }
#else
  static void init() {}
  static void plan_with_nthreads(int /*nthreads*/) {}
  static void cleanup_threads() {}
#endif
};

#endif // FFTW_DEFS_H
