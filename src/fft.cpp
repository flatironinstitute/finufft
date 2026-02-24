#ifdef FINUFFT_USE_DUCC0
#include "ducc0/fft/fftnd_impl.h" // only needed for do_fft body; not in any header
#endif

// --- Inlined from (now-deleted) detail/fft.hpp ---
// Full definition of Finufft_FFT_plan.
// DUCC0 path: no-op stub (ducc0 headers only needed in do_fft body above).
// FFTW path:  full float/double specialisations (includes fftw3.h here).

#ifdef FINUFFT_USE_DUCC0
#include <complex>

template<typename T> class Finufft_FFT_plan {
public:
  [[maybe_unused]] Finufft_FFT_plan(void (*)(void *) = nullptr,
                                    void (*)(void *) = nullptr, void * = nullptr) {}
  // deleting these operations to be consistent with the FFTW plans (see below)
  Finufft_FFT_plan(const Finufft_FFT_plan &)            = delete;
  Finufft_FFT_plan &operator=(const Finufft_FFT_plan &) = delete;
  [[maybe_unused]] void plan(const std::vector<int> & /*dims*/, size_t /*batchSize*/,
                             std::complex<T> * /*ptr*/, int /*sign*/, int /*options*/,
                             int /*nthreads*/) {}

  [[maybe_unused]] static void forget_wisdom() {}
  [[maybe_unused]] static void cleanup() {}
  [[maybe_unused]] static void cleanup_threads() {}
};

#else // FFTW path

//clang-format off
#include <complex>
#include <fftw3.h> // (after complex) needed so can typedef FFTW_CPX
//clang-format on
#include <mutex>
#include <vector>

template<typename T> class Finufft_FFT_plan {};

template<> class Finufft_FFT_plan<float> {
private:
  static std::mutex &mut() {
    static std::mutex mut_;
    return mut_;
  }
  fftwf_plan plan_, plan_adj_;

  void (*fftw_lock_fun)(void *);   // Function ptr that locks the FFTW planner
  void (*fftw_unlock_fun)(void *); // Function ptr that unlocks the FFTW planner
  void *lock_data;
  void lock() { fftw_lock_fun ? fftw_lock_fun(lock_data) : mut().lock(); }
  void unlock() { fftw_lock_fun ? fftw_unlock_fun(lock_data) : mut().unlock(); }

public:
  [[maybe_unused]] Finufft_FFT_plan(void (*fftw_lock_fun_)(void *)   = nullptr,
                                    void (*fftw_unlock_fun_)(void *) = nullptr,
                                    void *lock_data_                 = nullptr)
      : plan_(nullptr), plan_adj_(nullptr), fftw_lock_fun(fftw_lock_fun_),
        fftw_unlock_fun(fftw_unlock_fun_), lock_data(lock_data_) {
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
  // we have raw pointers in the object (the FFTW plans).
  // If we allow copying those, we end up destroying the plans multiple times.
  Finufft_FFT_plan(const Finufft_FFT_plan &) = delete;
  [[maybe_unused]] ~Finufft_FFT_plan() {
    lock();
    fftwf_destroy_plan(plan_);
    fftwf_destroy_plan(plan_adj_);
    unlock();
  }
  Finufft_FFT_plan &operator=(const Finufft_FFT_plan &) = delete;

  void plan
      [[maybe_unused]] (const std::vector<int> &dims, size_t batchSize,
                        std::complex<float> *ptr, int sign, int options, int nthreads) {
    uint64_t nf = 1;
    for (auto i : dims) nf *= i;
    lock();
    // Destroy existing plans before creating new ones (handles re-planning)
    if (plan_) {
      fftwf_destroy_plan(plan_);
      plan_ = nullptr;
    }
    if (plan_adj_) {
      fftwf_destroy_plan(plan_adj_);
      plan_adj_ = nullptr;
    }
#ifdef _OPENMP
    fftwf_plan_with_nthreads(nthreads);
#endif
    plan_     = fftwf_plan_many_dft(int(dims.size()), dims.data(), int(batchSize),
                                    reinterpret_cast<fftwf_complex *>(ptr), nullptr, 1,
                                    int(nf), reinterpret_cast<fftwf_complex *>(ptr), nullptr,
                                    1, int(nf), sign, unsigned(options));
    plan_adj_ = fftwf_plan_many_dft(int(dims.size()), dims.data(), int(batchSize),
                                    reinterpret_cast<fftwf_complex *>(ptr), nullptr, 1,
                                    int(nf), reinterpret_cast<fftwf_complex *>(ptr),
                                    nullptr, 1, int(nf), -sign, unsigned(options));
    unlock();
  }
  void execute [[maybe_unused]] (std::complex<float> *data) const {
    fftwf_execute_dft(plan_, reinterpret_cast<fftwf_complex *>(data),
                      reinterpret_cast<fftwf_complex *>(data));
  }
  void execute_adjoint [[maybe_unused]] (std::complex<float> *data) const {
    fftwf_execute_dft(plan_adj_, reinterpret_cast<fftwf_complex *>(data),
                      reinterpret_cast<fftwf_complex *>(data));
  }

  static void forget_wisdom [[maybe_unused]] () { fftwf_forget_wisdom(); }
  static void cleanup [[maybe_unused]] () { fftwf_cleanup(); }
  static void cleanup_threads [[maybe_unused]] () {
#ifdef _OPENMP
    fftwf_cleanup_threads();
#endif
  }
};

template<> class Finufft_FFT_plan<double> {
private:
  static std::mutex &mut() {
    static std::mutex mut_;
    return mut_;
  }
  fftw_plan plan_, plan_adj_;

  void (*fftw_lock_fun)(void *);   // Function ptr that locks the FFTW planner
  void (*fftw_unlock_fun)(void *); // Function ptr that unlocks the FFTW planner
  void *lock_data;
  void lock() { fftw_lock_fun ? fftw_lock_fun(lock_data) : mut().lock(); }
  void unlock() { fftw_lock_fun ? fftw_unlock_fun(lock_data) : mut().unlock(); }

public:
  [[maybe_unused]] Finufft_FFT_plan(void (*fftw_lock_fun_)(void *)   = nullptr,
                                    void (*fftw_unlock_fun_)(void *) = nullptr,
                                    void *lock_data_                 = nullptr)
      : plan_(nullptr), plan_adj_(nullptr), fftw_lock_fun(fftw_lock_fun_),
        fftw_unlock_fun(fftw_unlock_fun_), lock_data(lock_data_) {
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
    fftw_destroy_plan(plan_adj_);
    unlock();
  }
  Finufft_FFT_plan &operator=(const Finufft_FFT_plan &) = delete;

  void plan
      [[maybe_unused]] (const std::vector<int> &dims, size_t batchSize,
                        std::complex<double> *ptr, int sign, int options, int nthreads) {
    uint64_t nf = 1;
    for (auto i : dims) nf *= i;
    lock();
    // Destroy existing plans before creating new ones (handles re-planning)
    if (plan_) {
      fftw_destroy_plan(plan_);
      plan_ = nullptr;
    }
    if (plan_adj_) {
      fftw_destroy_plan(plan_adj_);
      plan_adj_ = nullptr;
    }
#ifdef _OPENMP
    fftw_plan_with_nthreads(nthreads);
#endif
    plan_     = fftw_plan_many_dft(int(dims.size()), dims.data(), int(batchSize),
                                   reinterpret_cast<fftw_complex *>(ptr), nullptr, 1, int(nf),
                                   reinterpret_cast<fftw_complex *>(ptr), nullptr, 1, int(nf),
                                   sign, unsigned(options));
    plan_adj_ = fftw_plan_many_dft(int(dims.size()), dims.data(), int(batchSize),
                                   reinterpret_cast<fftw_complex *>(ptr), nullptr, 1,
                                   int(nf), reinterpret_cast<fftw_complex *>(ptr),
                                   nullptr, 1, int(nf), -sign, unsigned(options));
    unlock();
  }
  void execute [[maybe_unused]] (std::complex<double> *data) const {
    fftw_execute_dft(plan_, reinterpret_cast<fftw_complex *>(data),
                     reinterpret_cast<fftw_complex *>(data));
  }
  void execute_adjoint [[maybe_unused]] (std::complex<double> *data) const {
    fftw_execute_dft(plan_adj_, reinterpret_cast<fftw_complex *>(data),
                     reinterpret_cast<fftw_complex *>(data));
  }

  static void forget_wisdom [[maybe_unused]] () { fftw_forget_wisdom(); }
  static void cleanup [[maybe_unused]] () { fftw_cleanup(); }
  static void cleanup_threads [[maybe_unused]] () {
#ifdef _OPENMP
    fftw_cleanup_threads();
#endif
  }
};

#endif // FINUFFT_USE_DUCC0

#include <finufft/finufft_core.hpp> // FINUFFT_PLAN_T (pulls in fft.hpp forward decl)
#include <finufft/finufft_utils.hpp> // CNTime
#include <finufft/xsimd.hpp>         // aligned_allocator
#include <algorithm>                  // std::min (for DUCC0 path)

using namespace std;

// --- Finufft_FFT_plan_deleter (defined here where Finufft_FFT_plan is complete) ---
template<typename T>
void Finufft_FFT_plan_deleter<T>::operator()(Finufft_FFT_plan<T> *p) const {
  delete p;
}
template struct Finufft_FFT_plan_deleter<float>;
template struct Finufft_FFT_plan_deleter<double>;

// --- FINUFFT_FFT_DEFAULT ---
#ifdef FINUFFT_USE_DUCC0
extern "C" const int FINUFFT_FFT_DEFAULT = -1;
#else
extern "C" const int FINUFFT_FFT_DEFAULT = FFTW_ESTIMATE;
#endif

// --- FINUFFT_PLAN_T destructor (needs complete Finufft_FFT_plan type) ---
template<typename TF> FINUFFT_PLAN_T<TF>::~FINUFFT_PLAN_T() {}
template FINUFFT_PLAN_T<float>::~FINUFFT_PLAN_T();
template FINUFFT_PLAN_T<double>::~FINUFFT_PLAN_T();

// --- gridsize_for_fft ---
template<typename TF>
std::vector<int> FINUFFT_PLAN_T<TF>::gridsize_for_fft() const
// Returns grid dims in fftw_plan_many_dft / ducc0 order.
// 2/24/26 Barbone: converted from free function.
{
  if (dim == 1) return {(int)nfdim[0]};
  if (dim == 2) return {(int)nfdim[1], (int)nfdim[0]};
  return {(int)nfdim[2], (int)nfdim[1], (int)nfdim[0]};
}
template std::vector<int> FINUFFT_PLAN_T<float>::gridsize_for_fft() const;
template std::vector<int> FINUFFT_PLAN_T<double>::gridsize_for_fft() const;

// --- do_fft ---
template<typename TF>
void FINUFFT_PLAN_T<TF>::do_fft(TC *fwBatch, int ntrans_actual [[maybe_unused]],
                                 bool adjoint) const
// Execute FFT on fwBatch (in-place, batchSize transforms).
// FFTW: ntrans_actual ignored (plan already sized to batchSize).
// DUCC0: used for partial FFTs.
// 2/24/26 Barbone: converted from free function.
{
#ifdef FINUFFT_USE_DUCC0
  size_t nthreads = min<size_t>(MY_OMP_GET_MAX_THREADS(), opts.nthreads);
  const auto ns   = gridsize_for_fft();
  vector<size_t> arrdims, axes;
  // ntrans_actual may be smaller than batchSize, which we can use
  // to our advantage with ducc FFT.
  arrdims.push_back(size_t(ntrans_actual));
  arrdims.push_back(size_t(ns[0]));
  axes.push_back(1);
  if (dim >= 2) {
    arrdims.push_back(size_t(ns[1]));
    axes.push_back(2);
  }
  if (dim >= 3) {
    arrdims.push_back(size_t(ns[2]));
    axes.push_back(3);
  }
  // in DUCC0, "forward=true/false" corresponds to an FFT exponent sign of -/+.
  // Analogous to FFTW, transforms are not normalized in either direction.
  bool forward   = (fftSign < 0) != adjoint;
  bool spreading = (type == 1) != adjoint;
  ducc0::vfmav<std::complex<TF>> data(fwBatch, arrdims);
#ifdef FINUFFT_NO_DUCC0_TWEAKS
  ducc0::c2c(data, data, axes, forward, TF(1), nthreads);
#else
  /* When spreading, only the low-frequency parts of the output fine grid are
     going to be used, and when interpolating, the high frequency parts of the
     input fine grid are zero by definition. This can be used to reduce the
     total FFT work for 2D and 3D NUFFTs. One of the FFT axes always has to be
     transformed fully (that's why there is no savings for 1D NUFFTs), for the
     second axis we need to do (roughly) a fraction of 1/oversampling_factor
     of all 1D FFTs, and for the last remaining axis the factor is
     1/oversampling_factor^2. */
  if (dim == 1)        // 1D: no chance for FFT shortcuts
    ducc0::c2c(data, data, axes, forward, TF(1), nthreads);
  else if (dim == 2) { // 2D: do partial FFTs
    if (mstu[0] < 2)   // something is weird, do standard FFT
      ducc0::c2c(data, data, axes, forward, TF(1), nthreads);
    else {
      size_t y_lo = size_t((mstu[0] + 1) / 2);
      size_t y_hi = size_t(ns[1] - mstu[0] / 2);
      // the next line is analogous to the Python statement "sub1 = data[:, :, :y_lo]"
      auto sub1 = ducc0::subarray(data, {{}, {}, {0, y_lo}});
      // the next line is analogous to the Python statement "sub2 = data[:, :, y_hi:]"
      auto sub2 = ducc0::subarray(data, {{}, {}, {y_hi, ducc0::MAXIDX}});
      if (spreading) // spreading, not all parts of the output array are needed
        // do axis 2 in full
        ducc0::c2c(data, data, {2}, forward, TF(1), nthreads);
      // do only parts of axis 1
      ducc0::c2c(sub1, sub1, {1}, forward, TF(1), nthreads);
      ducc0::c2c(sub2, sub2, {1}, forward, TF(1), nthreads);
      if (!spreading) // interpolation, parts of the input array are zero
        // do axis 2 in full
        ducc0::c2c(data, data, {2}, forward, TF(1), nthreads);
    }
  } else {                              // 3D
    if ((mstu[0] < 2) || (mstu[1] < 2)) // something is weird, do standard FFT
      ducc0::c2c(data, data, axes, forward, TF(1), nthreads);
    else {
      size_t z_lo = size_t((mstu[0] + 1) / 2);
      size_t z_hi = size_t(ns[2] - mstu[0] / 2);
      size_t y_lo = size_t((mstu[1] + 1) / 2);
      size_t y_hi = size_t(ns[1] - mstu[1] / 2);
      auto sub1   = ducc0::subarray(data, {{}, {}, {}, {0, z_lo}});
      auto sub2   = ducc0::subarray(data, {{}, {}, {}, {z_hi, ducc0::MAXIDX}});
      auto sub3   = ducc0::subarray(sub1, {{}, {}, {0, y_lo}, {}});
      auto sub4   = ducc0::subarray(sub1, {{}, {}, {y_hi, ducc0::MAXIDX}, {}});
      auto sub5   = ducc0::subarray(sub2, {{}, {}, {0, y_lo}, {}});
      auto sub6   = ducc0::subarray(sub2, {{}, {}, {y_hi, ducc0::MAXIDX}, {}});
      if (spreading) { // spreading, not all parts of the output array are needed
        // do axis 3 in full
        ducc0::c2c(data, data, {3}, forward, TF(1), nthreads);
        // do only parts of axis 2
        ducc0::c2c(sub1, sub1, {2}, forward, TF(1), nthreads);
        ducc0::c2c(sub2, sub2, {2}, forward, TF(1), nthreads);
      }
      // do even smaller parts of axis 1
      ducc0::c2c(sub3, sub3, {1}, forward, TF(1), nthreads);
      ducc0::c2c(sub4, sub4, {1}, forward, TF(1), nthreads);
      ducc0::c2c(sub5, sub5, {1}, forward, TF(1), nthreads);
      ducc0::c2c(sub6, sub6, {1}, forward, TF(1), nthreads);
      if (!spreading) { // interpolation, parts of the input array are zero
        // do only parts of axis 2
        ducc0::c2c(sub1, sub1, {2}, forward, TF(1), nthreads);
        ducc0::c2c(sub2, sub2, {2}, forward, TF(1), nthreads);
        // do axis 3 in full
        ducc0::c2c(data, data, {3}, forward, TF(1), nthreads);
      }
    }
  }
#endif
#else // FFTW path: ntrans_actual ignored (plan already sized to batchSize)
  if (adjoint)
    fftPlan->execute_adjoint(fwBatch);
  else
    fftPlan->execute(fwBatch);
#endif
}
template void FINUFFT_PLAN_T<float>::do_fft(std::complex<float> *, int, bool) const;
template void FINUFFT_PLAN_T<double>::do_fft(std::complex<double> *, int, bool) const;

// --- create_fft_plan ---
// Allocates the fftPlan unique_ptr; needs complete Finufft_FFT_plan type.
// Called from the constructor in detail/makeplan.hpp.
template<typename TF> void FINUFFT_PLAN_T<TF>::create_fft_plan() {
  fftPlan.reset(new Finufft_FFT_plan<TF>(
      opts.fftw_lock_fun, opts.fftw_unlock_fun, opts.fftw_lock_data));
}
template void FINUFFT_PLAN_T<float>::create_fft_plan();
template void FINUFFT_PLAN_T<double>::create_fft_plan();

// --- init_grid_kerFT_FFT ---
// Helper to initialize spreader, phiHat (Fourier series), and FFT plan.
// Used by constructor (when upsampfac given) and setpts (when upsampfac deferred).
// Returns 0 on success, or an error code if set_nf_type12 or alloc fails.
// Moved from detail/makeplan.hpp to fft.cpp so the complete Finufft_FFT_plan
// type is available without a detail/ header.
template<typename TF> int FINUFFT_PLAN_T<TF>::init_grid_kerFT_FFT() {
  using namespace finufft::utils;
  CNTime timer{};
  spopts.spread_direction = type;
  constexpr TF EPSILON    = std::numeric_limits<TF>::epsilon();

  if (opts.spreadinterponly) { // (unusual case of no NUFFT, just report)
    // spreadinterp grid will simply be the user's "mode" grid...
    for (int idim = 0; idim < dim; ++idim) nfdim[idim] = mstu[idim];

    if (opts.debug) { // "long long" here is to avoid warnings with printf...
      printf("[%s] %dd spreadinterponly(dir=%d): (ms,mt,mu)=(%lld,%lld,%lld)"
             "\n               ntrans=%d nthr=%d batchSize=%d kernel width ns=%d",
             __func__, dim, type, (long long)mstu[0], (long long)mstu[1],
             (long long)mstu[2], ntrans, opts.nthreads, batchSize, spopts.nspread);
      if (batchSize == 1) // spread_thread has no effect in this case
        printf("\n");
      else
        printf(" spread_thread=%d\n", opts.spread_thread);
    }

  } else {               // ..... usual NUFFT: eval Fourier series, alloc workspace .....

    if (opts.showwarn) { // user warn round-off error (due to prob condition #)...
      for (int idim = 0; idim < dim; ++idim)
        if (EPSILON * mstu[idim] > 1.0)
          fprintf(stderr,
                  "%s warning: rounding err (due to cond # of prob) eps_mach*N%d = %.3g "
                  "> 1 !\n",
                  __func__, idim, (double)(EPSILON * mstu[idim]));
    }

    // determine fine grid sizes, sanity check, then alloc...
    for (int idim = 0; idim < dim; ++idim) {
      int nfier = set_nf_type12(mstu[idim], &nfdim[idim]);
      if (nfier) return nfier;                  // nf too big; we're done
      phiHat[idim].resize(nfdim[idim] / 2 + 1); // alloc fseries
    }

    if (opts.debug) { // "long long" here is to avoid warnings with printf...
      printf("[%s] %dd%d: (ms,mt,mu)=(%lld,%lld,%lld) "
             "(nf1,nf2,nf3)=(%lld,%lld,%lld)\n               ntrans=%d nthr=%d "
             "batchSize=%d ",
             __func__, dim, type, (long long)mstu[0], (long long)mstu[1],
             (long long)mstu[2], (long long)nfdim[0], (long long)nfdim[1],
             (long long)nfdim[2], ntrans, opts.nthreads, batchSize);
      if (batchSize == 1) // spread_thread has no effect in this case
        printf("\n");
      else
        printf(" spread_thread=%d\n", opts.spread_thread);
    }

    // STEP 0: get Fourier coeffs of spreading kernel along each fine grid dim
    timer.restart();
    for (int idim = 0; idim < dim; ++idim)
      onedim_fseries_kernel(nfdim[idim], phiHat[idim]);
    if (opts.debug)
      printf("[%s] kernel fser (ns=%d):\t\t%.3g s\n", __func__, spopts.nspread,
             timer.elapsedsec());

    if (nf() * batchSize > MAX_NF) {
      fprintf(stderr,
              "[%s] fwBatch would be bigger than MAX_NF, not attempting memory "
              "allocation!\n",
              __func__);
      return FINUFFT_ERR_MAXNALLOC;
    }

    timer.restart(); // plan the FFTW (to act in-place on the workspace fwBatch)
    int nthr_fft  = opts.nthreads;
    const auto ns = gridsize_for_fft();
    std::vector<TC, xsimd::aligned_allocator<TC, 64>> fwBatch(nf() * batchSize);
    fftPlan->plan(ns, batchSize, fwBatch.data(), fftSign, opts.fftw, nthr_fft);
    if (opts.debug)
      printf("[%s] FFT plan (mode %d, nthr=%d):\t%.3g s\n", __func__, opts.fftw, nthr_fft,
             timer.elapsedsec());
  }
  return 0;
}
template int FINUFFT_PLAN_T<float>::init_grid_kerFT_FFT();
template int FINUFFT_PLAN_T<double>::init_grid_kerFT_FFT();

// --- fftw global cleanup utilities ---
void finufft_fft_forget_wisdom() {
  Finufft_FFT_plan<float>::forget_wisdom();
  Finufft_FFT_plan<double>::forget_wisdom();
}
void finufft_fft_cleanup() {
  Finufft_FFT_plan<float>::cleanup();
  Finufft_FFT_plan<double>::cleanup();
}
void finufft_fft_cleanup_threads() {
  Finufft_FFT_plan<float>::cleanup_threads();
  Finufft_FFT_plan<double>::cleanup_threads();
}
