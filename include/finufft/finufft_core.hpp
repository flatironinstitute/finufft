#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "finufft_common/common.h"
#include "finufft_errors.h"

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday.
using BIGINT  = int64_t;
using UBIGINT = uint64_t;

// ------------- Library-wide algorithm parameter settings ----------------

// Library version (is a string)
#define FINUFFT_VER "2.5.0"

// Internal (nf1 etc) array allocation size that immediately raises error.
// (Note: next235 takes 1s for 1e11, so it is also to prevent hang here.)
// Increase this if you need >10TB (!) RAM...
inline constexpr BIGINT MAX_NF = BIGINT(1e12);

// Maximum allowed number M of NU points; useful to catch incorrectly cast int32
// values for M = nj (also nk in type 3)...
inline constexpr BIGINT MAX_NU_PTS = BIGINT(1e14);

// ----- OpenMP macros which also work when omp not present -----
// Allows compile-time switch off of openmp, so compilation without any openmp
// is done (Note: _OPENMP is automatically set by -fopenmp compile flag)
#ifdef _OPENMP
#include <omp.h>
// point to actual omp utils
static inline int MY_OMP_GET_NUM_THREADS [[maybe_unused]] () {
  return omp_get_num_threads();
}
static inline int MY_OMP_GET_MAX_THREADS [[maybe_unused]] () {
  return omp_get_max_threads();
}
static inline int MY_OMP_GET_THREAD_NUM [[maybe_unused]] () {
  return omp_get_thread_num();
}
static inline void MY_OMP_SET_NUM_THREADS [[maybe_unused]] (int x) {
  omp_set_num_threads(x);
}
#else
// non-omp safe dummy versions of omp utils...
static inline int MY_OMP_GET_NUM_THREADS [[maybe_unused]] () { return 1; }
static inline int MY_OMP_GET_MAX_THREADS [[maybe_unused]] () { return 1; }
static inline int MY_OMP_GET_THREAD_NUM [[maybe_unused]] () { return 0; }
static inline void MY_OMP_SET_NUM_THREADS [[maybe_unused]] (int) {}
#endif

#include <finufft/fft.hpp> // (must come after complex.h)
#include <finufft_common/constants.h>
#include <finufft_common/spread_opts.h>
#include <finufft_opts.h>
#include <finufft_common/spread_opts.h>

// group together a bunch of type 3 rescaling/centering/phasing parameters:
template<typename T> struct type3params {
  std::array<T, 3> X, C, D, h, gam; // x dim: X=halfwid C=center D=freqcen h,gam=rescale
};

template<typename TF> struct FINUFFT_PLAN_T { // the main plan class, fully C++

private:
  using TC = std::complex<TF>;

  int spreadinterpSortedBatch(int batchSize, std::complex<TF> *fwBatch,
                              std::complex<TF> *cBatch, bool adjoint) const;
  int deconvolveBatch(int batchSize, std::complex<TF> *fkBatch, std::complex<TF> *fwBatch,
                      bool adjoint) const;
  void deconvolveshuffle1d(int dir, TF prefac, BIGINT ms, TF *fk,
                           std::complex<TF> *fw) const;
  void deconvolveshuffle2d(int dir, TF prefac, BIGINT ms, BIGINT mt, TF *fk,
                           std::complex<TF> *fw) const;
  void deconvolveshuffle3d(int dir, TF prefac, BIGINT ms, BIGINT mt, BIGINT mu, TF *fk,
                           std::complex<TF> *fw) const;

  // These delete specifications just state the obvious,
  // but are here to silence compiler warnings.
  // Copy construction and assignent are already deleted implicitly
  // because of the unique_ptr member.
  FINUFFT_PLAN_T(const FINUFFT_PLAN_T &)            = delete;
  FINUFFT_PLAN_T &operator=(const FINUFFT_PLAN_T &) = delete;

public:
  int type; // transform type (Rokhlin naming): 1,2 or 3
  int dim;  // overall dimension: 1,2 or 3

private:
  int ntrans;             // how many transforms to do at once (vector or "many" mode)
  BIGINT nj;              // num of NU pts in type 1,2 (for type 3, num input x pts)
  BIGINT nk;              // number of NU freq pts (type 3 only)
  TF tol;                 // relative user tolerance
  int batchSize;          // # strength vectors to group together for FFTW, etc
  int nbatch;             // how many batches done to cover all ntrans vectors

  int nc             = 0; // number of Horner coefficients used for ES kernel (<= MAX_NC)
  int padded_ns      = 0; // SIMD-padded kernel width, set by precompute_horner_coeffs()
  bool upsamp_locked = false; // true if user specified upsampfac != 0, prevents auto
                              // update

public:
  std::array<BIGINT, 3> mstu; // number of modes in x,y,z directions
                              // (historical CMCL names are N1, N2, N3)

  // func for total # modes (prod of above three)...
  BIGINT N() const { return mstu[0] * mstu[1] * mstu[2]; }

  std::array<BIGINT, 3> nfdim{1, 1, 1}; // internal fine grid size in x,y,z directions
  // func to return total # fine grid points...
  BIGINT nf() const { return nfdim[0] * nfdim[1] * nfdim[2]; }

  int fftSign; // sign in exponential for NUFFT defn, guaranteed to be +-1

private:
  std::array<std::vector<TF>, 3> phiHat; // FT of kernel in t1,2, on x,y,z-axis mode grid

  std::vector<BIGINT> sortIndices; // precomputed NU pt permutation, speeds spread/interp
  bool didSort;                    // whether binsorting used (false: identity perm used)

  // for t1,2: ptr to user-supplied NU pts (no new allocs).
  // for t3: will become ptr to internally allocated "primed" (scaled) Xp, Yp, Zp vecs
  std::array<const TF *, 3> XYZ = {nullptr, nullptr, nullptr};

  // type 3 specific
  std::array<const TF *, 3> STU = {nullptr, nullptr, nullptr}; // ptrs to user's target
                                                               // NU-point arrays (no new
                                                               // allocs)
  std::vector<TC> prephase; // pre-phase, for all input NU pts
  std::vector<TC> deconv;   // reciprocal of kernel FT, phase, all output NU pts
  std::array<std::vector<TF>, 3> XYZp; // internal primed NU points (x'_j, etc)
  std::array<std::vector<TF>, 3> STUp; // internal primed targs (s'_k, etc)
  type3params<TF> t3P; // groups together type 3 shift, scale, phase, parameters
  std::unique_ptr<const FINUFFT_PLAN_T<TF>> innerT2plan; // ptr used for type 2 in step 2
                                                         // of type 3

  // other internal structs
  std::unique_ptr<Finufft_FFT_plan<TF>, Finufft_FFT_plan_deleter<TF>> fftPlan;

  // store piecewise Horner coeffs for ns intervals of kernel: ns x nc table
  alignas(64) std::array<TF, finufft::common::MAX_NSPREAD *
                                 finufft::common::MAX_NC> horner_coeffs{0};

public:
  finufft_opts opts; // this and spopts could be made ptrs

private:
  finufft_spread_opts spopts;

  int execute_internal(TC *cj, TC *fk, bool adjoint = false, int ntrans_actual = -1,
                       TC *aligned_scratch = nullptr, size_t scratch_size = 0) const;
  int setup_spreadinterp();
  void precompute_horner_coeffs();
  int set_nf_type12(BIGINT ms, BIGINT *nf) const;
  void onedim_fseries_kernel(BIGINT nf, std::vector<TF> &fwkerhalf) const;
  void set_nhg_type3(int idim, TF S, TF X);
  // Compile-time-dispatched kernel method templates (NS=nspread, NC=horner degree).
  // Bodies are defined in detail/interp.hpp and detail/spread.hpp respectively.
  template<int NS, int NC>
  int interpSorted_kernel(TF *data_uniform, TF *data_nonuniform) const;
  template<int NS, int NC>
  void spread_subproblem_1d_kernel(BIGINT off1, UBIGINT size1, TF *FINUFFT_RESTRICT du,
                                   UBIGINT M, const TF *kx, const TF *dd) const noexcept;
  template<int NS, int NC>
  void spread_subproblem_2d_kernel(BIGINT off1, BIGINT off2, UBIGINT size1, UBIGINT size2,
                                   TF *FINUFFT_RESTRICT du, UBIGINT M, const TF *kx,
                                   const TF *ky, const TF *dd) const noexcept;
  template<int NS, int NC>
  void spread_subproblem_3d_kernel(BIGINT off1, BIGINT off2, BIGINT off3, UBIGINT size1,
                                   UBIGINT size2, UBIGINT size3, TF *FINUFFT_RESTRICT du,
                                   UBIGINT M, const TF *kx, const TF *ky, const TF *kz,
                                   const TF *dd) const noexcept;

  // Nested caller types used to dispatch to compile-time ns/nc kernel specialisations.
  // Bodies are defined in detail/spread.hpp and detail/interp.hpp respectively.
  struct SpreadSubproblem1dCaller;
  struct SpreadSubproblem2dCaller;
  struct SpreadSubproblem3dCaller;
  struct InterpSortedCaller;

  void bin_sort_singlethread(double bin_size_x, double bin_size_y, double bin_size_z);
  void bin_sort_multithread(double bin_size_x, double bin_size_y, double bin_size_z,
                            int nthr);
  template<bool thread_safe>
  void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
                           UBIGINT padded_size1, UBIGINT size1, UBIGINT size2, UBIGINT size3,
                           TF *FINUFFT_RESTRICT data_uniform, const TF *du0) const;
  void get_subgrid(BIGINT &offset1, BIGINT &offset2, BIGINT &offset3, BIGINT &padded_size1,
                   BIGINT &size1, BIGINT &size2, BIGINT &size3, UBIGINT M, const TF *kx,
                   const TF *ky, const TF *kz) const;

  void indexSort();
  void spread_subproblem_1d(BIGINT off1, UBIGINT size1, TF *du, UBIGINT M, TF *kx,
                            TF *dd) const noexcept;
  void spread_subproblem_2d(BIGINT off1, BIGINT off2, UBIGINT size1, UBIGINT size2,
                            TF *FINUFFT_RESTRICT du, UBIGINT M, const TF *kx,
                            const TF *ky, const TF *dd) const noexcept;
  void spread_subproblem_3d(BIGINT off1, BIGINT off2, BIGINT off3, UBIGINT size1,
                            UBIGINT size2, UBIGINT size3, TF *du, UBIGINT M, TF *kx,
                            TF *ky, TF *kz, TF *dd) const noexcept;
  int spreadSorted(TF *FINUFFT_RESTRICT data_uniform, const TF *data_nonuniform) const;
  int interpSorted(TF *FINUFFT_RESTRICT data_uniform,
                   TF *FINUFFT_RESTRICT data_nonuniform) const;
  int spreadinterpSorted(TF *data_uniform, TF *data_nonuniform, bool adjoint) const;
  TF evaluate_kernel_runtime(TF x) const;
  std::vector<int> gridsize_for_fft() const;
  void do_fft(TC *fwBatch, int ntrans_actual, bool adjoint) const;

  // Precomputed quadrature-based 1D kernel FT evaluator (used by type-3 setpts).
  // Nested class: has access to plan's private members via implicit friendship.
  class Kernel_onedim_FT {
    std::vector<TF> z, f;

  public:
    Kernel_onedim_FT(const FINUFFT_PLAN_T &plan);
    FINUFFT_ALWAYS_INLINE TF operator()(TF k) const {
      TF x = 0;
      for (size_t n = 0; n < z.size(); ++n)
        x += f[n] * 2 * std::cos(k * z[n]); // pos & neg freq pair
      return x;
    }
  };

  // Helper to initialize spreader, phiHat (Fourier series), and FFT plan.
  // Used by constructor (when upsampfac given) and setpts (when upsampfac deferred).
  int init_grid_kerFT_FFT();

  // Allocates fftPlan (needs complete Finufft_FFT_plan type); defined in fft.cpp.
  void create_fft_plan();

public:
  FINUFFT_PLAN_T(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans, TF tol,
                 const finufft_opts *opts, int &ier);
  ~FINUFFT_PLAN_T(); // defined in src/fft.cpp where Finufft_FFT_plan is complete

  // Remaining actions (not create/delete) in guru interface are now methods...
  int setpts(BIGINT nj, const TF *xj, const TF *yj, const TF *zj, BIGINT nk, const TF *s,
             const TF *t, const TF *u);

  int execute(TC *cj, TC *fk) const { return execute_internal(cj, fk, false); }
  int execute_adjoint(TC *cj, TC *fk) const { return execute_internal(cj, fk, true); }

  // accessors for reading the internal state of the plan
  BIGINT Nj() const { return nj; }
  BIGINT Nk() const { return nk; }
  TF Tol() const { return tol; }
  int Ntrans() const { return ntrans; }
  const std::array<const TF *, 3> &getSTU() const { return STU; }
};

inline void finufft_default_opts_t(finufft_opts *o)
// Sets default nufft opts (referenced by all language interfaces too).
// See finufft_opts.h for meanings.
// This was created to avoid uncertainty about C++11 style static initialization
// when called from MEX, but now is generally used. Barnett 10/30/17 onwards.
// Discussion (Marco Barbone: 5.8.2024): These are user-facing.
// The various options could be macros to follow c standard library conventions.
// Question: would these be enums? Ans: no, let's keep ints/doubles for now.

// For FFW=DUCC, opts.fftw=-1 is the default to be more informative than 0
// (which coincides with the code FFTW_MEASURE; see fftw3.h).

// Sphinx sucks the below code block into the web docs, hence keep it clean...
{
  // sphinx tag (don't remove): @defopts_start
  o->modeord          = 0;
  o->spreadinterponly = 0;

  o->debug        = 0;
  o->spread_debug = 0;
  o->showwarn     = 1;

  o->nthreads = 0;
  o->fftw     = FINUFFT_FFT_DEFAULT; // FFTW_ESTIMATE (=64) for FFTW; -1 for DUCC0
  o->spread_sort        = 2;
  o->spread_kerevalmeth = 1; // deprecated
  o->spread_kerpad      = 1; // deprecated
  o->upsampfac          = 0.0;
  o->spread_thread      = 0;
  o->maxbatchsize       = 0;
  o->spread_nthr_atomic = -1;
  o->spread_max_sp_size = 0;
  o->spread_kerformula  = 0;
  o->fftw_lock_fun      = nullptr;
  o->fftw_unlock_fun    = nullptr;
  o->fftw_lock_data     = nullptr;
  // sphinx tag (don't remove): @defopts_end
}
template<typename TF>
int finufft_makeplan_t(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans,
                       TF tol, FINUFFT_PLAN_T<TF> **pp, const finufft_opts *opts);
