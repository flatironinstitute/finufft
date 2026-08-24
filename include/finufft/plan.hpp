#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "finufft_common/common.h"
#include "finufft_errors.h"

// BIGINT/UBIGINT moved to <finufft_common/defines.h> (transitively included
// below via common.h) so that <finufft/simd.hpp> can use them without
// pulling in this whole header.

// ------------- Library-wide algorithm parameter settings ----------------

// Library version (is a string)
#define FINUFFT_VER "2.6.0-dev"

// Internal (nf1 etc) array allocation size that immediately raises error.
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
inline int MY_OMP_GET_NUM_THREADS [[maybe_unused]] () { return omp_get_num_threads(); }
inline int MY_OMP_GET_MAX_THREADS [[maybe_unused]] () { return omp_get_max_threads(); }
inline int MY_OMP_GET_THREAD_NUM [[maybe_unused]] () { return omp_get_thread_num(); }
inline void MY_OMP_SET_NUM_THREADS [[maybe_unused]] (int x) { omp_set_num_threads(x); }
using my_omp_lock_t = omp_lock_t;
inline void MY_OMP_INIT_LOCK [[maybe_unused]] (my_omp_lock_t *l) { omp_init_lock(l); }
inline void MY_OMP_DESTROY_LOCK [[maybe_unused]] (my_omp_lock_t *l) {
  omp_destroy_lock(l);
}
inline void MY_OMP_SET_LOCK [[maybe_unused]] (my_omp_lock_t *l) { omp_set_lock(l); }
inline void MY_OMP_UNSET_LOCK [[maybe_unused]] (my_omp_lock_t *l) { omp_unset_lock(l); }
#else
// non-omp safe dummy versions of omp utils...
inline int MY_OMP_GET_NUM_THREADS [[maybe_unused]] () { return 1; }
inline int MY_OMP_GET_MAX_THREADS [[maybe_unused]] () { return 1; }
inline int MY_OMP_GET_THREAD_NUM [[maybe_unused]] () { return 0; }
inline void MY_OMP_SET_NUM_THREADS [[maybe_unused]] (int) {}
struct my_omp_lock_t {};
inline void MY_OMP_INIT_LOCK [[maybe_unused]] (my_omp_lock_t *) {}
inline void MY_OMP_DESTROY_LOCK [[maybe_unused]] (my_omp_lock_t *) {}
inline void MY_OMP_SET_LOCK [[maybe_unused]] (my_omp_lock_t *) {}
inline void MY_OMP_UNSET_LOCK [[maybe_unused]] (my_omp_lock_t *) {}
#endif

// Forward declaration only. Full definition in src/fft.cpp.
template<typename T> class Finufft_FFT_plan;

// Custom deleter for unique_ptr<Finufft_FFT_plan<T>> so that the complete
// Finufft_FFT_plan type is only required in fft.cpp (where operator() is defined),
// not in every TU that instantiates FINUFFT_PLAN_T's constructor or destructor.
template<typename T> struct Finufft_FFT_plan_deleter {
  void operator()(Finufft_FFT_plan<T> *p) const; // defined in fft.cpp
};

// FFTW global cleanup utilities (defined in fft.cpp).
// FINUFFT_EXPORT_TEST: exported only when FINUFFT_BUILD_TESTS is set.
#include <finufft_common/defines.h>
FINUFFT_EXPORT_TEST void finufft_fft_forget_wisdom();
FINUFFT_EXPORT_TEST void finufft_fft_cleanup();
FINUFFT_EXPORT_TEST void finufft_fft_cleanup_threads();
#include <finufft_common/constants.h>
#include <finufft_common/spread_opts.h>
#include <finufft_opts.h>

// Tile metadata produced by bin-sort and consumed by the tiled spread/interp driver.
struct SpreadTileData {
  std::vector<BIGINT> starts;        // length ntiles+1; tile t is starts[t]..starts[t+1]
  int edge = 0;                      // fine grid points per tile edge
  std::array<BIGINT, 3> nt{1, 1, 1}; // per-axis tile counts
  std::array<BIGINT, 3> ngrid{1, 1, 1}; // per-axis fine grid points the tiles cover
  bool empty() const { return starts.empty(); }
  void clear() { *this = {}; }
};

// How the spread splits the points into subproblems. Pure function of the tile layout and
// the thread geometry (see spread_schedule in spread.hpp), so a test can check the
// schedule without running a transform.
struct SpreadSchedule {
  std::vector<UBIGINT> bounds;       // NU index breakpoints, one subproblem per gap
  UBIGINT points_per_subproblem = 0; // points one subproblem may hold
};

// A subgrid of the fine grid: its lowest corner and its extents in fine grid points.
// padded_size1 is the row stride of the buffer, always set through set_size1.
struct Subgrid {
  BIGINT off1 = 0, off2 = 0, off3 = 0;
  BIGINT padded_size1 = 1, size1 = 1, size2 = 1, size3 = 1;
  UBIGINT cells() const {
    return UBIGINT(padded_size1) * UBIGINT(size2) * UBIGINT(size3);
  }
  // Sets the first axis and the row stride together, so the two cannot disagree. tail is
  // the complex cells the innermost SIMD store writes past a row's end; a stride of a
  // whole eight cache lines reaches only eight L1 sets, so one more line goes in there.
  template<class T> void set_size1(BIGINT s, BIGINT tail) noexcept {
    constexpr BIGINT line  = 64 / BIGINT(2 * sizeof(T)); // complex cells per cache line
    constexpr BIGINT alias = 8 * line;
    const BIGINT need      = s + tail;
    size1                  = s;
    padded_size1           = need + (need % alias == 0 ? line : 0);
  }
};

template<typename TF> struct FINUFFT_PLAN_T { // the main plan class, fully C++

private:
  using TC = std::complex<TF>;

  // Type 3 rescaling/centering/phasing parameters:
  struct type3params {
    std::array<TF, 3> X, C, D, h, gam; // X=halfwid C=center D=freqcen h,gam=rescale
  };

  int spreadinterpSortedBatch(int batchSize, std::complex<TF> *fwBatch,
                              std::complex<TF> *cBatch, bool adjoint) const;
  int deconvolveBatch(int batchSize, std::complex<TF> *fkBatch, std::complex<TF> *fwBatch,
                      bool adjoint) const;
  void deconvolveshuffle1d(int dir, TF prefac, TF *fk, std::complex<TF> *fw) const;
  void deconvolveshuffle2d(int dir, TF prefac, TF *fk, std::complex<TF> *fw) const;
  void deconvolveshuffle3d(int dir, TF prefac, TF *fk, std::complex<TF> *fw) const;

  // These delete specifications just state the obvious,
  // but are here to silence compiler warnings.
  // Copy construction and assignent are already deleted implicitly
  // because of the unique_ptr member.
  FINUFFT_PLAN_T(const FINUFFT_PLAN_T &)            = delete;
  FINUFFT_PLAN_T &operator=(const FINUFFT_PLAN_T &) = delete;

  // ---------- Mutable computed state (Rust M-paradigm) ----------
  // All state that is built up across makeplan/setpts lives here.
  // Configuration set once in the constructor stays on the plan itself.
  struct M {
    // --- Spreader configuration (computed by setup_spreadinterp) ---
    TF tol{};                      // user tolerance, clamped to machine eps by spreader
    finufft_spread_opts spopts{};  // spreading kernel parameters (nspread, beta, etc.)
    int nc = 0;     // number of Horner polynomial coefficients (<= MAX_NC)
    size_t padded_ns = 0;          // SIMD-padded kernel width
    // Worst-case sizing: simd.hpp's static_asserts pin
    // max_kernel_buffer_stride<float|double> <= MAX_NSPREAD<double>, so this
    // loose bound is provably safe. Tightening would force plan.hpp to depend
    // on simd.hpp/xsimd, which the test headers don't have on their path.
    alignas(64) std::array<TF, finufft::common::MAX_NSPREAD<double> *
                                   finufft::common::MAX_NC> horner_coeffs{0};
    // piecewise Horner coefficients table (ns x nc layout)

    // --- Fine grid (computed by init_grid_kerFT_FFT or set_nhg_type3) ---
    std::array<BIGINT, 3> nfdim{1, 1, 1};  // upsampled grid dimensions
    std::array<std::vector<TF>, 3> phiHat;  // FT of spreading kernel on mode grids

    // --- NU point data (set by setpts) ---
    BIGINT nj = 0;                 // number of nonuniform source points
    BIGINT nk = 0;                 // number of nonuniform target freqs (type 3 only)
    std::array<const TF *, 3> XYZ{nullptr, nullptr, nullptr};
                                   // pointers to user's NU source coords (no alloc)
    std::vector<BIGINT> sortIndices;  // bin-sort permutation of NU points
    SpreadTileData tiles;             // tile offsets + metadata; empty unless tile-binned
    bool didSort = false;             // whether bin-sorting was applied

    // --- Type 3 workspace (set by setpts for type 3 only) ---
    std::array<const TF *, 3> STU{nullptr, nullptr, nullptr};
                                   // pointers to user's target NU-point arrays (no alloc)
    std::vector<TC> prephase;      // pre-phase factors for all input NU pts
    std::vector<TC> deconv;        // 1/kernel_FT * phase at all output NU pts
    std::array<std::vector<TF>, 3> XYZp;  // rescaled/centered source NU points (x'_j)
    std::array<std::vector<TF>, 3> STUp;  // rescaled/centered target freqs (s'_k)
    type3params t3P;               // type 3 rescaling/centering/phasing params
    std::unique_ptr<const FINUFFT_PLAN_T<TF>> innerT2plan;
                                   // inner type-2 plan used in step 2 of type 3

    // --- FFT plan (created in constructor or init_grid_kerFT_FFT) ---
    std::unique_ptr<Finufft_FFT_plan<TF>, Finufft_FFT_plan_deleter<TF>> fftPlan;
  };

  M m; // all mutable computed state lives here

public:
  int type; // transform type (Rokhlin naming): 1,2 or 3
  int dim;  // overall dimension: 1,2 or 3

private:
  int ntrans;             // how many transforms to do at once (vector or "many" mode)
  int batchSize;          // # strength vectors to group together for FFTW, etc
  int nbatch;             // how many batches done to cover all ntrans vectors
  bool upsamp_locked = false; // true if user specified upsampfac != 0, prevents auto
                              // update

public:
  std::array<BIGINT, 3> mstu; // number of modes in x,y,z directions
                              // (historical CMCL names are N1, N2, N3)

  // func for total # modes (prod of above three)...
  BIGINT N() const { return mstu[0] * mstu[1] * mstu[2]; }

  // func to return total # fine grid points...
  BIGINT nf() const { return m.nfdim[0] * m.nfdim[1] * m.nfdim[2]; }

  int fftSign; // sign in exponential for NUFFT defn, guaranteed to be +-1

public:
  finufft_opts opts; // this and spopts could be made ptrs

private:

  int execute_internal(TC *cj, TC *fk, bool adjoint = false, int ntrans_actual = -1,
                       TC *aligned_scratch = nullptr, size_t scratch_size = 0) const;
  void setup_spreadinterp(); // throws FINUFFT_ERR_EPS_TOO_SMALL if tol unachievable
  void check_sigma(); // throws FINUFFT_ERR_EPS_TOO_SMALL if sigma too low for tol
  double best_upsampfac() const; // density-aware sigma, types 1/2 (setpts)
  // complexity-based sigma for type 3 (outer spread + inner t2 cost); X,S are the
  // source/target interval half-widths over dims (see setpts).
  double best_upsampfac_type3(const TF *X, const TF *S, BIGINT nk) const;
  void precompute_horner_coeffs();
  void set_nf_type12(BIGINT ms, BIGINT *nf) const;
  void onedim_fseries_kernel(BIGINT nf, std::vector<TF> &fwkerhalf) const;
  void set_nhg_type3(int idim, TF S, TF X);
  // Compile-time-dispatched kernel method templates (NS=nspread, NC=horner degree).
  // Bodies are defined in interp.hpp and spread.hpp respectively.
  template<int NS, int NC, int NDIMS>
  void interp_subproblem_kernel(
      BIGINT off1, BIGINT off2, BIGINT off3, UBIGINT padded_size1, UBIGINT size2,
      UBIGINT size3, const TF *du, UBIGINT M, const TF *kx, const TF *ky, const TF *kz,
      const BIGINT *idx, TF *FINUFFT_RESTRICT dd) const noexcept;
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

  // Nested caller types that turn the runtime kernel width and Horner degree into
  // template arguments, one per dimension and direction. Bodies are in spread.hpp and
  // interp.hpp.
  struct SpreadSubproblem1dCaller;
  struct SpreadSubproblem2dCaller;
  struct SpreadSubproblem3dCaller;
  struct InterpSubproblem1dCaller;
  struct InterpSubproblem2dCaller;
  struct InterpSubproblem3dCaller;

  void bin_sort_singlethread(int cell, int cell_bits, SpreadTileData &tile_data_out);
  void bin_sort_multithread(int cell, int nthr, int cell_bits,
                            SpreadTileData &tile_data_out);
  // Runs of a padded subgrid that are contiguous in both it and the fine grid it wraps
  // onto: f(gi, si, n) gets the two element offsets and the run length. The wrap
  // arithmetic lives here alone, so a pass over the pair only says what it does per run.
  template<typename OnRun>
  void walk_wrapped_subgrid(const Subgrid &sub, OnRun &&on_run) const;
  void copy_wrapped_subgrid(const Subgrid &sub, const TF *data_uniform,
                            TF *FINUFFT_RESTRICT du0) const;
  template<bool thread_safe>
  void add_wrapped_subgrid(const Subgrid &sub, TF *FINUFFT_RESTRICT data_uniform,
                           const TF *du0) const;
  // The smallest subgrid holding the kernel support of all M points.
  Subgrid get_subgrid(UBIGINT M, const TF *kx, const TF *ky, const TF *kz) const;

  void spreadcheck() const;
  void indexSort();
  // The tiled spread policy, read off plan members: the doublings of the sort cell per
  // tile edge, and how the sorted points split into subproblems. The rules are pure
  // functions in spread.hpp, so a test can drive them on a synthetic tile layout.
  int tile_doublings(int cell) const noexcept;
  SpreadSchedule make_schedule(int nthr, int batchSize) const;
  // One entry point per dimension and direction, so the per-dimension TUs
  // (spreadinterp_1d/2d/3d.cpp) instantiate one dimension each. The signatures are
  // uniform: ky and kz are unread below their dimension and may be null there.
  void spread_subproblem_1d(const Subgrid &sub, TF *FINUFFT_RESTRICT du, UBIGINT M,
                            const TF *kx, const TF *ky, const TF *kz,
                            const TF *dd) const noexcept;
  void spread_subproblem_2d(const Subgrid &sub, TF *FINUFFT_RESTRICT du, UBIGINT M,
                            const TF *kx, const TF *ky, const TF *kz,
                            const TF *dd) const noexcept;
  void spread_subproblem_3d(const Subgrid &sub, TF *FINUFFT_RESTRICT du, UBIGINT M,
                            const TF *kx, const TF *ky, const TF *kz,
                            const TF *dd) const noexcept;
  // idx[j] is where point j of this subproblem sat before the sort, which is where its
  // strength goes back to in dd.
  void interp_subproblem_1d(const Subgrid &sub, const TF *du, UBIGINT M, const TF *kx,
                            const TF *ky, const TF *kz, const BIGINT *idx,
                            TF *dd) const noexcept;
  void interp_subproblem_2d(const Subgrid &sub, const TF *du, UBIGINT M, const TF *kx,
                            const TF *ky, const TF *kz, const BIGINT *idx,
                            TF *dd) const noexcept;
  void interp_subproblem_3d(const Subgrid &sub, const TF *du, UBIGINT M, const TF *kx,
                            const TF *ky, const TF *kz, const BIGINT *idx,
                            TF *dd) const noexcept;
  // batchSize>1 folds the batch loop into the subproblem loop, so all nthr threads get
  // assigned (vector, subprob) pairs out of the batchSize*nb of them. The per-vector
  // strides are the plan's own 2*nf() and 2*nj.
  int spreadSorted(TF *FINUFFT_RESTRICT data_uniform, const TF *data_nonuniform,
                   int batchSize = 1) const;
  int interpSorted(TF *FINUFFT_RESTRICT data_uniform,
                   TF *FINUFFT_RESTRICT data_nonuniform, int batchSize = 1) const;
  TF evaluate_kernel_runtime(TF x) const;
  std::vector<int> gridsize_for_fft() const;
  void do_fft(TC *fwBatch, int ntrans_actual, bool adjoint) const;

  // Analytic PSWF self-FT 1D kernel FT evaluator, one Horner eval per target
  // (used by type-3 setpts). See finufft::kernel::pswf_selfft_params for the
  // identity. Nested class: accesses plan's private members via friendship.
  class Kernel_onedim_FT {
    const FINUFFT_PLAN_T *plan_ptr = nullptr;
    TF grid_scale = 0;
    TF prefac = 0;

  public:
    Kernel_onedim_FT(const FINUFFT_PLAN_T &plan);
    FINUFFT_ALWAYS_INLINE TF operator()(TF k) const {
      return prefac * plan_ptr->evaluate_kernel_runtime(k * grid_scale);
    }
  };

  // Helper to initialize spreader, phiHat (Fourier series), and FFT plan.
  // Used by constructor (when upsampfac given) and setpts (when upsampfac deferred).
  void init_grid_kerFT_FFT();

  // Allocates fftPlan (needs complete Finufft_FFT_plan type); defined in fft.cpp.
  void create_fft_plan();

public:
  // FINUFFT_EXPORT_TEST: a test drives the plan directly, so a shared build must export
  // the three members it calls out of line.
  FINUFFT_EXPORT_TEST FINUFFT_PLAN_T(int type, int dim, const BIGINT *n_modes, int iflag,
                                     int ntrans, TF tol, const finufft_opts *opts);
  FINUFFT_EXPORT_TEST ~FINUFFT_PLAN_T(); // defined in src/fft.cpp, where the FFT plan is
                                         // complete

  // Remaining actions (not create/delete) in guru interface are now methods...
  FINUFFT_EXPORT_TEST int setpts(BIGINT nj, const TF *xj, const TF *yj, const TF *zj,
                                 BIGINT nk, const TF *s, const TF *t, const TF *u);

  // which spread path the last setpts prepared: empty tiles mean the chunk cut
  const SpreadTileData &tile_data() const { return m.tiles; }
  // the permutation the sort produced, so a test can read a tile's points back
  const std::vector<BIGINT> &sort_indices() const { return m.sortIndices; }
  int nspread() const { return m.spopts.nspread; }
  // whether the last setpts sorted, and the fine grid the sort rule is stated over
  bool sorted() const { return m.didSort; }
  UBIGINT grid_size() const {
    return UBIGINT(m.nfdim[0]) * UBIGINT(m.nfdim[1]) * UBIGINT(m.nfdim[2]);
  }

  int execute(TC *cj, TC *fk) const { return execute_internal(cj, fk, false); }
  int execute_adjoint(TC *cj, TC *fk) const { return execute_internal(cj, fk, true); }

  // accessors for reading the internal state of the plan
  BIGINT Nj() const { return m.nj; }
  BIGINT Nk() const { return m.nk; }
  TF Tol() const { return m.tol; }
  int Ntrans() const { return ntrans; }
  const std::array<const TF *, 3> &getSTU() const { return m.STU; }
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
  FINUFFT_DIAGNOSTIC_PUSH
  FINUFFT_DISABLE_WARNING_DEPRECATED
  // sphinx tag (don't remove): @defopts_start
  o->modeord          = 0;
  o->spreadinterponly = 0;

  o->debug        = 0;
  o->spread_debug = 0;
  o->showwarn     = 1;

  o->nthreads           = 0;
  o->fftw               = FINUFFT_FFT_DEFAULT; // FFTW_ESTIMATE for FFTW; -1 for DUCC0
  o->spread_sort        = 2;
  o->spread_kerevalmeth = 1;                   // deprecated, retained for ABI
  o->spread_kerpad      = 1;                   // deprecated, retained for ABI
  o->upsampfac          = 0.0;
  o->spread_thread = 0; // deprecated, retained for ABI
  o->maxbatchsize       = 0;
  o->spread_nthr_atomic = -1;
  o->spread_max_sp_size  = 0; // deprecated, retained for ABI
  o->spread_kerformula  = 0;
  o->allow_eps_too_small = 0;
  o->fftw_lock_fun      = nullptr;
  o->fftw_unlock_fun    = nullptr;
  o->fftw_lock_data     = nullptr;
  // sphinx tag (don't remove): @defopts_end
  FINUFFT_DIAGNOSTIC_POP
}
template<typename TF>
int finufft_makeplan_t(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans,
                       TF tol, FINUFFT_PLAN_T<TF> **pp, const finufft_opts *opts);
