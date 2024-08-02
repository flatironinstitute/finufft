// Spreading/interpolating module within FINUFFT. Uses precision-switching
// macros for FLT, CPX, etc.

#include <finufft/defs.h>
#include <finufft/spreadinterp.h>
#include <finufft/utils.h>
#include <finufft/utils_precindep.h>

#include "ker_horner_allw_loop_constexpr.h"
#include "ker_lowupsampfac_horner_allw_loop_constexpr.h"

#include <xsimd/xsimd.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;
using namespace finufft::utils; // access to timer

namespace finufft::spreadinterp {

namespace { // anonymous namespace for internal structs equivalent to declaring everything
            // static
struct zip_low;
struct zip_hi;
template<unsigned cap> struct reverse_index;
template<unsigned cap> struct shuffle_index;
struct select_even;
struct select_odd;
// forward declaration to clean up the code and be able to use this everywhere in the file
template<class T, uint8_t N, uint8_t K = N> static constexpr auto BestSIMDHelper();
template<class T, uint8_t N> constexpr auto GetPaddedSIMDWidth();
template<class T, uint8_t N>
using PaddedSIMD = typename xsimd::make_sized_batch<T, GetPaddedSIMDWidth<T, N>()>::type;
template<class T> uint8_t get_padding(uint8_t ns);
template<class T, uint8_t ns> constexpr auto get_padding();
template<class T, uint8_t N>
using BestSIMD = typename decltype(BestSIMDHelper<T, N, xsimd::batch<T>::size>())::type;
template<class T, uint8_t N = 1> constexpr uint8_t min_simd_width();
template<class T, uint8_t N> constexpr auto find_optimal_simd_width();
template<class T, class V = typename T::value_type, std::size_t N = T::size>
constexpr auto initialize_complex_register(V a, V b) noexcept;
template<class arch_t>
constexpr auto zip_low_index =
    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t, zip_low>();
template<class arch_t>
constexpr auto zip_hi_index =
    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t, zip_hi>();
template<class arch_t>
constexpr auto select_even_mask =
    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t, select_even>();
template<class arch_t>
constexpr auto select_odd_mask =
    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t, select_odd>();
template<typename T, std::size_t N, std::size_t M, std::size_t PaddedM>
constexpr std::array<std::array<T, PaddedM>, N> pad_2D_array_with_zeros(
    const std::array<std::array<T, M>, N> &input) noexcept;
template<typename T> FINUFFT_ALWAYS_INLINE auto xsimd_to_array(const T &vec) noexcept;

FINUFFT_NEVER_INLINE
void print_subgrid_info(int ndims, BIGINT offset1, BIGINT offset2, BIGINT offset3,
                        UBIGINT padded_size1, UBIGINT size1, UBIGINT size2, UBIGINT size3,
                        UBIGINT M0);
} // namespace
// declarations of purely internal functions... (thus need not be in .h)
template<uint8_t ns, uint8_t kerevalmeth, class T,
         class simd_type = xsimd::make_sized_batch_t<T, find_optimal_simd_width<T, ns>()>,
         typename... V>
static FINUFFT_ALWAYS_INLINE auto ker_eval(FLT *FINUFFT_RESTRICT ker,
                                           const finufft_spread_opts &opts,
                                           const V... elems) noexcept;
static FINUFFT_ALWAYS_INLINE FLT fold_rescale(FLT x, UBIGINT N) noexcept;
template<class simd_type>
FINUFFT_ALWAYS_INLINE static simd_type fold_rescale(const simd_type &x,
                                                    UBIGINT N) noexcept;
static FINUFFT_ALWAYS_INLINE void set_kernel_args(
    FLT *args, FLT x, const finufft_spread_opts &opts) noexcept;
static FINUFFT_ALWAYS_INLINE void evaluate_kernel_vector(
    FLT *ker, FLT *args, const finufft_spread_opts &opts) noexcept;
template<uint8_t w, uint8_t upsampfact,
         class simd_type =
             xsimd::make_sized_batch_t<FLT, find_optimal_simd_width<FLT, w>()>> // aka ns
static FINUFFT_ALWAYS_INLINE void eval_kernel_vec_Horner(
    FLT *FINUFFT_RESTRICT ker, FLT x, const finufft_spread_opts &opts) noexcept;
template<uint8_t ns, class simd_type = PaddedSIMD<FLT, 2 * ns>>
static void interp_line(FLT *FINUFFT_RESTRICT target, const FLT *du, const FLT *ker,
                        BIGINT i1, UBIGINT N1);
template<uint8_t ns, class simd_type = PaddedSIMD<FLT, 2 * ns>>
static void interp_square(FLT *FINUFFT_RESTRICT target, const FLT *du, const FLT *ker1,
                          const FLT *ker2, BIGINT i1, BIGINT i2, UBIGINT N1, UBIGINT N2);
template<uint8_t ns, class simd_type = PaddedSIMD<FLT, 2 * ns>>
static void interp_cube(FLT *FINUFFT_RESTRICT target, const FLT *du, const FLT *ker1,
                        const FLT *ker2, const FLT *ker3, BIGINT i1, BIGINT i2, BIGINT i3,
                        UBIGINT N1, UBIGINT N2, UBIGINT N3);
static void spread_subproblem_1d(BIGINT off1, UBIGINT size1, FLT *du0, UBIGINT M0,
                                 FLT *kx0, FLT *dd0,
                                 const finufft_spread_opts &opts) noexcept;
static void spread_subproblem_2d(BIGINT off1, BIGINT off2, UBIGINT size1, UBIGINT size2,
                                 FLT *FINUFFT_RESTRICT du, UBIGINT M, const FLT *kx,
                                 const FLT *ky, const FLT *dd,
                                 const finufft_spread_opts &opts) noexcept;
static void spread_subproblem_3d(BIGINT off1, BIGINT off2, BIGINT off3, UBIGINT size1,
                                 UBIGINT size2, UBIGINT size3, FLT *du0, UBIGINT M0,
                                 FLT *kx0, FLT *ky0, FLT *kz0, FLT *dd0,
                                 const finufft_spread_opts &opts) noexcept;
template<bool thread_safe>
static void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
                                UBIGINT padded_size1, UBIGINT size1, UBIGINT size2,
                                UBIGINT size3, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                                FLT *FINUFFT_RESTRICT data_uniform, const FLT *du0);
static void bin_sort_singlethread(BIGINT *ret, UBIGINT M, const FLT *kx, const FLT *ky,
                                  const FLT *kz, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                                  double bin_size_x, double bin_size_y, double bin_size_z,
                                  int debug);
void bin_sort_multithread(BIGINT *ret, UBIGINT M, FLT *kx, FLT *ky, FLT *kz, UBIGINT N1,
                          UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y,
                          double bin_size_z, int debug, int nthr);
static void get_subgrid(BIGINT &offset1, BIGINT &offset2, BIGINT &offset3,
                        BIGINT &padded_size1, BIGINT &size1, BIGINT &size2, BIGINT &size3,
                        UBIGINT M0, FLT *kx0, FLT *ky0, FLT *kz0, int ns, int ndims);

// ==========================================================================
int spreadinterp(UBIGINT N1, UBIGINT N2, UBIGINT N3, FLT *data_uniform, UBIGINT M,
                 FLT *kx, FLT *ky, FLT *kz, FLT *data_nonuniform,
                 const finufft_spread_opts &opts)
/* ------------Spreader/interpolator for 1, 2, or 3 dimensions --------------
   If opts.spread_direction=1, evaluate, in the 1D case,

                         N1-1
   data_nonuniform[j] =  SUM phi(kx[j] - n) data_uniform[n],   for j=0...M-1
                         n=0

   If opts.spread_direction=2, evaluate its transpose, in the 1D case,

                      M-1
   data_uniform[n] =  SUM phi(kx[j] - n) data_nonuniform[j],   for n=0...N1-1
                      j=0

   In each case phi is the spreading kernel, which has support
   [-opts.nspread/2,opts.nspread/2]. In 2D or 3D, the generalization with
   product of 1D kernels is performed.
   For 1D set N2=N3=1; for 2D set N3=1; for 3D set N1,N2,N3>1.

   Notes:
   No particular normalization of the spreading kernel is assumed.
   Uniform (U) points are centered at coords
   [0,1,...,N1-1] in 1D, analogously in 2D and 3D. They are stored in x
   fastest, y medium, z slowest ordering, up to however many
   dimensions are relevant; note that this is Fortran-style ordering for an
   array f(x,y,z), but C style for f[z][y][x]. This is to match the Fortran
   interface of the original CMCL libraries.
   Non-uniform (NU) points kx,ky,kz are real, and may lie in the central three
   periods in each coordinate (these are folded into the central period).
   The finufft_spread_opts struct must have been set up already by calling setup_kernel.
   It is assumed that 2*opts.nspread < min(N1,N2,N3), so that the kernel
   only ever wraps once when falls below 0 or off the top of a uniform grid
   dimension.

   Inputs:
   N1,N2,N3 - grid sizes in x (fastest), y (medium), z (slowest) respectively.
              If N2==1, 1D spreading is done. If N3==1, 2D spreading.
          Otherwise, 3D.
   M - number of NU pts.
   kx, ky, kz - length-M real arrays of NU point coordinates (only kx read in
                1D, only kx and ky read in 2D).

        These should lie in the box -pi<=kx<=pi. Points outside this domain are also
        correctly folded back into this domain.
   opts - spread/interp options struct, documented in ../include/finufft_spread_opts.h

   Inputs/Outputs:
   data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
   data_nonuniform - input strengths of the sources (dir=1)
                     OR output values at targets (dir=2)
   Returned value:
   0 indicates success; other values have meanings in ../docs/error.rst, with
   following modifications:
      3 : one or more non-trivial box dimensions is less than 2.nspread.
      5 : failed allocate sort indices

   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
   error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17. parallel sort 2/9/18
   No separate subprob indices in t-1 2/11/18.
   sort_threads (since for M<<N, multithread sort slower than single) 3/27/18
   kereval, kerpad 4/24/18
   Melody Shih split into 3 routines: check, sort, spread. Jun 2018, making
   this routine just a caller to them. Name change, Barnett 7/27/18
   Tidy, Barnett 5/20/20. Tidy doc, Barnett 10/22/20.
*/
{
  int ier = spreadcheck(N1, N2, N3, M, kx, ky, kz, opts);
  if (ier) return ier;
  BIGINT *sort_indices = (BIGINT *)malloc(sizeof(BIGINT) * M);
  if (!sort_indices) {
    fprintf(stderr, "%s failed to allocate sort_indices!\n", __func__);
    return FINUFFT_ERR_SPREAD_ALLOC;
  }
  int did_sort = indexSort(sort_indices, N1, N2, N3, M, kx, ky, kz, opts);
  spreadinterpSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz,
                     data_nonuniform, opts, did_sort);
  free(sort_indices);
  return 0;
}

static constexpr uint8_t ndims_from_Ns(const UBIGINT N1, const UBIGINT N2,
                                       const UBIGINT N3)
/* rule for getting number of spreading dimensions from the list of Ns per dim.
   Split out, Barnett 7/26/18
*/
{
  return 1 + (N2 > 1) + (N3 > 1);
}

int spreadcheck(UBIGINT N1, UBIGINT N2, UBIGINT N3, UBIGINT M, FLT *kx, FLT *ky, FLT *kz,
                const finufft_spread_opts &opts)
/* This does just the input checking and reporting for the spreader.
   See spreadinterp() for input arguments and meaning of returned value.
   Split out by Melody Shih, Jun 2018. Finiteness chk Barnett 7/30/18.
   Marco Barbone 5.8.24 removed bounds check as new foldrescale is not limited to
   [-3pi,3pi)
*/
{
  // INPUT CHECKING & REPORTING .... cuboid not too small for spreading?
  int minN = 2 * opts.nspread;
  if (N1 < minN || (N2 > 1 && N2 < minN) || (N3 > 1 && N3 < minN)) {
    fprintf(stderr,
            "%s error: one or more non-trivial box dims is less than 2.nspread!\n",
            __func__);
    return FINUFFT_ERR_SPREAD_BOX_SMALL;
  }
  if (opts.spread_direction != 1 && opts.spread_direction != 2) {
    fprintf(stderr, "%s error: opts.spread_direction must be 1 or 2!\n", __func__);
    return FINUFFT_ERR_SPREAD_DIR;
  }
  return 0;
}

int indexSort(BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3, UBIGINT M,
              FLT *kx, FLT *ky, FLT *kz, const finufft_spread_opts &opts)
/* This makes a decision whether or not to sort the NU pts (influenced by
   opts.sort), and if yes, calls either single- or multi-threaded bin sort,
   writing reordered index list to sort_indices. If decided not to sort, the
   identity permutation is written to sort_indices.
   The permutation is designed to make RAM access close to contiguous, to
   speed up spreading/interpolation, in the case of disordered NU points.

   Inputs:
    M        - number of input NU points.
    kx,ky,kz - length-M arrays of real coords of NU pts. Domain is [-pi, pi),
                points outside are folded in.
               (only kz used in 1D, only kx and ky used in 2D.)
    N1,N2,N3 - integer sizes of overall box (set N2=N3=1 for 1D, N3=1 for 2D).
               1 = x (fastest), 2 = y (medium), 3 = z (slowest).
    opts     - spreading options struct, see ../include/finufft_spread_opts.h
   Outputs:
    sort_indices - a good permutation of NU points. (User must preallocate
                   to length M.) Ie, kx[sort_indices[j]], j=0,..,M-1, is a good
                   ordering for the x-coords of NU pts, etc.
    returned value - whether a sort was done (1) or not (0).

   Barnett 2017; split out by Melody Shih, Jun 2018. Barnett nthr logic 2024.
*/
{
  CNTime timer{};
  uint8_t ndims = ndims_from_Ns(N1, N2, N3);
  auto N        = N1 * N2 * N3; // U grid (periodic box) sizes

  // heuristic binning box size for U grid... affects performance:
  double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;
  // put in heuristics based on cache sizes (only useful for single-thread) ?

  int better_to_sort =
      !(ndims == 1 && (opts.spread_direction == 2 || (M > 1000 * N1))); // 1D small-N or
                                                                        // dir=2 case:
                                                                        // don't sort

  timer.start();                           // if needed, sort all the NU pts...
  int did_sort = 0;
  auto maxnthr = MY_OMP_GET_MAX_THREADS(); // used if both below opts default
  if (opts.nthreads > 0)
    maxnthr = opts.nthreads;               // user nthreads overrides, without limit
  if (opts.sort_threads > 0)
    maxnthr = opts.sort_threads;           // high-priority override, also no limit
  // At this point: maxnthr = the max threads sorting could use
  // (we don't print warning here, since: no showwarn in spread_opts, and finufft
  // already warned about it. spreadinterp-only advanced users will miss a warning)
  if (opts.sort == 1 || (opts.sort == 2 && better_to_sort)) {
    // store a good permutation ordering of all NU pts (dim=1,2 or 3)
    int sort_debug = (opts.debug >= 2); // show timing output?
    int sort_nthr  = opts.sort_threads; // 0, or user max # threads for sort
#ifndef _OPENMP
    sort_nthr = 1;                      // if single-threaded lib, override user
#endif
    if (sort_nthr == 0) // multithreaded auto choice: when N>>M, one thread is better!
      sort_nthr = (10 * M > N) ? maxnthr : 1; // heuristic
    if (sort_nthr == 1)
      bin_sort_singlethread(sort_indices, M, kx, ky, kz, N1, N2, N3, bin_size_x,
                            bin_size_y, bin_size_z, sort_debug);
    else // sort_nthr>1, user fixes # threads (>=2)
      bin_sort_multithread(sort_indices, M, kx, ky, kz, N1, N2, N3, bin_size_x,
                           bin_size_y, bin_size_z, sort_debug, sort_nthr);
    if (opts.debug)
      printf("\tsorted (%d threads):\t%.3g s\n", sort_nthr, timer.elapsedsec());
    did_sort = 1;
  } else {
#pragma omp parallel for num_threads(maxnthr) schedule(static, 1000000)
    for (BIGINT i = 0; i < M; i++) // here omp helps xeon, hinders i7
      sort_indices[i] = i;         // the identity permutation
    if (opts.debug)
      printf("\tnot sorted (sort=%d): \t%.3g s\n", (int)opts.sort, timer.elapsedsec());
  }
  return did_sort;
}

int spreadinterpSorted(const BIGINT *sort_indices, const UBIGINT N1, const UBIGINT N2,
                       const UBIGINT N3, FLT *data_uniform, const UBIGINT M,
                       FLT *FINUFFT_RESTRICT kx, FLT *FINUFFT_RESTRICT ky,
                       FLT *FINUFFT_RESTRICT kz, FLT *FINUFFT_RESTRICT data_nonuniform,
                       const finufft_spread_opts &opts, int did_sort)
/* Logic to select the main spreading (dir=1) vs interpolation (dir=2) routine.
   See spreadinterp() above for inputs arguments and definitions.
   Return value should always be 0 (no error reporting).
   Split out by Melody Shih, Jun 2018; renamed Barnett 5/20/20.
*/
{
  if (opts.spread_direction == 1) // ========= direction 1 (spreading) =======
    spreadSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform,
                 opts, did_sort);

  else // ================= direction 2 (interpolation) ===========
    interpSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform,
                 opts);

  return 0;
}

// --------------------------------------------------------------------------
int spreadSorted(const BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                 FLT *FINUFFT_RESTRICT data_uniform, UBIGINT M, FLT *FINUFFT_RESTRICT kx,
                 FLT *FINUFFT_RESTRICT ky, FLT *FINUFFT_RESTRICT kz,
                 const FLT *data_nonuniform, const finufft_spread_opts &opts,
                 int did_sort)
// Spread NU pts in sorted order to a uniform grid. See spreadinterp() for doc.
{
  CNTime timer{};
  const auto ndims = ndims_from_Ns(N1, N2, N3);
  const auto N     = N1 * N2 * N3;             // output array size
  const auto ns    = opts.nspread;             // abbrev. for w, kernel width
  auto nthr        = MY_OMP_GET_MAX_THREADS(); // guess # threads to use to spread
  if (opts.nthreads > 0) nthr = opts.nthreads; // user override, now without limit
#ifndef _OPENMP
  nthr = 1;                                    // single-threaded lib must override user
#endif
  if (opts.debug)
    printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d\n", ndims,
           (long long)M, (long long)N1, (long long)N2, (long long)N3, nthr);
  timer.start();
  std::fill(data_uniform, data_uniform + 2 * N, 0.0); // zero the output array
  if (opts.debug) printf("\tzero output array\t%.3g s\n", timer.elapsedsec());
  if (M == 0)                                         // no NU pts, we're done
    return 0;

  auto spread_single = (nthr == 1) || (M * 100 < N); // low-density heuristic?
  spread_single      = false;                        // for now
  timer.start();
  if (spread_single) { // ------- Basic single-core t1 spreading ------
    for (UBIGINT j = 0; j < M; j++) {
      // *** todo, not urgent
      // ... (question is: will the index wrapping per NU pt slow it down?)
    }
    if (opts.debug) printf("\tt1 simple spreading:\t%.3g s\n", timer.elapsedsec());
  } else { // ------- Fancy multi-core blocked t1 spreading ----
           // Splits sorted inds (jfm's advanced2), could double RAM.
    // choose nb (# subprobs) via used nthreads:
    auto nb = std::min((UBIGINT)nthr, M); // simply split one subprob per thr...
    if (nb * (BIGINT)opts.max_subproblem_size < M) { // ...or more subprobs to cap size
      nb = 1 + (M - 1) / opts.max_subproblem_size;   // int div does
                                                     // ceil(M/opts.max_subproblem_size)
      if (opts.debug)
        printf("\tcapping subproblem sizes to max of %d\n", opts.max_subproblem_size);
    }
    if (M * 1000 < N) { // low-density heuristic: one thread per NU pt!
      nb = M;
      if (opts.debug) printf("\tusing low-density speed rescue nb=M...\n");
    }
    if (!did_sort && nthr == 1) {
      nb = 1;
      if (opts.debug) printf("\tunsorted nthr=1: forcing single subproblem...\n");
    }
    if (opts.debug && nthr > opts.atomic_threshold)
      printf("\tnthr big: switching add_wrapped OMP from critical to atomic (!)\n");

    std::vector<UBIGINT> brk(nb + 1); // NU index breakpoints defining nb subproblems
    for (int p = 0; p <= nb; ++p) brk[p] = (M * p + nb - 1) / nb;

#pragma omp parallel num_threads(nthr)
    {
      // local copies of NU pts and data for each subproblem
      std::vector<FLT> kx0{}, ky0{}, kz0{}, dd0{}, du0{};
#pragma omp for schedule(dynamic, 1)               // each is big
      for (int isub = 0; isub < nb; isub++) {      // Main loop through the subproblems
        const auto M0 = brk[isub + 1] - brk[isub]; // # NU pts in this subproblem
        // copy the location and data vectors for the nonuniform points
        kx0.resize(M0);
        ky0.resize(M0 * (N2 > 1));
        kz0.resize(M0 * (N3 > 1));
        dd0.resize(2 * M0);                            // complex strength data
        for (auto j = 0; j < M0; j++) {                // todo: can avoid this copying?
          const auto kk = sort_indices[j + brk[isub]]; // NU pt from subprob index list
          kx0[j]        = fold_rescale(kx[kk], N1);
          if (N2 > 1) ky0[j] = fold_rescale(ky[kk], N2);
          if (N3 > 1) kz0[j] = fold_rescale(kz[kk], N3);
          dd0[j * 2]     = data_nonuniform[kk * 2];     // real part
          dd0[j * 2 + 1] = data_nonuniform[kk * 2 + 1]; // imag part
        }
        // get the subgrid which will include padding by roughly nspread/2
        // get_subgrid sets
        BIGINT offset1, offset2, offset3, padded_size1, size1, size2, size3;
        // sets offsets and sizes
        get_subgrid(offset1, offset2, offset3, padded_size1, size1, size2, size3, M0,
                    kx0.data(), ky0.data(), kz0.data(), ns, ndims);
        if (opts.debug > 1) {
          print_subgrid_info(ndims, offset1, offset2, offset3, padded_size1, size1, size2,
                             size3, M0);
        }
        // allocate output data for this subgrid
        du0.resize(2 * padded_size1 * size2 * size3); // complex
        // Spread to subgrid without need for bounds checking or wrapping
        if (!(opts.flags & TF_OMIT_SPREADING)) {
          if (ndims == 1)
            spread_subproblem_1d(offset1, padded_size1, du0.data(), M0, kx0.data(),
                                 dd0.data(), opts);
          else if (ndims == 2)
            spread_subproblem_2d(offset1, offset2, padded_size1, size2, du0.data(), M0,
                                 kx0.data(), ky0.data(), dd0.data(), opts);
          else
            spread_subproblem_3d(offset1, offset2, offset3, padded_size1, size2, size3,
                                 du0.data(), M0, kx0.data(), ky0.data(), kz0.data(),
                                 dd0.data(), opts);
        }
        // do the adding of subgrid to output
        if (!(opts.flags & TF_OMIT_WRITE_TO_GRID)) {
          if (nthr > opts.atomic_threshold) { // see above for debug reporting
            add_wrapped_subgrid<true>(offset1, offset2, offset3, padded_size1, size1,
                                      size2, size3, N1, N2, N3, data_uniform,
                                      du0.data()); // R Blackwell's atomic version
          } else {
#pragma omp critical
            add_wrapped_subgrid<false>(offset1, offset2, offset3, padded_size1, size1,
                                       size2, size3, N1, N2, N3, data_uniform,
                                       du0.data());
          }
        }
      } // end main loop over subprobs
    }
    if (opts.debug)
      printf("\tt1 fancy spread: \t%.3g s (%ld subprobs)\n", timer.elapsedsec(), nb);
  } // end of choice of which t1 spread type to use
  return 0;
};

// --------------------------------------------------------------------------
template<uint16_t ns, uint16_t kerevalmeth>
FINUFFT_NEVER_INLINE static int interpSorted_kernel(
    const BIGINT *sort_indices, const UBIGINT N1, const UBIGINT N2, const UBIGINT N3,
    const FLT *data_uniform, const UBIGINT M, FLT *FINUFFT_RESTRICT kx,
    FLT *FINUFFT_RESTRICT ky, FLT *FINUFFT_RESTRICT kz,
    FLT *FINUFFT_RESTRICT data_nonuniform, const finufft_spread_opts &opts)
// Interpolate to NU pts in sorted order from a uniform grid.
// See spreadinterp() for doc.
{
  using simd_type                 = PaddedSIMD<FLT, 2 * ns>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto alignment = arch_t::alignment();
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto ns2 = ns * FLT(0.5); // half spread width, used as stencil shift

  CNTime timer{};
  const auto ndims = ndims_from_Ns(N1, N2, N3);
  auto nthr        = MY_OMP_GET_MAX_THREADS(); // guess # threads to use to interp
  if (opts.nthreads > 0) nthr = opts.nthreads; // user override, now without limit
#ifndef _OPENMP
  nthr = 1;                                    // single-threaded lib must override user
#endif
  if (opts.debug)
    printf("\tinterp %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d\n", ndims,
           (long long)M, (long long)N1, (long long)N2, (long long)N3, nthr);
  timer.start();
#pragma omp parallel num_threads(nthr)
  {
    static constexpr auto CHUNKSIZE = simd_size; // number of targets per chunk
    alignas(alignment) UBIGINT jlist[CHUNKSIZE];
    alignas(alignment) FLT xjlist[CHUNKSIZE], yjlist[CHUNKSIZE], zjlist[CHUNKSIZE];
    alignas(alignment) FLT outbuf[2 * CHUNKSIZE];
    // Kernels: static alloc is faster, so we do it for up to 3D...
    alignas(alignment) std::array<FLT, 3 * MAX_NSPREAD> kernel_values{0};
    auto *FINUFFT_RESTRICT ker1 = kernel_values.data();
    auto *FINUFFT_RESTRICT ker2 = kernel_values.data() + MAX_NSPREAD;
    auto *FINUFFT_RESTRICT ker3 = kernel_values.data() + 2 * MAX_NSPREAD;

    // Loop over interpolation chunks
    // main loop over NU trgs, interp each from U
    // (note: windows omp doesn't like unsigned loop vars)
#pragma omp for schedule(dynamic, 1000) // assign threads to NU targ pts:
    for (BIGINT i = 0; i < M; i += CHUNKSIZE) {
      // Setup buffers for this chunk
      const UBIGINT bufsize = (i + CHUNKSIZE > M) ? M - i : CHUNKSIZE;
      for (int ibuf = 0; ibuf < bufsize; ibuf++) {
        UBIGINT j    = sort_indices[i + ibuf];
        jlist[ibuf]  = j;
        xjlist[ibuf] = fold_rescale(kx[j], N1);
        if (ndims >= 2) yjlist[ibuf] = fold_rescale(ky[j], N2);
        if (ndims == 3) zjlist[ibuf] = fold_rescale(kz[j], N3);
      }

      // Loop over targets in chunk
      for (int ibuf = 0; ibuf < bufsize; ibuf++) {
        const auto xj = xjlist[ibuf];
        const auto yj = (ndims > 1) ? yjlist[ibuf] : 0;
        const auto zj = (ndims > 2) ? zjlist[ibuf] : 0;

        auto *FINUFFT_RESTRICT target = outbuf + 2 * ibuf;

        // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
        const auto i1 = BIGINT(std::ceil(xj - ns2)); // leftmost grid index
        const auto i2 = (ndims > 1) ? BIGINT(std::ceil(yj - ns2)) : 0; // min y grid index
        const auto i3 = (ndims > 2) ? BIGINT(std::ceil(zj - ns2)) : 0; // min z grid index

        const auto x1 = std::ceil(xj - ns2) - xj; // shift of ker center, in [-w/2,-w/2+1]
        const auto x2 = (ndims > 1) ? std::ceil(yj - ns2) - yj : 0;
        const auto x3 = (ndims > 2) ? std::ceil(zj - ns2) - zj : 0;

        // eval kernel values patch and use to interpolate from uniform data...
        if (!(opts.flags & TF_OMIT_SPREADING)) {
          switch (ndims) {
          case 1:
            ker_eval<ns, kerevalmeth, FLT, simd_type>(kernel_values.data(), opts, x1);
            interp_line<ns, simd_type>(target, data_uniform, ker1, i1, N1);
            break;
          case 2:
            ker_eval<ns, kerevalmeth, FLT, simd_type>(kernel_values.data(), opts, x1, x2);
            interp_square<ns, simd_type>(target, data_uniform, ker1, ker2, i1, i2, N1,
                                         N2);
            break;
          case 3:
            ker_eval<ns, kerevalmeth, FLT, simd_type>(kernel_values.data(), opts, x1, x2,
                                                      x3);
            interp_cube<ns, simd_type>(target, data_uniform, ker1, ker2, ker3, i1, i2, i3,
                                       N1, N2, N3);
            break;
          default: // can't get here
            FINUFFT_UNREACHABLE;
            break;
          }
        }
      } // end loop over targets in chunk

      // Copy result buffer to output array
      for (int ibuf = 0; ibuf < bufsize; ibuf++) {
        const UBIGINT j            = jlist[ibuf];
        data_nonuniform[2 * j]     = outbuf[2 * ibuf];
        data_nonuniform[2 * j + 1] = outbuf[2 * ibuf + 1];
      }

    } // end NU targ loop
  } // end parallel section
  if (opts.debug) printf("\tt2 spreading loop: \t%.3g s\n", timer.elapsedsec());
  return 0;
}

template<uint16_t NS>
static int interpSorted_dispatch(
    const BIGINT *sort_indices, const UBIGINT N1, const UBIGINT N2, const UBIGINT N3,
    FLT *FINUFFT_RESTRICT data_uniform, const UBIGINT M, FLT *FINUFFT_RESTRICT kx,
    FLT *FINUFFT_RESTRICT ky, FLT *FINUFFT_RESTRICT kz,
    FLT *FINUFFT_RESTRICT data_nonuniform, const finufft_spread_opts &opts) {
  static_assert(MIN_NSPREAD <= NS && NS <= MAX_NSPREAD,
                "NS must be in the range (MIN_NSPREAD, MAX_NSPREAD)");
  if constexpr (NS == MIN_NSPREAD) { // Base case
    if (opts.kerevalmeth)
      return interpSorted_kernel<MIN_NSPREAD, true>(
          sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts);
    else {
      return interpSorted_kernel<MIN_NSPREAD, false>(
          sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts);
    }
  } else {
    if (opts.nspread == NS) {
      if (opts.kerevalmeth) {
        return interpSorted_kernel<NS, true>(sort_indices, N1, N2, N3, data_uniform, M,
                                             kx, ky, kz, data_nonuniform, opts);
      } else {
        return interpSorted_kernel<NS, false>(sort_indices, N1, N2, N3, data_uniform, M,
                                              kx, ky, kz, data_nonuniform, opts);
      }
    } else {
      return interpSorted_dispatch<NS - 1>(sort_indices, N1, N2, N3, data_uniform, M, kx,
                                           ky, kz, data_nonuniform, opts);
    }
  }
}

int interpSorted(const BIGINT *sort_indices, const UBIGINT N1, const UBIGINT N2,
                 const UBIGINT N3, FLT *FINUFFT_RESTRICT data_uniform, const UBIGINT M,
                 FLT *FINUFFT_RESTRICT kx, FLT *FINUFFT_RESTRICT ky,
                 FLT *FINUFFT_RESTRICT kz, FLT *FINUFFT_RESTRICT data_nonuniform,
                 const finufft_spread_opts &opts) {
  return interpSorted_dispatch<MAX_NSPREAD>(sort_indices, N1, N2, N3, data_uniform, M, kx,
                                            ky, kz, data_nonuniform, opts);
}

///////////////////////////////////////////////////////////////////////////

int setup_spreader(finufft_spread_opts &opts, FLT eps, double upsampfac, int kerevalmeth,
                   int debug, int showwarn, int dim)
/* Initializes spreader kernel parameters given desired NUFFT tolerance eps,
   upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), ker eval meth
   (either 0:exp(sqrt()), 1: Horner ppval), and some debug-level flags.
   Also sets all default options in finufft_spread_opts. See finufft_spread_opts.h for
   opts. dim is spatial dimension (1,2, or 3). See finufft.cpp:finufft_plan() for where
   upsampfac is set. Must call this before any kernel evals done, otherwise segfault
   likely. Returns: 0  : success FINUFFT_WARN_EPS_TOO_SMALL : requested eps cannot be
   achieved, but proceed with best possible eps otherwise : failure (see codes in defs.h);
   spreading must not proceed Barnett 2017. debug, loosened eps logic 6/14/20.
*/
{
  if (upsampfac != 2.0 && upsampfac != 1.25) { // nonstandard sigma
    if (kerevalmeth == 1) {
      fprintf(stderr,
              "FINUFFT setup_spreader: nonstandard upsampfac=%.3g cannot be handled by "
              "kerevalmeth=1\n",
              upsampfac);
      return FINUFFT_ERR_HORNER_WRONG_BETA;
    }
    if (upsampfac <= 1.0) { // no digits would result
      fprintf(stderr, "FINUFFT setup_spreader: error, upsampfac=%.3g is <=1.0\n",
              upsampfac);
      return FINUFFT_ERR_UPSAMPFAC_TOO_SMALL;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (showwarn && upsampfac > 4.0)
      fprintf(stderr,
              "FINUFFT setup_spreader warning: upsampfac=%.3g way too large to be "
              "beneficial.\n",
              upsampfac);
  }

  // write out default finufft_spread_opts (some overridden in setup_spreader_for_nufft)
  opts.spread_direction = 0; // user should always set to 1 or 2 as desired
  opts.sort             = 2; // 2:auto-choice
  opts.kerpad           = 0; // affects only evaluate_kernel_vector
  opts.kerevalmeth      = kerevalmeth;
  opts.upsampfac        = upsampfac;
  opts.nthreads         = 0; // all avail
  opts.sort_threads     = 0; // 0:auto-choice
  // heuristic dir=1 chunking for nthr>>1, typical for intel i7 and skylake...
  opts.max_subproblem_size = (dim == 1) ? 10000 : 100000;
  opts.flags               = 0; // 0:no timing flags (>0 for experts only)
  opts.debug               = 0; // 0:no debug output
  // heuristic nthr above which switch OMP critical to atomic (add_wrapped...):
  opts.atomic_threshold = 10; // R Blackwell's value

  int ns, ier = 0;            // Set kernel width w (aka ns, nspread) then copy to opts...
  if (eps < EPSILON) {        // safety; there's no hope of beating e_mach
    if (showwarn)
      fprintf(stderr, "%s warning: increasing tol=%.3g to eps_mach=%.3g.\n", __func__,
              (double)eps, (double)EPSILON);
    eps = EPSILON; // only changes local copy (not any opts)
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }
  if (upsampfac == 2.0)                      // standard sigma (see SISC paper)
    ns = std::ceil(-log10(eps / (FLT)10.0)); // 1 digit per power of 10
  else                                       // custom sigma
    ns = std::ceil(-log(eps) / (PI * sqrt(1.0 - 1.0 / upsampfac))); // formula, gam=1
  ns = max(2, ns);        // (we don't have ns=1 version yet)
  if (ns > MAX_NSPREAD) { // clip to fit allocated arrays, Horner rules
    if (showwarn)
      fprintf(stderr,
              "%s warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; "
              "clipping to max %d.\n",
              __func__, upsampfac, (double)eps, ns, MAX_NSPREAD);
    ns  = MAX_NSPREAD;
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }
  opts.nspread = ns;
  // setup for reference kernel eval (via formula): select beta width param...
  // (even when kerevalmeth=1, this ker eval needed for FTs in onedim_*_kernel)
  opts.ES_halfwidth = (double)ns / 2; // constants to help (see below routines)
  opts.ES_c         = 4.0 / (double)(ns * ns);
  double betaoverns = 2.30;           // gives decent betas for default sigma=2.0
  if (ns == 2) betaoverns = 2.20;     // some small-width tweaks...
  if (ns == 3) betaoverns = 2.26;
  if (ns == 4) betaoverns = 2.38;
  if (upsampfac != 2.0) { // again, override beta for custom sigma
    FLT gamma  = 0.97;    // must match devel/gen_all_horner_C_code.m !
    betaoverns = gamma * PI * (1.0 - 1.0 / (2 * upsampfac)); // formula based on cutoff
  }
  opts.ES_beta = betaoverns * ns; // set the kernel beta parameter
  if (debug)
    printf("%s (kerevalmeth=%d) eps=%.3g sigma=%.3g: chose ns=%d beta=%.3g\n", __func__,
           kerevalmeth, (double)eps, upsampfac, ns, opts.ES_beta);

  return ier;
}

FLT evaluate_kernel(FLT x, const finufft_spread_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.(sqrt(1 - (2x/n_s)^2) - 1)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg finufft/onedim_* 2/17/17.
   Rescaled so max is 1, Barnett 7/21/24
*/
{
  if (abs(x) >= (FLT)opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp((FLT)opts.ES_beta * (sqrt((FLT)1.0 - (FLT)opts.ES_c * x * x) - (FLT)1.0));
}

template<uint8_t ns>
void set_kernel_args(FLT *args, FLT x) noexcept
// Fills vector args[] with kernel arguments x, x+1, ..., x+ns-1.
// needed for the vectorized kernel eval of Ludvig af K.
{
  for (int i = 0; i < ns; i++) args[i] = x + (FLT)i;
}
template<uint8_t N>
void evaluate_kernel_vector(FLT *ker, FLT *args, const finufft_spread_opts &opts) noexcept
/* Evaluate ES kernel for a vector of N arguments; by Ludvig af K.
   If opts.kerpad true, args and ker must be allocated for Npad, and args is
   written to (to pad to length Npad), only first N outputs are correct.
   Barnett 4/24/18 option to pad to mult of 4 for better SIMD vectorization.
   Rescaled so max is 1, Barnett 7/21/24

   Obsolete (replaced by Horner), but keep around for experimentation since
   works for arbitrary beta. Formula must match reference implementation.
*/
{
  FLT b = (FLT)opts.ES_beta;
  FLT c = (FLT)opts.ES_c;
  if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
    // Note (by Ludvig af K): Splitting kernel evaluation into two loops
    // seems to benefit auto-vectorization.
    // gcc 5.4 vectorizes first loop; gcc 7.2 vectorizes both loops
    int Npad = N;
    if (opts.kerpad) {               // since always same branch, no speed hit
      Npad = 4 * (1 + (N - 1) / 4);  // pad N to mult of 4; help i7 GCC, not xeon
      for (int i = N; i < Npad; ++i) // pad with 1-3 zeros for safe eval
        args[i] = 0.0;
    }
    for (int i = 0; i < Npad; i++) { // Loop 1: Compute exponential arguments
      // care! 1.0 is double...
      ker[i] = b * (sqrt((FLT)1.0 - c * args[i] * args[i]) - (FLT)1.0);
    }
    if (!(opts.flags & TF_OMIT_EVALUATE_EXPONENTIAL))
      for (int i = 0; i < Npad; i++) // Loop 2: Compute exponentials
        ker[i] = exp(ker[i]);
    if (opts.kerpad) {
      // padded part should be zero, in spread_subproblem_nd_kernels, there are
      // out of bound writes to trg arrays
      for (int i = N; i < Npad; ++i) ker[i] = 0.0;
    }
  } else {
    for (int i = 0; i < N; i++) // dummy for timing only
      ker[i] = 1.0;
  }
  // Separate check from arithmetic (Is this really needed? doesn't slow down)
  for (int i = 0; i < N; i++)
    if (abs(args[i]) >= (FLT)opts.ES_halfwidth) ker[i] = 0.0;
}

template<uint8_t w, uint8_t upsampfact, class simd_type> // aka ns
void eval_kernel_vec_Horner(FLT *FINUFFT_RESTRICT ker, const FLT x,
                            const finufft_spread_opts &opts) noexcept
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
This is the current evaluation method, since it's faster (except i7 w=16).
Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */

{
  // scale so local grid offset z in[-1,1]
  const FLT z                         = std::fma(FLT(2.0), x, FLT(w - 1));
  using arch_t                        = typename simd_type::arch_type;
  static constexpr auto alignment     = arch_t::alignment();
  static constexpr auto simd_size     = simd_type::size;
  static constexpr auto padded_ns     = (w + simd_size - 1) & ~(simd_size - 1);
  static constexpr auto horner_coeffs = []() constexpr noexcept {
    if constexpr (upsampfact == 200) {
      return get_horner_coeffs_200<FLT, w>();
    } else if constexpr (upsampfact == 125) {
      return get_horner_coeffs_125<FLT, w>();
    }
  }();
  static constexpr auto nc          = horner_coeffs.size();
  static constexpr auto use_ker_sym = (simd_size < w);

  alignas(alignment) static constexpr auto padded_coeffs =
      pad_2D_array_with_zeros<FLT, nc, w, padded_ns>(horner_coeffs);

  // use kernel symmetry trick if w > simd_size
  if constexpr (use_ker_sym) {
    static constexpr uint8_t tail          = w % simd_size;
    static constexpr uint8_t if_odd_degree = ((nc + 1) % 2);
    static constexpr uint8_t offset_start  = tail ? w - tail : w - simd_size;
    static constexpr uint8_t end_idx       = (w + (tail > 0)) / 2;
    const simd_type zv{z};
    const auto z2v = zv * zv;

    // some xsimd constant for shuffle or inverse
    static constexpr auto shuffle_batch = []() constexpr noexcept {
      if constexpr (tail) {
        return xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t,
                                          shuffle_index<tail>>();
      } else {
        return xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t,
                                          reverse_index<simd_size>>();
      }
    }();

    // process simd vecs
    simd_type k_prev, k_sym{0};
    for (uint8_t i{0}, offset = offset_start; i < end_idx;
         i += simd_size, offset -= simd_size) {
      auto k_odd = [i]() constexpr noexcept {
        if constexpr (if_odd_degree) {
          return simd_type::load_aligned(padded_coeffs[0].data() + i);
        } else {
          return simd_type{0};
        }
      }();
      auto k_even = simd_type::load_aligned(padded_coeffs[if_odd_degree].data() + i);
      for (uint8_t j{1 + if_odd_degree}; j < nc; j += 2) {
        const auto cji_odd  = simd_type::load_aligned(padded_coeffs[j].data() + i);
        const auto cji_even = simd_type::load_aligned(padded_coeffs[j + 1].data() + i);
        k_odd               = xsimd::fma(k_odd, z2v, cji_odd);
        k_even              = xsimd::fma(k_even, z2v, cji_even);
      }
      // left part
      xsimd::fma(k_odd, zv, k_even).store_aligned(ker + i);
      // right part symmetric to the left part
      if (offset >= end_idx) {
        if constexpr (tail) {
          // to use aligned store, we need shuffle the previous k_sym and current k_sym
          k_prev = k_sym;
          k_sym  = xsimd::fnma(k_odd, zv, k_even);
          xsimd::shuffle(k_sym, k_prev, shuffle_batch).store_aligned(ker + offset);
        } else {
          xsimd::swizzle(xsimd::fnma(k_odd, zv, k_even), shuffle_batch)
              .store_aligned(ker + offset);
        }
      }
    }
  } else {
    const simd_type zv(z);
    for (uint8_t i = 0; i < w; i += simd_size) {
      auto k = simd_type::load_aligned(padded_coeffs[0].data() + i);
      for (uint8_t j = 1; j < nc; ++j) {
        const auto cji = simd_type::load_aligned(padded_coeffs[j].data() + i);
        k              = xsimd::fma(k, zv, cji);
      }
      k.store_aligned(ker + i);
    }
  }
}

template<uint8_t ns>
static void interp_line_wrap(FLT *FINUFFT_RESTRICT target, const FLT *du, const FLT *ker,
                             const BIGINT i1, const UBIGINT N1) {
  /* This function is called when the kernel wraps around the grid. It is
     slower than interp_line.
     M. Barbone July 2024: - moved the logic to a separate function
                           - using fused multiply-add (fma) for better performance
     */
  std::array<FLT, 2> out{0};
  BIGINT j = i1;
  if (i1 < 0) { // wraps at left
    j += BIGINT(N1);
    for (uint8_t dx = 0; dx < -i1; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
    j -= BIGINT(N1);
    for (uint8_t dx = -i1; dx < ns; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
  } else if (i1 + ns >= N1) { // wraps at right
    for (uint8_t dx = 0; dx < N1 - i1; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
    j -= BIGINT(N1);
    for (uint8_t dx = N1 - i1; dx < ns; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
  } else {
    // padding is okay for ker, but it might spill over du array
    // so this checks for that case and does not explicitly vectorize
    for (uint8_t dx = 0; dx < ns; ++dx, ++j) {
      out[0] = std::fma(du[2 * j], ker[dx], out[0]);
      out[1] = std::fma(du[2 * j + 1], ker[dx], out[1]);
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<uint8_t ns, class simd_type>
void interp_line(FLT *FINUFFT_RESTRICT target, const FLT *du, const FLT *ker,
                 const BIGINT i1, const UBIGINT N1) {
  /* 1D interpolate complex values from size-ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   1d kernel evaluation list ker1.
   Inputs:
   du : input regular grid of size 2*N1 (alternating real,imag)
   ker1 : length-ns real array of 1d kernel evaluations
   i1 : start (left-most) x-coord index to read du from, where the indices
        of du run from 0 to N1-1, and indices outside that range are wrapped.
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Periodic wrapping in the du array is applied, assuming N1>=ns.
   Internally, dx indices into ker array j is index in complex du array.
   Barnett 6/16/17.
    M. Barbone July 2024: - moved wrapping logic to interp_line_wrap
                          - using explicit SIMD vectorization to overcome the out[2] array
                            limitation
*/
  using arch_t                       = typename simd_type::arch_type;
  static constexpr auto padding      = get_padding<FLT, 2 * ns>();
  static constexpr auto alignment    = arch_t::alignment();
  static constexpr auto simd_size    = simd_type::size;
  static constexpr auto regular_part = (2 * ns + padding) & (-(2 * simd_size));
  std::array<FLT, 2> out{0};
  const auto j = i1;
  // removing the wrapping leads up to 10% speedup in certain cases
  // moved the wrapping to another function to reduce instruction cache pressure
  if (i1 < 0 || i1 + ns >= N1 || i1 + ns + (padding + 1) / 2 >= N1) {
    return interp_line_wrap<ns>(target, du, ker, i1, N1);
  } else { // doesn't wrap
    // logic largely similar to spread 1D kernel, please see the explanation there
    // for the first part of this code
    const auto res = [du, j, ker]() constexpr noexcept {
      const auto du_ptr = du + 2 * j;
      simd_type res_low{0}, res_hi{0};
      for (uint8_t dx{0}; dx < regular_part; dx += 2 * simd_size) {
        const auto ker_v   = simd_type::load_aligned(ker + dx / 2);
        const auto du_pt0  = simd_type::load_unaligned(du_ptr + dx);
        const auto du_pt1  = simd_type::load_unaligned(du_ptr + dx + simd_size);
        const auto ker0low = xsimd::swizzle(ker_v, zip_low_index<arch_t>);
        const auto ker0hi  = xsimd::swizzle(ker_v, zip_hi_index<arch_t>);
        res_low            = xsimd::fma(ker0low, du_pt0, res_low);
        res_hi             = xsimd::fma(ker0hi, du_pt1, res_hi);
      }

      if constexpr (regular_part < 2 * ns) {
        const auto ker0    = simd_type::load_unaligned(ker + (regular_part / 2));
        const auto du_pt   = simd_type::load_unaligned(du_ptr + regular_part);
        const auto ker0low = xsimd::swizzle(ker0, zip_low_index<arch_t>);
        res_low            = xsimd::fma(ker0low, du_pt, res_low);
      }

      // This does a horizontal sum using a loop instead of relying on SIMD instructions
      // this is faster than the code below but less elegant.
      // lambdas here to limit the scope of temporary variables and have the compiler
      // optimize the code better
      return res_low + res_hi;
    }();
    const auto res_array = xsimd_to_array(res);
    for (uint8_t i{0}; i < simd_size; i += 2) {
      out[0] += res_array[i];
      out[1] += res_array[i + 1];
    }
    // this is where the code differs from spread_kernel, the interpolator does an extra
    // reduction step to SIMD elements down to 2 elements
    // This is known as horizontal sum in SIMD terminology

    // This does a horizontal sum using vector instruction,
    // is slower than summing and looping
    // clang-format off
    // const auto res_real = xsimd::shuffle(res_low, res_hi, select_even_mask<arch_t>);
    // const auto res_imag = xsimd::shuffle(res_low, res_hi, select_odd_mask<arch_t>);
    // out[0]              = xsimd::reduce_add(res_real);
    // out[1]              = xsimd::reduce_add(res_imag);
    // clang-format on
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<uint8_t ns, class simd_type>
static void interp_square_wrap(FLT *FINUFFT_RESTRICT target, const FLT *du,
                               const FLT *ker1, const FLT *ker2, const BIGINT i1,
                               const BIGINT i2, const UBIGINT N1, const UBIGINT N2) {
  /*
   * This function is called when the kernel wraps around the grid. It is slower than
   * the non wrapping version.
   * There is an extra case for when ker is padded and spills over the du array.
   * In this case uses the old non wrapping version.
   */
  std::array<FLT, 2> out{0};
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto alignment = arch_t::alignment();
  if (i1 >= 0 && i1 + ns <= N1 && i2 >= 0 && i2 + ns <= N2) {
    // store a horiz line (interleaved real,imag)
    alignas(alignment) std::array<FLT, 2 * ns> line{0};
    // add remaining const-y lines to the line (expensive inner loop)
    for (uint8_t dy{0}; dy < ns; ++dy) {
      const auto *l_ptr = du + 2 * (N1 * (i2 + dy) + i1); // (see above)
      for (uint8_t l{0}; l < 2 * ns; ++l) {
        line[l] = std::fma(ker2[dy], l_ptr[l], line[l]);
      }
    }
    // apply x kernel to the (interleaved) line and add together
    for (uint8_t dx{0}; dx < ns; dx++) {
      out[0] = std::fma(line[2 * dx], ker1[dx], out[0]);
      out[1] = std::fma(line[2 * dx + 1], ker1[dx], out[1]);
    }
  } else {
    std::array<UBIGINT, ns> j1{}, j2{}; // 1d ptr lists
    auto x = i1, y = i2;                // initialize coords
    for (uint8_t d{0}; d < ns; d++) {   // set up ptr lists
      if (x < 0) x += BIGINT(N1);
      if (x >= N1) x -= BIGINT(N1);
      j1[d] = x++;
      if (y < 0) y += BIGINT(N2);
      if (y >= N2) y -= BIGINT(N2);
      j2[d] = y++;
    }
    for (uint8_t dy{0}; dy < ns; dy++) { // use the pts lists
      const UBIGINT oy = N1 * j2[dy];    // offset due to y
      for (uint8_t dx{0}; dx < ns; dx++) {
        const auto k    = ker1[dx] * ker2[dy];
        const UBIGINT j = oy + j1[dx];
        out[0] += du[2 * j] * k;
        out[1] += du[2 * j + 1] * k;
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<uint8_t ns, class simd_type>
void interp_square(FLT *FINUFFT_RESTRICT target, const FLT *du, const FLT *ker1,
                   const FLT *ker2, const BIGINT i1, const BIGINT i2, const UBIGINT N1,
                   const UBIGINT N2)
/* 2D interpolate complex values from a ns*ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   ns*ns outer product of the 1d kernel lists ker1 and ker2.
   Inputs:
   du : input regular grid of size 2*N1*N2 (alternating real,imag)
   ker1, ker2 : length-ns real arrays of 1d kernel evaluations
   i1 : start (left-most) x-coord index to read du from, where the indices
        of du run from 0 to N1-1, and indices outside that range are wrapped.
   i2 : start (bottom) y-coord index to read du from.
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Periodic wrapping in the du array is applied, assuming N1,N2>=ns.
   Internally, dx,dy indices into ker array, l indices the 2*ns interleaved
   line array, j is index in complex du array.
   Barnett 6/16/17.
   No-wrap case sped up for FMA/SIMD by Martin Reinecke 6/19/23, with this note:
   "It reduces the number of arithmetic operations per "iteration" in the
   innermost loop from 2.5 to 2, and these two can be converted easily to a
   fused multiply-add instruction (potentially vectorized). Also the strides
   of all invoved arrays in this loop are now 1, instead of the mixed 1 and 2
   before. Also the accumulation onto a double[2] is limiting the vectorization
   pretty badly. I think this is now much more analogous to the way the spread
   operation is implemented, which has always been much faster when I tested
   it."
   M. Barbone July 2024: - moved the wrapping logic to interp_square_wrap
                         - using explicit SIMD vectorization to overcome the out[2] array
                           limitation
   The code is largely similar to 1D interpolation, please see the explanation there
*/
{
  std::array<FLT, 2> out{0};
  // no wrapping: avoid ptrs
  using arch_t                          = typename simd_type::arch_type;
  static constexpr auto padding         = get_padding<FLT, 2 * ns>();
  static constexpr auto alignment       = arch_t::alignment();
  static constexpr auto simd_size       = simd_type::size;
  static constexpr uint8_t line_vectors = (2 * ns + padding) / simd_size;
  if (i1 >= 0 && i1 + ns <= N1 && i2 >= 0 && i2 + ns <= N2 &&
      (i1 + ns + (padding + 1) / 2 < N1)) {
    const auto line = [du, N1, i1 = UBIGINT(i1), i2 = UBIGINT(i2),
                       ker2]() constexpr noexcept {
      // new array du_pts to store the du values for the current y line
      std::array<simd_type, line_vectors> line{0};
      // block for first y line, to avoid explicitly initializing line with zeros
      // add remaining const-y lines to the line (expensive inner loop)
      for (uint8_t dy{0}; dy < ns; dy++) {
        const auto l_ptr = du + 2 * (N1 * (i2 + dy) + i1); // (see above)
        const simd_type ker2_v{ker2[dy]};
        for (uint8_t l{0}; l < line_vectors; ++l) {
          const auto du_pt = simd_type::load_unaligned(l * simd_size + l_ptr);
          line[l]          = xsimd::fma(ker2_v, du_pt, line[l]);
        }
      }
      return line;
    }();
    // This is the same as 1D interpolation
    // using lambda to limit the scope of the temporary variables
    const auto res = [ker1, &line]() constexpr noexcept {
      // apply x kernel to the (interleaved) line and add together
      simd_type res_low{0}, res_hi{0};
      // Start the loop from the second iteration
      for (uint8_t i{0}; i < (line_vectors & ~1); // NOLINT(*-too-small-loop-variable)
           i += 2) {
        const auto ker1_v  = simd_type::load_aligned(ker1 + i * simd_size / 2);
        const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t>);
        const auto ker1hi  = xsimd::swizzle(ker1_v, zip_hi_index<arch_t>);
        res_low            = xsimd::fma(ker1low, line[i], res_low);
        res_hi             = xsimd::fma(ker1hi, line[i + 1], res_hi);
      }
      if constexpr (line_vectors % 2) {
        const auto ker1_v =
            simd_type::load_aligned(ker1 + (line_vectors - 1) * simd_size / 2);
        const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t>);
        res_low            = xsimd::fma(ker1low, line.back(), res_low);
      }
      return res_low + res_hi;
    }();
    const auto res_array = xsimd_to_array(res);
    for (uint8_t i{0}; i < simd_size; i += 2) {
      out[0] += res_array[i];
      out[1] += res_array[i + 1];
    }
  } else { // wraps somewhere: use ptr list
    // this is slower than above, but occurs much less often, with fractional
    // rate O(ns/min(N1,N2)). Thus this code doesn't need to be so optimized.
    return interp_square_wrap<ns, simd_type>(target, du, ker1, ker2, i1, i2, N1, N2);
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<uint8_t ns, class simd_type>
static void interp_cube_wrapped(FLT *FINUFFT_RESTRICT target, const FLT *du,
                                const FLT *ker1, const FLT *ker2, const FLT *ker3,
                                const BIGINT i1, const BIGINT i2, const BIGINT i3,
                                const UBIGINT N1, const UBIGINT N2, const UBIGINT N3) {
  /*
   * This function is called when the kernel wraps around the cube.
   * Similarly to 2D and 1D wrapping, this is slower than the non wrapping version.
   */
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto alignment = arch_t::alignment();
  const auto in_bounds_1          = (i1 >= 0) & (i1 + ns <= N1);
  const auto in_bounds_2          = (i2 >= 0) & (i2 + ns <= N2);
  const auto in_bounds_3          = (i3 >= 0) & (i3 + ns <= N3);
  std::array<FLT, 2> out{0};
  // case no wrapping needed but padding spills over du array.
  // Hence, no explicit vectorization but the code is still faster
  if (FINUFFT_LIKELY(in_bounds_1 && in_bounds_2 && in_bounds_3)) {
    // no wrapping: avoid ptrs (by far the most common case)
    // store a horiz line (interleaved real,imag)
    // initialize line with zeros; hard to avoid here, but overhead small in 3D
    alignas(alignment) std::array<FLT, 2 * ns> line{0};
    // co-add y and z contributions to line in x; do not apply x kernel yet
    // This is expensive innermost loop
    for (uint8_t dz{0}; dz < ns; ++dz) {
      const auto oz = N1 * N2 * (i3 + dz);                      // offset due to z
      for (uint8_t dy{0}; dy < ns; ++dy) {
        const auto l_ptr = du + 2 * (oz + N1 * (i2 + dy) + i1); // ptr start of line
        const auto ker23 = ker2[dy] * ker3[dz];
        for (uint8_t l{0}; l < 2 * ns; ++l) { // loop over ns interleaved (R,I) pairs
          line[l] = std::fma(l_ptr[l], ker23, line[l]);
        }
      }
    }
    // apply x kernel to the (interleaved) line and add together (cheap)
    for (uint8_t dx{0}; dx < ns; ++dx) {
      out[0] = std::fma(line[2 * dx], ker1[dx], out[0]);
      out[1] = std::fma(line[2 * dx + 1], ker1[dx], out[1]);
    }
  } else {
    // ...can be slower since this case only happens with probability
    // O(ns/min(N1,N2,N3))
    alignas(alignment) std::array<UBIGINT, ns> j1{}, j2{}, j3{}; // 1d ptr lists
    auto x = i1, y = i2, z = i3;                                 // initialize coords
    for (uint8_t d{0}; d < ns; d++) {                            // set up ptr lists
      if (x < 0) x += BIGINT(N1);
      if (x >= N1) x -= BIGINT(N1);
      j1[d] = x++;
      if (y < 0) y += BIGINT(N2);
      if (y >= N2) y -= BIGINT(N2);
      j2[d] = y++;
      if (z < 0) z += BIGINT(N3);
      if (z >= N3) z -= BIGINT(N3);
      j3[d] = z++;
    }
    for (uint8_t dz{0}; dz < ns; dz++) {     // use the pts lists
      const auto oz = N1 * N2 * j3[dz];      // offset due to z
      for (uint8_t dy{0}; dy < ns; dy++) {
        const auto oy    = oz + N1 * j2[dy]; // offset due to y & z
        const auto ker23 = ker2[dy] * ker3[dz];
        for (uint8_t dx{0}; dx < ns; dx++) {
          const auto k = ker1[dx] * ker23;
          const auto j = oy + j1[dx];
          out[0] += du[2 * j] * k;
          out[1] += du[2 * j + 1] * k;
        }
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<uint8_t ns, class simd_type>
void interp_cube(FLT *FINUFFT_RESTRICT target, const FLT *du, const FLT *ker1,
                 const FLT *ker2, const FLT *ker3, const BIGINT i1, const BIGINT i2,
                 const BIGINT i3, const UBIGINT N1, const UBIGINT N2, const UBIGINT N3)
/* 3D interpolate complex values from a ns*ns*ns block of the du (uniform grid
   data) array to a single complex output value "target", using as weights the
   ns*ns*ns outer product of the 1d kernel lists ker1, ker2, and ker3.
   Inputs:
   du : input regular grid of size 2*N1*N2*N3 (alternating real,imag)
   ker1, ker2, ker3 : length-ns real arrays of 1d kernel evaluations
   i1 : start (left-most) x-coord index to read du from, where the indices
        of du run from 0 to N1-1, and indices outside that range are wrapped.
   i2 : start (bottom) y-coord index to read du from.
   i3 : start (lowest) z-coord index to read du from.
   ns : kernel width (must be <=MAX_NSPREAD)
   Outputs:
   target : size 2 array (containing real,imag) of interpolated output

   Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
   Internally, dx,dy,dz indices into ker array, l indices the 2*ns interleaved
   line array, j is index in complex du array.

   Internally, dx,dy,dz indices into ker array, j index in complex du array.
   Barnett 6/16/17.
   No-wrap case sped up for FMA/SIMD by Reinecke 6/19/23
   (see above note in interp_square)
   Barbone July 2024: - moved wrapping logic to interp_cube_wrapped
                      - using explicit SIMD vectorization to overcome the out[2] array
                        limitation
   The code is largely similar to 2D and 1D interpolation, please see the explanation
   there
*/
{
  using arch_t                          = typename simd_type::arch_type;
  static constexpr auto padding         = get_padding<FLT, 2 * ns>();
  static constexpr auto alignment       = arch_t::alignment();
  static constexpr auto simd_size       = simd_type::size;
  static constexpr auto ker23_size      = (ns + simd_size - 1) & -simd_size;
  static constexpr uint8_t line_vectors = (2 * ns + padding) / simd_size;
  const auto in_bounds_1                = (i1 >= 0) & (i1 + ns <= N1);
  const auto in_bounds_2                = (i2 >= 0) & (i2 + ns <= N2);
  const auto in_bounds_3                = (i3 >= 0) & (i3 + ns <= N3);
  std::array<FLT, 2> out{0};
  if (in_bounds_1 && in_bounds_2 && in_bounds_3 && (i1 + ns + (padding + 1) / 2 < N1)) {
    const auto line = [N1, N2, i1 = UBIGINT(i1), i2 = UBIGINT(i2), i3 = UBIGINT(i3), ker2,
                       ker3, du]() constexpr noexcept {
      std::array<simd_type, line_vectors> line{0};
      for (uint8_t dz{0}; dz < ns; ++dz) {
        const auto oz = N1 * N2 * (i3 + dz);                      // offset due to z
        for (uint8_t dy{0}; dy < ns; ++dy) {
          const auto l_ptr = du + 2 * (oz + N1 * (i2 + dy) + i1); // ptr start of line
          const simd_type ker23{ker2[dy] * ker3[dz]};
          for (uint8_t l{0}; l < line_vectors; ++l) {
            const auto du_pt = simd_type::load_unaligned(l * simd_size + l_ptr);
            line[l]          = xsimd::fma(ker23, du_pt, line[l]);
          }
        }
      }
      return line;
    }();
    const auto res = [ker1, &line]() constexpr noexcept {
      // apply x kernel to the (interleaved) line and add together
      simd_type res_low{0}, res_hi{0};
      // Start the loop from the second iteration
      for (uint8_t i{0}; i < (line_vectors & ~1); // NOLINT(*-too-small-loop-variable)
           i += 2) {
        const auto ker1_v  = simd_type::load_aligned(ker1 + i * simd_size / 2);
        const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t>);
        const auto ker1hi  = xsimd::swizzle(ker1_v, zip_hi_index<arch_t>);
        res_low            = xsimd::fma(ker1low, line[i], res_low);
        res_hi             = xsimd::fma(ker1hi, line[i + 1], res_hi);
      }
      if constexpr (line_vectors % 2) {
        const auto ker1_v =
            simd_type::load_aligned(ker1 + (line_vectors - 1) * simd_size / 2);
        const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t>);
        res_low            = xsimd::fma(ker1low, line.back(), res_low);
      }
      return res_low + res_hi;
    }();
    const auto res_array = xsimd_to_array(res);
    for (uint8_t i{0}; i < simd_size; i += 2) {
      out[0] += res_array[i];
      out[1] += res_array[i + 1];
    }
  } else {
    return interp_cube_wrapped<ns, simd_type>(target, du, ker1, ker2, ker3, i1, i2, i3,
                                              N1, N2, N3);
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<uint8_t ns, bool kerevalmeth>
FINUFFT_NEVER_INLINE void spread_subproblem_1d_kernel(
    const BIGINT off1, const UBIGINT size1, FLT *FINUFFT_RESTRICT du, const UBIGINT M,
    const FLT *const kx, const FLT *const dd, const finufft_spread_opts &opts) noexcept {
  /* 1D spreader from nonuniform to uniform subproblem grid, without wrapping.
     Inputs:
     off1 - integer offset of left end of du subgrid from that of overall fine
            periodized output grid {0,1,...N-1}.
     size1 - integer length of output subgrid du
     M - number of NU pts in subproblem
     kx (length M) - are rescaled NU source locations, should lie in
                     [off1+ns/2,off1+size1-1-ns/2] so as kernels stay in bounds
     dd (length M complex, interleaved) - source strengths
     Outputs:
     du (length size1 complex, interleaved) - preallocated uniform subgrid array

     The reason periodic wrapping is avoided in subproblems is speed: avoids
     conditionals, indirection (pointers), and integer mod. Originally 2017.
     Kernel eval mods by Ludvig al Klinteberg.
     Fixed so rounding to integer grid consistent w/ get_subgrid, prevents
     chance of segfault when epsmach*N1>O(1), assuming max() and ceil() commute.
     This needed off1 as extra arg. AHB 11/30/20.
     Vectorized using xsimd by M. Barbone 06/24.
  */
  using simd_type                 = PaddedSIMD<FLT, 2 * ns>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto padding   = get_padding<FLT, 2 * ns>();
  static constexpr auto alignment = arch_t::alignment();
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto ns2       = ns * FLT(0.5); // half spread width
  // something weird here. Reversing ker{0} and std fill causes ker
  // to be zeroed inside the loop GCC uses AVX, clang AVX2
  alignas(alignment) std::array<FLT, MAX_NSPREAD> ker{0};
  std::fill(du, du + 2 * size1, 0); // zero output
  // no padding needed if MAX_NSPREAD is 16
  // the largest read is 16 floats with avx512
  // if larger instructions will be available or half precision is used, this should be
  // padded
  for (uint64_t i{0}; i < M; i++) { // loop over NU pts
    // initializes a dd_pt that is const
    // should not make a difference in performance
    // but is a hint to the compiler that after the lambda
    // dd_pt is not modified and can be kept as is in a register
    // given (re, im) in this case dd[i*2] and dd[i*2+1]
    // this function returns a simd register of size simd_size
    // initialized as follows:
    // +-----------------------+
    // |re|im|re|im|re|im|re|im|
    // +-----------------------+
    const auto dd_pt = initialize_complex_register<simd_type>(dd[i * 2], dd[i * 2 + 1]);
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = BIGINT(std::ceil(kx[i] - ns2)); // fine grid start index
    // FLT(i1) has different semantics and results an extra cast
    const auto x1 = [i, kx]() constexpr noexcept {
      auto x1 = std::ceil(kx[i] - ns2) - kx[i]; // x1 in [-w/2,-w/2+1], up to rounding
      // However if N1*epsmach>O(1) then can cause O(1) errors in x1, hence ppoly
      // kernel evaluation will fall outside their designed domains, >>1 errors.
      // This can only happen if the overall error would be O(1) anyway. Clip x1??
      if (x1 < -ns2) x1 = -ns2;
      if (x1 > -ns2 + 1) x1 = -ns2 + 1; // ***
      return x1;
    }();
    // Libin improvement: pass ker as a parameter and allocate it outside the loop
    // gcc13 + 10% speedup
    ker_eval<ns, kerevalmeth, FLT, simd_type>(ker.data(), opts, x1);
    //    const auto ker = ker_eval<ns, kerevalmeth, FLT, simd_type>(opts, x1);
    const auto j = i1 - off1; // offset rel to subgrid, starts the output indices
    auto *FINUFFT_RESTRICT trg = du + 2 * j; // restrict helps compiler to vectorize
    // du is padded, so we can use SIMD even if we write more than ns values in du
    // ker is also padded.
    // regular_part, source Agner Fog
    // [VCL](https://www.agner.org/optimize/vcl_manual.pdf)
    // Given 2*ns+padding=L so that L = M*simd_size
    // if M is even then regular_part == M else regular_part == (M-1) * simd_size
    // this means that the elements from regular_part to L are a special case that
    // needs a different handling. These last elements are not computed in the loop but,
    // the if constexpr block at the end of the loop takes care of them.
    // This allows to save one load at each loop iteration.
    // The special case, allows to minimize padding otherwise out of bounds access.
    // See below for the details.
    static constexpr auto regular_part = (2 * ns + padding) & (-(2 * simd_size));
    // this loop increment is 2*simd_size by design
    // it allows to save one load this way at each iteration

    // This does for each element e of the subgrid, x1 defined above and pt the NU point
    // the following: e += scaled_kernel(2*x1/n_s)*pt, where "scaled_kernel" is defined
    // on [-1,1].
    // Using uint8_t in loops to favor unrolling.
    // Most compilers limit the unrolling to 255, uint8_t is at most 255
    for (uint8_t dx{0}; dx < regular_part; dx += 2 * simd_size) {
      // read ker_v which is simd_size wide from ker
      // ker_v looks like this:
      // +-----------------------+
      // |y0|y1|y2|y3|y4|y5|y6|y7|
      // +-----------------------+
      const auto ker_v = simd_type::load_aligned(ker.data() + dx / 2);
      // read 2*SIMD vectors from the subproblem grid
      const auto du_pt0 = simd_type::load_unaligned(trg + dx);
      const auto du_pt1 = simd_type::load_unaligned(trg + dx + simd_size);
      // swizzle is faster than zip_lo(ker_v, ker_v) and zip_hi(ker_v, ker_v)
      // swizzle in this case is equivalent to zip_lo and zip_hi respectively
      const auto ker0low = xsimd::swizzle(ker_v, zip_low_index<arch_t>);
      // ker 0 looks like this now:
      // +-----------------------+
      // |y0|y0|y1|y1|y2|y2|y3|y3|
      // +-----------------------+
      const auto ker0hi = xsimd::swizzle(ker_v, zip_hi_index<arch_t>);
      // ker 1 looks like this now:
      // +-----------------------+
      // |y4|y4|y5|y5|y6|y6|y7|y7|
      // +-----------------------+
      // same as before each element of the subproblem grid is multiplied by the
      // corresponding element of the kernel since dd_pt is re|im interleaves res0 is also
      // correctly re|im interleaved
      // doing this for two SIMD vectors at once allows to fully utilize ker_v instead of
      // wasting the higher half
      const auto res0 = xsimd::fma(ker0low, dd_pt, du_pt0);
      const auto res1 = xsimd::fma(ker0hi, dd_pt, du_pt1);
      res0.store_unaligned(trg + dx);
      res1.store_unaligned(trg + dx + simd_size);
    }
    // sanity check at compile time that all the elements are computed
    static_assert(regular_part + simd_size >= 2 * ns);
    // case where the 2*ns is not a multiple of 2*simd_size
    // checking 2*ns instead of 2*ns+padding as we do not need to compute useless zeros...
    if constexpr (regular_part < 2 * ns) {
      // here we need to load the last kernel values,
      // but we can avoid computing extra padding
      // also this padding will result in out-of-bounds access to trg
      // The difference between this and the loop is that ker0hi is not computed and
      // the corresponding memory is not accessed
      const auto ker0    = simd_type::load_unaligned(ker.data() + (regular_part / 2));
      const auto du_pt   = simd_type::load_unaligned(trg + regular_part);
      const auto ker0low = xsimd::swizzle(ker0, zip_low_index<arch_t>);
      const auto res     = xsimd::fma(ker0low, dd_pt, du_pt);
      res.store_unaligned(trg + regular_part);
    }
  }
}

template<uint8_t NS>
static void spread_subproblem_1d_dispatch(
    const BIGINT off1, const UBIGINT size1, FLT *FINUFFT_RESTRICT du, const UBIGINT M,
    const FLT *kx, const FLT *dd, const finufft_spread_opts &opts) noexcept {
  /* this is a dispatch function that will call the correct kernel based on the ns
   it recursively iterates from MAX_NSPREAD to MIN_NSPREAD
   it generates the following code:
   if (ns == MAX_NSPREAD) {
     if (opts.kerevalmeth) {
       return spread_subproblem_1d_kernel<MAX_NSPREAD, true>(off1, size1, du, M, kx, dd,
       opts);
    } else {
       return spread_subproblem_1d_kernel<MAX_NSPREAD, false>(off1, size1, du, M, kx, dd,
       opts);
   }
   if (ns == MAX_NSPREAD-1) {
     if (opts.kerevalmeth) {
       return spread_subproblem_1d_kernel<MAX_NSPREAD-1, true>(off1, size1, du, M, kx, dd,
       opts);
     } else {
       return spread_subproblem_1d_kernel<MAX_NSPREAD-1, false>(off1, size1, du, M, kx,
       dd, opts);
     }
   }
   ...
   NOTE: using a big MAX_NSPREAD will generate a lot of code
         if MAX_NSPREAD gets too large it will crash the compiler with a compile time
         stack overflow. Older compiler will just throw an internal error without
         providing any useful information on the error.
         This is a known issue with template metaprogramming.
         If you increased MAX_NSPREAD and the code does not compile, try reducing it.
  */
  static_assert(MIN_NSPREAD <= NS && NS <= MAX_NSPREAD,
                "NS must be in the range (MIN_NSPREAD, MAX_NSPREAD)");
  if constexpr (NS == MIN_NSPREAD) { // Base case
    if (opts.kerevalmeth)
      return spread_subproblem_1d_kernel<MIN_NSPREAD, true>(off1, size1, du, M, kx, dd,
                                                            opts);
    else {
      return spread_subproblem_1d_kernel<MIN_NSPREAD, false>(off1, size1, du, M, kx, dd,
                                                             opts);
    }
  } else {
    if (opts.nspread == NS) {
      if (opts.kerevalmeth) {
        return spread_subproblem_1d_kernel<NS, true>(off1, size1, du, M, kx, dd, opts);
      } else {
        return spread_subproblem_1d_kernel<NS, false>(off1, size1, du, M, kx, dd, opts);
      }
    } else {
      return spread_subproblem_1d_dispatch<NS - 1>(off1, size1, du, M, kx, dd, opts);
    }
  }
}

void spread_subproblem_1d(BIGINT off1, UBIGINT size1, FLT *du, UBIGINT M, FLT *kx,
                          FLT *dd, const finufft_spread_opts &opts) noexcept
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array
   For algoritmic details see spread_subproblem_1d_kernel.
*/
{
  spread_subproblem_1d_dispatch<MAX_NSPREAD>(off1, size1, du, M, kx, dd, opts);
}

template<uint8_t ns, bool kerevalmeth>
FINUFFT_NEVER_INLINE static void spread_subproblem_2d_kernel(
    const BIGINT off1, const BIGINT off2, const UBIGINT size1, const UBIGINT size2,
    FLT *FINUFFT_RESTRICT du, const UBIGINT M, const FLT *kx, const FLT *ky,
    const FLT *dd, const finufft_spread_opts &opts) noexcept
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array
   For algoritmic details see spread_subproblem_1d_kernel.
*/
{
  using simd_type                 = PaddedSIMD<FLT, 2 * ns>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto padding   = get_padding<FLT, 2 * ns>();
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = arch_t::alignment();
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in all three directions in a single kernel evaluation call.
  static constexpr auto ns2 = ns * FLT(0.5); // half spread width
  alignas(alignment) std::array<FLT, 2 * MAX_NSPREAD> kernel_values{0};
  std::fill(du, du + 2 * size1 * size2, 0);  // initialized to 0 due to the padding
  for (uint64_t pt = 0; pt < M; pt++) {      // loop over NU pts
    const auto dd_pt = initialize_complex_register<simd_type>(dd[pt * 2], dd[pt * 2 + 1]);
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = (BIGINT)std::ceil(kx[pt] - ns2); // fine grid start indices
    const auto i2 = (BIGINT)std::ceil(ky[pt] - ns2);
    const auto x1 = (FLT)std::ceil(kx[pt] - ns2) - kx[pt];
    const auto x2 = (FLT)std::ceil(ky[pt] - ns2) - ky[pt];
    ker_eval<ns, kerevalmeth, FLT, simd_type>(kernel_values.data(), opts, x1, x2);
    const auto *ker1 = kernel_values.data();
    const auto *ker2 = kernel_values.data() + MAX_NSPREAD;
    // Combine kernel with complex source value to simplify inner loop
    // here 2* is because of complex
    static constexpr uint8_t kerval_vectors = (2 * ns + padding) / simd_size;
    static_assert(kerval_vectors > 0, "kerval_vectors must be greater than 0");
    // wrapping this in a lambda gives an extra 10% speedup (gcc13)
    // the compiler realizes the values are constant after the lambda
    // Guess: it realizes what is the invariant and moves some operations outside the loop
    //        it might also realize that some variables are not needed anymore and can
    //        re-use the registers with other data.
    const auto ker1val_v = [ker1, dd_pt]() constexpr noexcept {
      // array of simd_registers that will store the kernel values
      std::array<simd_type, kerval_vectors> ker1val_v{};
      // similar to the 1D case, we compute the kernel values in advance
      // and store them in simd_registers.
      // Compared to the 1D case the difference is that here ker values are stored in
      // an array of simd_registers.
      // This is a hint to the compiler to keep the values in registers, instead of
      // pushing them to the stack.
      // Same as the 1D case, the loop is structured in a way to half the number of loads
      // This cause an issue with the last elements, but this is handled in the
      // `if constexpr`.
      // For more details please read the 1D case. The difference is that
      // here the loop is on the number of simd vectors In the 1D case the loop is on the
      // number of elements in the kernel
      for (uint8_t i = 0; i < (kerval_vectors & ~1); // NOLINT(*-too-small-loop-variable)
           i += 2) {
        const auto ker1_v  = simd_type::load_aligned(ker1 + i * simd_size / 2);
        const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t>);
        const auto ker1hi  = xsimd::swizzle(ker1_v, zip_hi_index<arch_t>);
        // this initializes the entire vector registers with the same value
        // the ker1val_v[i] looks like this:
        // +-----------------------+
        // |y0|y0|y0|y0|y0|y0|y0|y0|
        // +-----------------------+
        ker1val_v[i]     = ker1low * dd_pt;
        ker1val_v[i + 1] = ker1hi * dd_pt; // same as above
      }
      if constexpr (kerval_vectors % 2) {
        const auto ker1_v =
            simd_type::load_unaligned(ker1 + (kerval_vectors - 1) * simd_size / 2);
        const auto res = xsimd::swizzle(ker1_v, zip_low_index<arch_t>) * dd_pt;
        ker1val_v[kerval_vectors - 1] = res;
      }
      return ker1val_v;
    }();

    // critical inner loop:
    for (auto dy = 0; dy < ns; ++dy) {
      const auto j = size1 * (i2 - off2 + dy) + i1 - off1; // should be in subgrid
      auto *FINUFFT_RESTRICT trg = du + 2 * j;
      const simd_type kerval_v(ker2[dy]);
      for (uint8_t i = 0; i < kerval_vectors; ++i) {
        const auto trg_v  = simd_type::load_unaligned(trg + i * simd_size);
        const auto result = xsimd::fma(kerval_v, ker1val_v[i], trg_v);
        result.store_unaligned(trg + i * simd_size);
      }
    }
  }
}

template<uint8_t NS>
void spread_subproblem_2d_dispatch(
    const BIGINT off1, const BIGINT off2, const UBIGINT size1, const UBIGINT size2,
    FLT *FINUFFT_RESTRICT du, const UBIGINT M, const FLT *kx, const FLT *ky,
    const FLT *dd, const finufft_spread_opts &opts) {
  static_assert(MIN_NSPREAD <= NS && NS <= MAX_NSPREAD,
                "NS must be in the range (MIN_NSPREAD, MAX_NSPREAD)");
  if constexpr (NS == MIN_NSPREAD) { // Base case
    if (opts.kerevalmeth)
      return spread_subproblem_2d_kernel<MIN_NSPREAD, true>(off1, off2, size1, size2, du,
                                                            M, kx, ky, dd, opts);
    else {
      return spread_subproblem_2d_kernel<MIN_NSPREAD, false>(off1, off2, size1, size2, du,
                                                             M, kx, ky, dd, opts);
    }
  } else {
    if (opts.nspread == NS) {
      if (opts.kerevalmeth) {
        return spread_subproblem_2d_kernel<NS, true>(off1, off2, size1, size2, du, M, kx,
                                                     ky, dd, opts);
      } else {
        return spread_subproblem_2d_kernel<NS, false>(off1, off2, size1, size2, du, M, kx,
                                                      ky, dd, opts);
      }
    } else {
      return spread_subproblem_2d_dispatch<NS - 1>(off1, off2, size1, size2, du, M, kx,
                                                   ky, dd, opts);
    }
  }
}

void spread_subproblem_2d(const BIGINT off1, const BIGINT off2, const UBIGINT size1,
                          const UBIGINT size2, FLT *FINUFFT_RESTRICT du, const UBIGINT M,
                          const FLT *kx, const FLT *ky, const FLT *dd,
                          const finufft_spread_opts &opts) noexcept
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array
   For algoritmic details see spread_subproblem_1d_kernel.
*/
{
  spread_subproblem_2d_dispatch<MAX_NSPREAD>(off1, off2, size1, size2, du, M, kx, ky, dd,
                                             opts);
}

template<uint8_t ns, bool kerevalmeth>
FINUFFT_NEVER_INLINE void spread_subproblem_3d_kernel(
    const BIGINT off1, const BIGINT off2, const BIGINT off3, const UBIGINT size1,
    const UBIGINT size2, const UBIGINT size3, FLT *FINUFFT_RESTRICT du, const UBIGINT M,
    const FLT *kx, const FLT *ky, const FLT *kz, const FLT *dd,
    const finufft_spread_opts &opts) noexcept {
  using simd_type                 = PaddedSIMD<FLT, 2 * ns>;
  using arch_t                    = typename simd_type::arch_type;
  static constexpr auto padding   = get_padding<FLT, 2 * ns>();
  static constexpr auto simd_size = simd_type::size;
  static constexpr auto alignment = arch_t::alignment();

  static constexpr auto ns2 = ns * FLT(0.5); // half spread width
  alignas(alignment) std::array<FLT, 3 * MAX_NSPREAD> kernel_values{0};
  std::fill(du, du + 2 * size1 * size2 * size3, 0);

  for (uint64_t pt = 0; pt < M; pt++) { // loop over NU pts
    const auto dd_pt = initialize_complex_register<simd_type>(dd[pt * 2], dd[pt * 2 + 1]);
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = (BIGINT)std::ceil(kx[pt] - ns2); // fine grid start indices
    const auto i2 = (BIGINT)std::ceil(ky[pt] - ns2);
    const auto i3 = (BIGINT)std::ceil(kz[pt] - ns2);
    const auto x1 = std::ceil(kx[pt] - ns2) - kx[pt];
    const auto x2 = std::ceil(ky[pt] - ns2) - ky[pt];
    const auto x3 = std::ceil(kz[pt] - ns2) - kz[pt];

    ker_eval<ns, kerevalmeth, FLT, simd_type>(kernel_values.data(), opts, x1, x2, x3);
    const auto *ker1 = kernel_values.data();
    const auto *ker2 = kernel_values.data() + MAX_NSPREAD;
    const auto *ker3 = kernel_values.data() + 2 * MAX_NSPREAD;
    // Combine kernel with complex source value to simplify inner loop
    // here 2* is because of complex
    // kerval_vectors is the number of SIMD iterations needed to compute all the elements
    static constexpr uint8_t kerval_vectors = (2 * ns + padding) / simd_size;
    static_assert(kerval_vectors > 0, "kerval_vectors must be greater than 0");
    const auto ker1val_v = [ker1, dd_pt]() constexpr noexcept {
      std::array<simd_type, kerval_vectors> ker1val_v{};
      // Iterate over kerval_vectors but in case the number of kerval_vectors is odd
      // we need to handle the last batch separately
      // to the & ~1 is to ensure that we do not iterate over the last batch if it is odd
      // as it sets the last bit to 0
      for (uint8_t i = 0; i < (kerval_vectors & ~1); // NOLINT(*-too-small-loop-variable
           i += 2) {
        const auto ker1_v  = simd_type::load_aligned(ker1 + i * simd_size / 2);
        const auto ker1low = xsimd::swizzle(ker1_v, zip_low_index<arch_t>);
        const auto ker1hi  = xsimd::swizzle(ker1_v, zip_hi_index<arch_t>);
        ker1val_v[i]       = ker1low * dd_pt;
        ker1val_v[i + 1]   = ker1hi * dd_pt;
      }
      // (at compile time) check if the number of kerval_vectors is odd
      // if it is we need to handle the last batch separately
      if constexpr (kerval_vectors % 2) {
        const auto ker1_v =
            simd_type::load_unaligned(ker1 + (kerval_vectors - 1) * simd_size / 2);
        const auto res = xsimd::swizzle(ker1_v, zip_low_index<arch_t>) * dd_pt;
        ker1val_v[kerval_vectors - 1] = res;
      }
      return ker1val_v;
    }();
    // critical inner loop:
    for (uint8_t dz{0}; dz < ns; ++dz) {
      const auto oz = size1 * size2 * (i3 - off3 + dz);           // offset due to z
      for (uint8_t dy{0}; dy < ns; ++dy) {
        const auto j = oz + size1 * (i2 - off2 + dy) + i1 - off1; // should be in subgrid
        auto *FINUFFT_RESTRICT trg = du + 2 * j;
        const simd_type kerval_v(ker2[dy] * ker3[dz]);
        for (uint8_t i{0}; i < kerval_vectors; ++i) {
          const auto trg_v  = simd_type::load_unaligned(trg + i * simd_size);
          const auto result = xsimd::fma(kerval_v, ker1val_v[i], trg_v);
          result.store_unaligned(trg + i * simd_size);
        }
      }
    }
  }
}

template<uint8_t NS>
void spread_subproblem_3d_dispatch(
    BIGINT off1, BIGINT off2, BIGINT off3, UBIGINT size1, UBIGINT size2, UBIGINT size3,
    FLT *du, UBIGINT M, const FLT *kx, const FLT *ky, const FLT *kz, const FLT *dd,
    const finufft_spread_opts &opts) noexcept {
  static_assert(MIN_NSPREAD <= NS && NS <= MAX_NSPREAD,
                "NS must be in the range (MIN_NSPREAD, MAX_NSPREAD)");
  if constexpr (NS == MIN_NSPREAD) { // Base case
    if (opts.kerevalmeth)
      return spread_subproblem_3d_kernel<MIN_NSPREAD, true>(
          off1, off2, off3, size1, size2, size3, du, M, kx, ky, kz, dd, opts);
    else {
      return spread_subproblem_3d_kernel<MIN_NSPREAD, false>(
          off1, off2, off3, size1, size2, size3, du, M, kx, ky, kz, dd, opts);
    }
  } else {
    if (opts.nspread == NS) {
      if (opts.kerevalmeth) {
        return spread_subproblem_3d_kernel<NS, true>(off1, off2, off3, size1, size2,
                                                     size3, du, M, kx, ky, kz, dd, opts);
      } else {
        return spread_subproblem_3d_kernel<NS, false>(off1, off2, off3, size1, size2,
                                                      size3, du, M, kx, ky, kz, dd, opts);
      }
    } else {
      return spread_subproblem_3d_dispatch<NS - 1>(off1, off2, off3, size1, size2, size3,
                                                   du, M, kx, ky, kz, dd, opts);
    }
  }
}

void spread_subproblem_3d(BIGINT off1, BIGINT off2, BIGINT off3, UBIGINT size1,
                          UBIGINT size2, UBIGINT size3, FLT *du, UBIGINT M, FLT *kx,
                          FLT *ky, FLT *kz, FLT *dd,
                          const finufft_spread_opts &opts) noexcept
/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
See above docs/notes for spread_subproblem_2d.
kx,ky,kz (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in each dim.
dd (size M complex) are complex source strengths
du (size size1*size2*size3) is uniform complex output array
*/
{
  spread_subproblem_3d_dispatch<MAX_NSPREAD>(off1, off2, off3, size1, size2, size3, du, M,
                                             kx, ky, kz, dd, opts);
}

template<bool thread_safe>
void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
                         UBIGINT padded_size1, UBIGINT size1, UBIGINT size2,
                         UBIGINT size3, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                         FLT *FINUFFT_RESTRICT data_uniform, const FLT *const du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   padded_size1,2,3 give the size of subgrid.
   Works in all dims. Thread-safe variant of the above routine,
   using atomic writes (R Blackwell, Nov 2020).
   Merged the thread_safe and the not thread_safe version of the function into one
   (M. Barbone 06/24).
*/
{
  std::vector<BIGINT> o2(size2), o3(size3);
  static auto accumulate = [](FLT &a, FLT b) {
    if constexpr (thread_safe) { // NOLINT(*-branch-clone)
#pragma omp atomic
      a += b;
    } else {
      a += b;
    }
  };

  BIGINT y = offset2, z = offset3; // fill wrapped ptr lists in slower dims y,z...
  for (int i = 0; i < size2; ++i) {
    if (y < 0) y += BIGINT(N2);
    if (y >= N2) y -= BIGINT(N2);
    o2[i] = y++;
  }
  for (int i = 0; i < size3; ++i) {
    if (z < 0) z += BIGINT(N3);
    if (z >= N3) z -= BIGINT(N3);
    o3[i] = z++;
  }
  UBIGINT nlo = (offset1 < 0) ? -offset1 : 0; // # wrapping below in x
  UBIGINT nhi = (offset1 + size1 > N1) ? offset1 + size1 - N1 : 0; // " above in x
  // this triple loop works in all dims
  for (int dz = 0; dz < size3; dz++) {               // use ptr lists in each axis
    const auto oz = N1 * N2 * o3[dz];                // offset due to z (0 in <3D)
    for (int dy = 0; dy < size2; dy++) {
      const auto oy              = N1 * o2[dy] + oz; // off due to y & z (0 in 1D)
      auto *FINUFFT_RESTRICT out = data_uniform + 2 * oy;
      const auto in = du0 + 2 * padded_size1 * (dy + size2 * dz); // ptr to subgrid array
      auto o        = 2 * (offset1 + N1);                         // 1d offset for output
      for (auto j = 0; j < 2 * nlo; j++) { // j is really dx/2 (since re,im parts)
        accumulate(out[j + o], in[j]);
      }
      o = 2 * offset1;
      for (auto j = 2 * nlo; j < 2 * (size1 - nhi); j++) {
        accumulate(out[j + o], in[j]);
      }
      o = 2 * (offset1 - N1);
      for (auto j = 2 * (size1 - nhi); j < 2 * size1; j++) {
        accumulate(out[j + o], in[j]);
      }
    }
  }
}

void bin_sort_singlethread(
    BIGINT *ret, const UBIGINT M, const FLT *kx, const FLT *ky, const FLT *kz,
    const UBIGINT N1, const UBIGINT N2, const UBIGINT N3, const double bin_size_x,
    const double bin_size_y, const double bin_size_z, const int debug)
/* Returns permutation of all nonuniform points with good RAM access,
 * ie less cache misses for spreading, in 1D, 2D, or 3D. Single-threaded version
 *
 * This is achieved by binning into cuboids (of given bin_size within the
 * overall box domain), then reading out the indices within
 * these bins in a Cartesian cuboid ordering (x fastest, y med, z slowest).
 * Finally the permutation is inverted, so that the good ordering is: the
 * NU pt of index ret[0], the NU pt of index ret[1],..., NU pt of index ret[M-1]
 *
 * Inputs: M - number of input NU points.
 *         kx,ky,kz - length-M arrays of real coords of NU pts in [-pi, pi).
 *                    Points outside this range are folded into it.
 *         N1,N2,N3 - integer sizes of overall box (N2=N3=1 for 1D, N3=1 for 2D)
 *         bin_size_x,y,z - what binning box size to use in each dimension
 *                    (in rescaled coords where ranges are [0,Ni] ).
 *                    For 1D, only bin_size_x is used; for 2D, it & bin_size_y.
 * Output:
 *         writes to ret a vector list of indices, each in the range 0,..,M-1.
 *         Thus, ret must have been preallocated for M BIGINTs.
 *
 * Notes: I compared RAM usage against declaring an internal vector and passing
 * back; the latter used more RAM and was slower.
 * Avoided the bins array, as in JFM's spreader of 2016,
 * tidied up, early 2017, Barnett.
 * Timings (2017): 3s for M=1e8 NU pts on 1 core of i7; 5s on 1 core of xeon.
 * Simplified by Martin Reinecke, 6/19/23 (no apparent effect on speed).
 */
{
  const auto isky = (N2 > 1), iskz = (N3 > 1); // ky,kz avail? (cannot access if not)
  // here the +1 is needed to allow round-off error causing i1=N1/bin_size_x,
  // for kx near +pi, ie foldrescale gives N1 (exact arith would be 0 to N1-1).
  // Note that round-off near kx=-pi stably rounds negative to i1=0.
  const auto nbins1         = BIGINT(FLT(N1) / bin_size_x + 1);
  const auto nbins2         = isky ? BIGINT(FLT(N2) / bin_size_y + 1) : 1;
  const auto nbins3         = iskz ? BIGINT(FLT(N3) / bin_size_z + 1) : 1;
  const auto nbins          = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x = FLT(1.0 / bin_size_x);
  const auto inv_bin_size_y = FLT(1.0 / bin_size_y);
  const auto inv_bin_size_z = FLT(1.0 / bin_size_z);
  // count how many pts in each bin
  std::vector<BIGINT> counts(nbins, 0);

  for (auto i = 0; i < M; i++) {
    // find the bin index in however many dims are needed
    const auto i1  = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2  = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3  = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
    ++counts[bin];
  }

  // compute the offsets directly in the counts array (no offset array)
  BIGINT current_offset = 0;
  for (BIGINT i = 0; i < nbins; i++) {
    BIGINT tmp = counts[i];
    counts[i]  = current_offset; // Reinecke's cute replacement of counts[i]
    current_offset += tmp;
  } // (counts now contains the index offsets for each bin)

  for (auto i = 0; i < M; i++) {
    // find the bin index (again! but better than using RAM)
    const auto i1    = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2    = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3    = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin   = i1 + nbins1 * (i2 + nbins2 * i3);
    ret[counts[bin]] = BIGINT(i); // fill the inverse map on the fly
    ++counts[bin];                // update the offsets
  }
}

void bin_sort_multithread(BIGINT *ret, UBIGINT M, FLT *kx, FLT *ky, FLT *kz, UBIGINT N1,
                          UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y,
                          double bin_size_z, int debug, int nthr)
/* Mostly-OpenMP'ed version of bin_sort.
   For documentation see: bin_sort_singlethread.
   Caution: when M (# NU pts) << N (# U pts), is SLOWER than single-thread.
   Originally by Barnett 2/8/18
   Explicit #threads control argument 7/20/20.
   Improved by Martin Reinecke, 6/19/23 (up to 50% faster at 1 thr/core).
   Todo: if debug, print timing breakdowns.
 */
{
  bool isky = (N2 > 1), iskz = (N3 > 1); // ky,kz avail? (cannot access if not)
  UBIGINT nbins1 = N1 / bin_size_x + 1, nbins2, nbins3; // see above note on why +1
  nbins2         = isky ? N2 / bin_size_y + 1 : 1;
  nbins3         = iskz ? N3 / bin_size_z + 1 : 1;
  UBIGINT nbins  = nbins1 * nbins2 * nbins3;
  if (nthr == 0)                       // should never happen in spreadinterp use
    fprintf(stderr, "[%s] nthr (%d) must be positive!\n", __func__, nthr);
  int nt = std::min(M, UBIGINT(nthr)); // handle case of less points than threads
  std::vector<UBIGINT> brk(nt + 1);    // list of start NU pt indices per thread

  // distribute the NU pts to threads once & for all...
  for (int t = 0; t <= nt; ++t)
    brk[t] = (UBIGINT)(0.5 + M * t / (double)nt); // start index for t'th chunk

  // set up 2d array (nthreads * nbins), just its pointers for now
  // (sub-vectors will be initialized later)
  std::vector<std::vector<UBIGINT>> counts(nt);

#pragma omp parallel num_threads(nt)
  { // parallel binning to each thread's count. Block done once per thread
    int t = MY_OMP_GET_THREAD_NUM(); // (we assume all nt threads created)
    auto &my_counts(counts[t]);      // name for counts[t]
    my_counts.resize(nbins, 0);      // allocate counts[t], now in parallel region
    for (auto i = brk[t]; i < brk[t + 1]; i++) {
      // find the bin index in however many dims are needed
      BIGINT i1 = fold_rescale(kx[i], N1) / bin_size_x, i2 = 0, i3 = 0;
      if (isky) i2 = fold_rescale(ky[i], N2) / bin_size_y;
      if (iskz) i3 = fold_rescale(kz[i], N3) / bin_size_z;
      const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
      ++my_counts[bin]; // no clash btw threads
    }
  }

  // inner sum along both bin and thread (inner) axes to get global offsets
  UBIGINT current_offset = 0;
  for (UBIGINT b = 0; b < nbins; ++b) // (not worth omp)
    for (int t = 0; t < nt; ++t) {
      UBIGINT tmp  = counts[t][b];
      counts[t][b] = current_offset;
      current_offset += tmp;
    } // counts[t][b] is now the index offset as if t ordered fast, b slow

#pragma omp parallel num_threads(nt)
  {
    int t = MY_OMP_GET_THREAD_NUM();
    auto &my_counts(counts[t]);
    for (UBIGINT i = brk[t]; i < brk[t + 1]; i++) {
      // find the bin index (again! but better than using RAM)
      UBIGINT i1 = fold_rescale(kx[i], N1) / bin_size_x, i2 = 0, i3 = 0;
      if (isky) i2 = fold_rescale(ky[i], N2) / bin_size_y;
      if (iskz) i3 = fold_rescale(kz[i], N3) / bin_size_z;
      UBIGINT bin         = i1 + nbins1 * (i2 + nbins2 * i3);
      ret[my_counts[bin]] = i; // inverse is offset for this NU pt and thread
      ++my_counts[bin];        // update the offsets; no thread clash
    }
  }
}

void get_subgrid(BIGINT &offset1, BIGINT &offset2, BIGINT &offset3, BIGINT &padded_size1,
                 BIGINT &size1, BIGINT &size2, BIGINT &size3, UBIGINT M, FLT *kx, FLT *ky,
                 FLT *kz, int ns, int ndims)
/* Writes out the integer offsets and sizes of a "subgrid" (cuboid subset of
   Z^ndims) large enough to enclose all of the nonuniform points with
   (non-periodic) padding of half the kernel width ns to each side in
   each relevant dimension.

 Inputs:
   M - number of nonuniform points, ie, length of kx array (and ky if ndims>1,
       and kz if ndims>2)
   kx,ky,kz - coords of nonuniform points (ky only read if ndims>1,
              kz only read if ndims>2). To be useful for spreading, they are
              assumed to be in [0,Nj] for dimension j=1,..,ndims.
   ns - (positive integer) spreading kernel width.
   ndims - space dimension (1,2, or 3).

 Outputs:
   offset1,2,3 - left-most coord of cuboid in each dimension (up to ndims)
   padded_size1,2,3   - size of cuboid in each dimension.
                 Thus the right-most coord of cuboid is offset+size-1.
   Returns offset 0 and size 1 for each unused dimension (ie when ndims<3);
   this is required by the calling code.

 Example:
      inputs:
          ndims=1, M=2, kx[0]=0.2, ks[1]=4.9, ns=3
      outputs:
          offset1=-1 (since kx[0] spreads to {-1,0,1}, and -1 is the min)
          padded_size1=8 (since kx[1] spreads to {4,5,6}, so subgrid is {-1,..,6}
                   hence 8 grid points).
 Notes:
   1) Works in all dims 1,2,3.
   2) Rounding of the kx (and ky, kz) to the grid is tricky and must match the
   rounding step used in spread_subproblem_{1,2,3}d. Namely, the ceil of
   (the NU pt coord minus ns/2) gives the left-most index, in each dimension.
   This being done consistently is crucial to prevent segfaults in subproblem
   spreading. This assumes that max() and ceil() commute in the floating pt
   implementation.
   Originally by J Magland, 2017. AHB realised the rounding issue in
   6/16/17, but only fixed a rounding bug causing segfault in (highly
   inaccurate) single-precision with N1>>1e7 on 11/30/20.
   3) Requires O(M) RAM reads to find the k array bnds. Almost negligible in
   tests.
*/
{
  FLT ns2 = (FLT)ns / 2;
  FLT min_kx, max_kx; // 1st (x) dimension: get min/max of nonuniform points
  arrayrange(M, kx, &min_kx, &max_kx);
  offset1      = (BIGINT)std::ceil(min_kx - ns2); // min index touched by kernel
  size1        = (BIGINT)std::ceil(max_kx - ns2) - offset1 + ns; // int(ceil) first!
  padded_size1 = size1 + get_padding<FLT>(2 * ns) / 2;
  if (ndims > 1) {
    FLT min_ky, max_ky; // 2nd (y) dimension: get min/max of nonuniform points
    arrayrange(M, ky, &min_ky, &max_ky);
    offset2 = (BIGINT)std::ceil(min_ky - ns2);
    size2   = (BIGINT)std::ceil(max_ky - ns2) - offset2 + ns;
  } else {
    offset2 = 0;
    size2   = 1;
  }
  if (ndims > 2) {
    FLT min_kz, max_kz; // 3rd (z) dimension: get min/max of nonuniform points
    arrayrange(M, kz, &min_kz, &max_kz);
    offset3 = (BIGINT)std::ceil(min_kz - ns2);
    size3   = (BIGINT)std::ceil(max_kz - ns2) - offset3 + ns;
  } else {
    offset3 = 0;
    size3   = 1;
  }
}
/* local NU coord fold+rescale macro: does the following affine transform to x:
    (x+PI) mod PI    each to [0,N)
   Note: folding big numbers can cause numerical inaccuracies
   Martin Reinecke, 8.5.2024 used floor to speedup the function and removed the range
   limitation Marco Barbone, 8.5.2024 Changed it from a Macro to an inline function
*/
FLT fold_rescale(const FLT x, const UBIGINT N) noexcept {
  static constexpr const FLT x2pi = FLT(M_1_2PI);
  const FLT result                = x * x2pi + FLT(0.5);
  return (result - floor(result)) * FLT(N);
}

template<class simd_type>
simd_type fold_rescale(const simd_type &x, const BIGINT N) noexcept {
  const simd_type x2pi   = FLT(M_1_2PI);
  const simd_type result = xsimd::fma(x, x2pi, simd_type(0.5));
  return (result - xsimd::floor(result)) * simd_type(FLT(N));
}

template<uint8_t ns, uint8_t kerevalmeth, class T, class simd_type, typename... V>
auto ker_eval(FLT *FINUFFT_RESTRICT ker, const finufft_spread_opts &opts,
              const V... elems) noexcept {
  /* Utility function that allows to move the kernel evaluation outside the spreader for
     clarity Inputs are: ns = kernel width kerevalmeth = kernel evaluation method T =
     (single or double precision) type of the kernel simd_type = xsimd::batch for Horner
     vectorization (default is the optimal simd size) finufft_spread_opts as Horner needs
     the oversampling factor elems = kernel arguments examples usage is ker_eval<ns,
     kerevalmeth>(opts, x, y, z) // for 3D or ker_eval<ns, kerevalmeth>(opts, x, y) // for
     2D or ker_eval<ns, kerevalmeth>(opts, x) // for 1D
   */
  const std::array inputs{elems...};
  // compile time loop, no performance overhead
  for (auto i = 0; i < sizeof...(elems); ++i) {
    // compile time branch no performance overhead
    if constexpr (kerevalmeth == 1) {
      if (opts.upsampfac == 2.0) {
        eval_kernel_vec_Horner<ns, 200, simd_type>(ker + (i * MAX_NSPREAD), inputs[i],
                                                   opts);
      }
      if (opts.upsampfac == 1.25) {
        eval_kernel_vec_Horner<ns, 125, simd_type>(ker + (i * MAX_NSPREAD), inputs[i],
                                                   opts);
      }
    }
    if constexpr (kerevalmeth == 0) {
      alignas(simd_type::arch_type::alignment()) std::array<T, MAX_NSPREAD> kernel_args{};
      set_kernel_args<ns>(kernel_args.data(), inputs[i]);
      evaluate_kernel_vector<ns>(ker + (i * MAX_NSPREAD), kernel_args.data(), opts);
    }
  }
  return ker;
}

namespace {

template<typename T, std::size_t N, std::size_t M, std::size_t PaddedM>
constexpr array<std::array<T, PaddedM>, N> pad_2D_array_with_zeros(
    const array<std::array<T, M>, N> &input) noexcept {
  constexpr auto pad_with_zeros = [](const auto &input) constexpr noexcept {
    std::array<T, PaddedM> padded{0};
    for (auto i = 0; i < input.size(); ++i) {
      padded[i] = input[i];
    }
    return padded;
  };
  std::array<std::array<T, PaddedM>, N> output{};
  for (std::size_t i = 0; i < N; ++i) {
    output[i] = pad_with_zeros(input[i]);
  }
  return output;
}

template<class T, class V, size_t... Is>
constexpr T generate_sequence_impl(V a, V b, index_sequence<Is...>) noexcept {
  // utility function to generate a sequence of a, b interleaved as function arguments
  return T(((Is % 2 == 0) ? a : b)...);
}

template<class T, class V, std::size_t N>
constexpr auto initialize_complex_register(V a, V b) noexcept {
  // populates a SIMD register with a and b interleaved
  // for example:
  // +-------------------------------+
  // | a | b | a | b | a | b | a | b |
  // +-------------------------------+
  // it uses index_sequence to generate the sequence of a, b at compile time
  return generate_sequence_impl<T>(a, b, std::make_index_sequence<N>{});
}

// Below there is some template metaprogramming magic to find the best SIMD type
// for the given number of elements. The code is based on the xsimd library

// this finds the largest SIMD instruction set that can handle N elements
// void otherwise -> compile error
template<class T, uint8_t N, uint8_t K> constexpr auto BestSIMDHelper() {
  if constexpr (N % K == 0) { // returns void in the worst case
    return xsimd::make_sized_batch<T, K>{};
  } else {
    return BestSIMDHelper<T, N, (K >> 1)>();
  }
}

template<class T, uint8_t N> constexpr uint8_t min_simd_width() {
  // finds the smallest simd width that can handle N elements
  // simd size is batch size the SIMD width in xsimd terminology
  if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
    return min_simd_width<T, N * 2>();
  } else {
    return N;
  }
};

template<class T, uint8_t N> constexpr auto find_optimal_simd_width() {
  // finds the smallest simd width that minimizes the number of iterations
  // NOTE: might be suboptimal for some cases 2^N+1 for example
  // in the future we might want to implement a more sophisticated algorithm
  uint8_t optimal_simd_width = min_simd_width<T>();
  uint8_t min_iterations     = (N + optimal_simd_width - 1) / optimal_simd_width;
  for (uint8_t simd_width = optimal_simd_width;
       simd_width <= xsimd::batch<T, xsimd::best_arch>::size;
       simd_width *= 2) {
    uint8_t iterations = (N + simd_width - 1) / simd_width;
    if (iterations < min_iterations) {
      min_iterations     = iterations;
      optimal_simd_width = simd_width;
    }
  }
  return optimal_simd_width;
}

template<class T, uint8_t N> constexpr auto GetPaddedSIMDWidth() {
  // helper function to get the SIMD width with padding for the given number of elements
  // that minimizes the number of iterations
  return xsimd::make_sized_batch<T, find_optimal_simd_width<T, N>()>::type::size;
}

template<class T, uint8_t ns> constexpr auto get_padding() {
  // helper function to get the padding for the given number of elements
  // ns is known at compile time, rounds ns to the next multiple of the SIMD width
  // then subtracts ns to get the padding using a bitwise and trick
  // WARING: this trick works only for power of 2s
  // SOURCE: Agner Fog's VCL manual
  constexpr uint8_t width = GetPaddedSIMDWidth<T, ns>();
  return ((ns + width - 1) & (-width)) - ns;
}

template<class T, uint8_t ns> constexpr auto get_padding_helper(uint8_t runtime_ns) {
  // helper function to get the padding for the given number of elements where ns is
  // known at runtime, it uses recursion to find the padding
  // this allows to avoid having a function with a large number of switch cases
  // as GetPaddedSIMDWidth requires a compile time value
  // it cannot be a lambda function because of the template recursion
  if constexpr (ns < 2) {
    return 0;
  } else {
    if (runtime_ns == ns) {
      return get_padding<T, ns>();
    } else {
      return get_padding_helper<T, ns - 1>(runtime_ns);
    }
  }
}

template<class T> uint8_t get_padding(uint8_t ns) {
  // return the padding as a function of the number of elements
  // 2 * MAX_NSPREAD is the maximum number of elements that we can have
  // that's why is hardcoded here
  return get_padding_helper<T, 2 * MAX_NSPREAD>(ns);
}

struct zip_low {
  // helper struct to get the lower half of a SIMD register and zip it with itself
  // it returns index 0, 0, 1, 1, ... N/2, N/2
  static constexpr unsigned get(unsigned index, unsigned /*size*/) { return index / 2; }
};
struct zip_hi {
  // helper struct to get the upper half of a SIMD register and zip it with itself
  // it returns index N/2, N/2, N/2+1, N/2+1, ... N, N
  static constexpr unsigned get(unsigned index, unsigned size) {
    return (size + index) / 2;
  }
};
template<unsigned cap> struct reverse_index {
  static constexpr unsigned get(unsigned index, const unsigned size) {
    return index < cap ? (cap - 1 - index) : index;
  }
};
template<unsigned cap> struct shuffle_index {
  static constexpr unsigned get(unsigned index, const unsigned size) {
    return index < cap ? (cap - 1 - index) : size + size + cap - 1 - index;
  }
};

struct select_even {
  static constexpr unsigned get(unsigned index, unsigned /*size*/) { return index * 2; }
};
struct select_odd {
  static constexpr unsigned get(unsigned index, unsigned /*size*/) {
    return index * 2 + 1;
  }
};

template<typename T> auto xsimd_to_array(const T &vec) noexcept {
  constexpr auto alignment = T::arch_type::alignment();
  alignas(alignment) std::array<typename T::value_type, T::size> array{};
  vec.store_aligned(array.data());
  return array;
}

void print_subgrid_info(int ndims, BIGINT offset1, BIGINT offset2, BIGINT offset3,
                        UBIGINT padded_size1, UBIGINT size1, UBIGINT size2, UBIGINT size3,
                        UBIGINT M0) {
  printf("size1 %ld, padded_size1 %ld\n", size1, padded_size1);
  switch (ndims) {
  case 1:
    printf("\tsubgrid: off %lld\t siz %lld\t #NU %lld\n", (long long)offset1,
           (long long)padded_size1, (long long)M0);
    break;
  case 2:
    printf("\tsubgrid: off %lld,%lld\t siz %lld,%lld\t #NU %lld\n", (long long)offset1,
           (long long)offset2, (long long)padded_size1, (long long)size2, (long long)M0);
    break;
  case 3:
    printf("\tsubgrid: off %lld,%lld,%lld\t siz %lld,%lld,%lld\t #NU %lld\n",
           (long long)offset1, (long long)offset2, (long long)offset3,
           (long long)padded_size1, (long long)size2, (long long)size3, (long long)M0);
    break;
  default:
    printf("Invalid number of dimensions: %d\n", ndims);
    break;
  }
}
} // namespace
} // namespace finufft::spreadinterp
