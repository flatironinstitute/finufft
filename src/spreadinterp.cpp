// Spreading/interpolating module within FINUFFT. Uses precision-switching
// macros for FLT, CPX, etc.

#include <finufft/spreadinterp.h>
#include <finufft/defs.h>
#include <finufft/utils.h>
#include <finufft/utils_precindep.h>

#include <xsimd/xsimd.hpp>

#include <stdlib.h>
#include <vector>
#include <math.h>
#include <stdio.h>


using namespace std;
using namespace finufft::utils;              // access to timer

namespace finufft {
  namespace spreadinterp {



// forward declaration to cleanup the code and be able to use this everywhere in shit file

template<class T, uint16_t N, uint16_t K = N>
static constexpr auto BestSIMDHelper();

template<class T, uint16_t N>
static constexpr auto GetPaddedSIMDSize();

template<class T, uint16_t N>
using PaddedSIMD = typename xsimd::make_sized_batch<T, GetPaddedSIMDSize<T, N>()>::type;

template<class T>
static uint16_t get_padding(uint16_t ns);

template<class T, uint16_t ns>
static constexpr auto get_padding();

template<class T, uint16_t N>
using BestSIMD = typename decltype(BestSIMDHelper<T, N, xsimd::batch<T>::size>())::type;

template<class T, uint16_t N = 1>
static constexpr uint16_t min_batch_size();

template<class T, uint16_t N, uint16_t batch_size = min_batch_size<T>(), uint16_t min_iterations = N, uint16_t optimal_batch_size = 1>
static constexpr uint16_t find_optimal_batch_size();


// declarations of purely internal functions... (thus need not be in .h)
static FINUFFT_ALWAYS_INLINE FLT fold_rescale(FLT x, BIGINT N) noexcept;
static FINUFFT_ALWAYS_INLINE void set_kernel_args(FLT *args, FLT x, const finufft_spread_opts& opts) noexcept;
static FINUFFT_ALWAYS_INLINE void evaluate_kernel_vector(FLT *ker, FLT *args, const finufft_spread_opts& opts, int N) noexcept;
static FINUFFT_ALWAYS_INLINE void eval_kernel_vec_Horner(FLT *ker, FLT x, int w, const finufft_spread_opts &opts) noexcept;
template<uint16_t w> // aka ns
static FINUFFT_ALWAYS_INLINE void eval_kernel_vec_Horner(FLT * __restrict__ ker, FLT x, const finufft_spread_opts &opts) noexcept;
void interp_line(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns);
void interp_square(FLT *out,FLT *du, FLT *ker1, FLT *ker2, BIGINT i1,BIGINT i2,BIGINT N1,BIGINT N2,int ns);
void interp_cube(FLT *out,FLT *du, FLT *ker1, FLT *ker2, FLT *ker3,
		 BIGINT i1,BIGINT i2,BIGINT i3,BIGINT N1,BIGINT N2,BIGINT N3,int ns);
void spread_subproblem_1d(BIGINT off1, BIGINT size1,FLT *du0,BIGINT M0,FLT *kx0,
                          FLT *dd0,const finufft_spread_opts& opts) noexcept;
void spread_subproblem_2d(BIGINT off1, BIGINT off2, BIGINT size1, BIGINT size2,
                          FLT * __restrict__ du, BIGINT M, const FLT *kx, const FLT *ky, const FLT *dd,
                          const finufft_spread_opts &opts) noexcept;
void spread_subproblem_3d(BIGINT off1,BIGINT off2, BIGINT off3, BIGINT size1,
                          BIGINT size2,BIGINT size3,FLT *du0,BIGINT M0,
			                    FLT *kx0,FLT *ky0,FLT *kz0,FLT *dd0,
			                    const finufft_spread_opts& opts) noexcept;

template<bool thread_safe>
void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
                         BIGINT padded_size1, BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
                         BIGINT N2, BIGINT N3, FLT * __restrict__ data_uniform,
                         const FLT * du0);

void bin_sort_singlethread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,
	      double bin_size_x,double bin_size_y,double bin_size_z, int debug);
void bin_sort_multithread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,
              double bin_size_x,double bin_size_y,double bin_size_z, int debug,
              int nthr);
void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &padded_size1, BIGINT &size1,
		 BIGINT &size2,BIGINT &size3,BIGINT M0,FLT* kx0,FLT* ky0,
		 FLT* kz0,int ns, int ndims);



// ==========================================================================
int spreadinterp(
        BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
        BIGINT M, FLT *kx, FLT *ky, FLT *kz, FLT *data_nonuniform,
        finufft_spread_opts opts)
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
  if (ier)
    return ier;
  BIGINT* sort_indices = (BIGINT*)malloc(sizeof(BIGINT)*M);
  if (!sort_indices) {
    fprintf(stderr,"%s failed to allocate sort_indices!\n",__func__);
    return FINUFFT_ERR_SPREAD_ALLOC;
  }
  int did_sort = indexSort(sort_indices, N1, N2, N3, M, kx, ky, kz, opts);
  spreadinterpSorted(sort_indices, N1, N2, N3, data_uniform,
                     M, kx, ky, kz, data_nonuniform, opts, did_sort);
  free(sort_indices);
  return 0;
}

static int ndims_from_Ns(BIGINT N1, BIGINT N2, BIGINT N3)
/* rule for getting number of spreading dimensions from the list of Ns per dim.
   Split out, Barnett 7/26/18
*/
{
  int ndims = 1;                // decide ndims: 1,2 or 3
  if (N2>1) ++ndims;
  if (N3>1) ++ndims;
  return ndims;
}

int spreadcheck(BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, FLT *kx, FLT *ky,
                FLT *kz, finufft_spread_opts opts)
/* This does just the input checking and reporting for the spreader.
   See spreadinterp() for input arguments and meaning of returned value.
   Split out by Melody Shih, Jun 2018. Finiteness chk Barnett 7/30/18.
   Marco Barbone 5.8.24 removed bounds check as new foldrescale is not limited to [-3pi,3pi)
*/
{
  // INPUT CHECKING & REPORTING .... cuboid not too small for spreading?
  int minN = 2*opts.nspread;
  if (N1<minN || (N2>1 && N2<minN) || (N3>1 && N3<minN)) {
    fprintf(stderr,"%s error: one or more non-trivial box dims is less than 2.nspread!\n",__func__);
    return FINUFFT_ERR_SPREAD_BOX_SMALL;
  }
  if (opts.spread_direction!=1 && opts.spread_direction!=2) {
    fprintf(stderr,"%s error: opts.spread_direction must be 1 or 2!\n",__func__);
    return FINUFFT_ERR_SPREAD_DIR;
  }
  return 0;
}


int indexSort(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, 
               FLT *kx, FLT *ky, FLT *kz, finufft_spread_opts opts)
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
  CNTime timer;
  int ndims = ndims_from_Ns(N1,N2,N3);
  BIGINT N=N1*N2*N3;            // U grid (periodic box) sizes
  
  // heuristic binning box size for U grid... affects performance:
  double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;
  // put in heuristics based on cache sizes (only useful for single-thread) ?

  int better_to_sort = !(ndims==1 && (opts.spread_direction==2 || (M > 1000*N1))); // 1D small-N or dir=2 case: don't sort

  timer.start();                 // if needed, sort all the NU pts...
  int did_sort=0;
  int maxnthr = MY_OMP_GET_MAX_THREADS();  // used if both below opts default
  if (opts.nthreads>0)
    maxnthr = opts.nthreads;         // user nthreads overrides, without limit
  if (opts.sort_threads>0)
    maxnthr = opts.sort_threads;     // high-priority override, also no limit
  // At this point: maxnthr = the max threads sorting could use
  // (we don't print warning here, since: no showwarn in spread_opts, and finufft
  // already warned about it. spreadinterp-only advanced users will miss a warning)
  if (opts.sort==1 || (opts.sort==2 && better_to_sort)) {
    // store a good permutation ordering of all NU pts (dim=1,2 or 3)
    int sort_debug = (opts.debug>=2);    // show timing output?
    int sort_nthr = opts.sort_threads;   // 0, or user max # threads for sort
#ifndef _OPENMP
    sort_nthr = 1;                       // if single-threaded lib, override user
#endif
    if (sort_nthr==0)   // multithreaded auto choice: when N>>M, one thread is better!
      sort_nthr = (10*M>N) ? maxnthr : 1;     // heuristic
    if (sort_nthr==1)
      bin_sort_singlethread(sort_indices,M,kx,ky,kz,N1,N2,N3,bin_size_x,bin_size_y,bin_size_z,sort_debug);
    else                                      // sort_nthr>1, user fixes # threads (>=2)
      bin_sort_multithread(sort_indices,M,kx,ky,kz,N1,N2,N3,bin_size_x,bin_size_y,bin_size_z,sort_debug,sort_nthr);
    if (opts.debug) 
      printf("\tsorted (%d threads):\t%.3g s\n",sort_nthr,timer.elapsedsec());
    did_sort=1;
  } else {
#pragma omp parallel for num_threads(maxnthr) schedule(static,1000000)
    for (BIGINT i=0; i<M; i++)                // here omp helps xeon, hinders i7
      sort_indices[i]=i;                      // the identity permutation
    if (opts.debug)
      printf("\tnot sorted (sort=%d): \t%.3g s\n",(int)opts.sort,timer.elapsedsec());
  }
  return did_sort;
}


int spreadinterpSorted(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, finufft_spread_opts opts, int did_sort)
/* Logic to select the main spreading (dir=1) vs interpolation (dir=2) routine.
   See spreadinterp() above for inputs arguments and definitions.
   Return value should always be 0 (no error reporting).
   Split out by Melody Shih, Jun 2018; renamed Barnett 5/20/20.
*/
{
  if (opts.spread_direction==1)  // ========= direction 1 (spreading) =======
    spreadSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts, did_sort);
  
  else           // ================= direction 2 (interpolation) ===========
    interpSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts, did_sort);
  
  return 0;
}


// --------------------------------------------------------------------------
int spreadSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, finufft_spread_opts opts, int did_sort)
// Spread NU pts in sorted order to a uniform grid. See spreadinterp() for doc.
{
  CNTime timer;
  int ndims = ndims_from_Ns(N1,N2,N3);
  BIGINT N=N1*N2*N3;            // output array size
  int ns=opts.nspread;          // abbrev. for w, kernel width
  int nthr = MY_OMP_GET_MAX_THREADS();  // guess # threads to use to spread
  if (opts.nthreads>0)
    nthr = opts.nthreads;       // user override, now without limit
#ifndef _OPENMP
  nthr = 1;                   // single-threaded lib must override user
#endif
  if (opts.debug)
    printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d\n",ndims,(long long)M,(long long)N1,(long long)N2,(long long)N3,nthr);
  
  timer.start();
  for (BIGINT i=0; i<2*N; i++) // zero the output array. std::fill is no faster
    data_uniform[i]=0.0;
  if (opts.debug) printf("\tzero output array\t%.3g s\n",timer.elapsedsec());
  if (M==0)                     // no NU pts, we're done
    return 0;
  
  int spread_single = (nthr==1) || (M*100<N);     // low-density heuristic?
  spread_single = 0;                 // for now
  timer.start();
  if (spread_single) {    // ------- Basic single-core t1 spreading ------
    for (BIGINT j=0; j<M; j++) {
      // *** todo, not urgent
      // ... (question is: will the index wrapping per NU pt slow it down?)
    }
    if (opts.debug) printf("\tt1 simple spreading:\t%.3g s\n",timer.elapsedsec());
    
  } else {           // ------- Fancy multi-core blocked t1 spreading ----
                     // Splits sorted inds (jfm's advanced2), could double RAM.
    // choose nb (# subprobs) via used nthreads:
    int nb = min((BIGINT)nthr,M);         // simply split one subprob per thr...
    if (nb*(BIGINT)opts.max_subproblem_size<M) {  // ...or more subprobs to cap size
      nb = 1 + (M-1)/opts.max_subproblem_size;  // int div does ceil(M/opts.max_subproblem_size)
      if (opts.debug) printf("\tcapping subproblem sizes to max of %d\n",opts.max_subproblem_size);
    }
    if (M*1000<N) {         // low-density heuristic: one thread per NU pt!
      nb = M;
      if (opts.debug) printf("\tusing low-density speed rescue nb=M...\n");
    }
    if (!did_sort && nthr==1) {
      nb = 1;
      if (opts.debug) printf("\tunsorted nthr=1: forcing single subproblem...\n");
    }
    if (opts.debug && nthr>opts.atomic_threshold)
      printf("\tnthr big: switching add_wrapped OMP from critical to atomic (!)\n");
      
    std::vector<BIGINT> brk(nb+1); // NU index breakpoints defining nb subproblems
    for (int p=0;p<=nb;++p)
      brk[p] = (BIGINT)(0.5 + M*p/(double)nb);
    
#pragma omp parallel for num_threads(nthr) schedule(dynamic,1)  // each is big
      for (int isub=0; isub<nb; isub++) {   // Main loop through the subproblems
        BIGINT M0 = brk[isub+1]-brk[isub];  // # NU pts in this subproblem
        // copy the location and data vectors for the nonuniform points
        FLT *kx0=(FLT*)malloc(sizeof(FLT)*M0), *ky0=NULL, *kz0=NULL;
        if (N2>1)
          ky0=(FLT*)malloc(sizeof(FLT)*M0);
        if (N3>1)
          kz0=(FLT*)malloc(sizeof(FLT)*M0);
        FLT *dd0=(FLT*)malloc(sizeof(FLT)*M0*2);    // complex strength data
        for (BIGINT j=0; j<M0; j++) {           // todo: can avoid this copying?
          BIGINT kk=sort_indices[j+brk[isub]];  // NU pt from subprob index list
          kx0[j]= fold_rescale(kx[kk], N1);
          if (N2>1) ky0[j]= fold_rescale(ky[kk], N2);
          if (N3>1) kz0[j]= fold_rescale(kz[kk], N3);
          dd0[j*2]=data_nonuniform[kk*2];     // real part
          dd0[j*2+1]=data_nonuniform[kk*2+1]; // imag part
        }
        // get the subgrid which will include padding by roughly nspread/2
        BIGINT offset1,offset2,offset3,padded_size1,size1,size2,size3; // get_subgrid sets
        get_subgrid(offset1, offset2, offset3, padded_size1, size1, size2, size3, M0, kx0, ky0, kz0, ns, ndims);  // sets offsets and sizes
        if (opts.debug>1) { // verbose
          printf("size1 %ld, padded_size1 %ld\n",size1, padded_size1);
          if (ndims==1)
            printf("\tsubgrid: off %lld\t siz %lld\t #NU %lld\n", (long long)offset1, (long long)padded_size1, (long long)M0);
          else if (ndims==2)
            printf("\tsubgrid: off %lld,%lld\t siz %lld,%lld\t #NU %lld\n", (long long)offset1, (long long)offset2, (long long)padded_size1, (long long)size2, (long long)M0);
          else
            printf("\tsubgrid: off %lld,%lld,%lld\t siz %lld,%lld,%lld\t #NU %lld\n", (long long)offset1, (long long)offset2, (long long)offset3, (long long)padded_size1, (long long)size2, (long long)size3, (long long)M0);
	}
        // allocate output data for this subgrid
        FLT *du0=(FLT*)malloc(sizeof(FLT) * 2 * padded_size1 * size2 * size3); // complex
        
        // Spread to subgrid without need for bounds checking or wrapping
        if (!(opts.flags & TF_OMIT_SPREADING)) {
          if (ndims==1)
            spread_subproblem_1d(offset1, padded_size1, du0, M0, kx0, dd0, opts);
          else if (ndims==2)
            spread_subproblem_2d(offset1, offset2, padded_size1, size2, du0, M0, kx0, ky0, dd0, opts);
          else
            spread_subproblem_3d(offset1, offset2, offset3, padded_size1, size2, size3, du0, M0, kx0, ky0, kz0, dd0, opts);
	}
        
        // do the adding of subgrid to output
        if (!(opts.flags & TF_OMIT_WRITE_TO_GRID)) {
          if (nthr > opts.atomic_threshold)   // see above for debug reporting
            add_wrapped_subgrid<true>(offset1, offset2, offset3, padded_size1, size1, size2, size3, N1, N2, N3, data_uniform, du0);   // R Blackwell's atomic version
          else {
#pragma omp critical
            add_wrapped_subgrid<false>(offset1, offset2, offset3, padded_size1, size1, size2, size3, N1, N2, N3, data_uniform, du0);
          }
        }

        // free up stuff from this subprob... (that was malloc'ed by hand)
        free(dd0);
        free(du0);
        free(kx0);
        if (N2>1) free(ky0);
        if (N3>1) free(kz0); 
      }     // end main loop over subprobs
      if (opts.debug) printf("\tt1 fancy spread: \t%.3g s (%d subprobs)\n",timer.elapsedsec(), nb);
    }   // end of choice of which t1 spread type to use
    return 0;
};


// --------------------------------------------------------------------------
int interpSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, finufft_spread_opts opts, int did_sort)
// Interpolate to NU pts in sorted order from a uniform grid.
// See spreadinterp() for doc.
{
  CNTime timer;
  int ndims = ndims_from_Ns(N1,N2,N3);
  int ns=opts.nspread;          // abbrev. for w, kernel width
  FLT ns2 = (FLT)ns/2;          // half spread width, used as stencil shift
  int nthr = MY_OMP_GET_MAX_THREADS();   // guess # threads to use to interp
  if (opts.nthreads>0)
    nthr = opts.nthreads;       // user override, now without limit
#ifndef _OPENMP
  nthr = 1;                   // single-threaded lib must override user
#endif
  if (opts.debug)
    printf("\tinterp %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d\n",ndims,(long long)M,(long long)N1,(long long)N2,(long long)N3,nthr);

  timer.start();  
#pragma omp parallel num_threads(nthr)
  {
#define CHUNKSIZE 16     // Chunks of Type 2 targets (Ludvig found by expt)
    BIGINT jlist[CHUNKSIZE];
    FLT xjlist[CHUNKSIZE], yjlist[CHUNKSIZE], zjlist[CHUNKSIZE];
    FLT outbuf[2*CHUNKSIZE];
    // Kernels: static alloc is faster, so we do it for up to 3D...
    FLT kernel_args[3*MAX_NSPREAD];
    FLT kernel_values[3*MAX_NSPREAD];
    FLT *ker1 = kernel_values;
    FLT *ker2 = kernel_values + ns;
    FLT *ker3 = kernel_values + 2*ns;       

    // Loop over interpolation chunks
#pragma omp for schedule (dynamic,1000)  // assign threads to NU targ pts:
    for (BIGINT i=0; i<M; i+=CHUNKSIZE)  // main loop over NU targs, interp each from U
      {
        // Setup buffers for this chunk
        int bufsize = (i+CHUNKSIZE > M) ? M-i : CHUNKSIZE;
        for (int ibuf=0; ibuf<bufsize; ibuf++) {
          BIGINT j = sort_indices[i+ibuf];
          jlist[ibuf] = j;
	  xjlist[ibuf] = fold_rescale(kx[j], N1);
	  if(ndims >=2)
	    yjlist[ibuf] = fold_rescale(ky[j], N2);
	  if(ndims == 3)
	    zjlist[ibuf] = fold_rescale(kz[j], N3);
	}
      
    // Loop over targets in chunk
    for (int ibuf=0; ibuf<bufsize; ibuf++) {
      FLT xj = xjlist[ibuf];
      FLT yj = (ndims > 1) ? yjlist[ibuf] : 0;
      FLT zj = (ndims > 2) ? zjlist[ibuf] : 0;

      FLT *target = outbuf+2*ibuf;
        
      // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
      BIGINT i1=(BIGINT)std::ceil(xj-ns2); // leftmost grid index
      BIGINT i2= (ndims > 1) ? (BIGINT)std::ceil(yj-ns2) : 0; // min y grid index
      BIGINT i3= (ndims > 2) ? (BIGINT)std::ceil(zj-ns2) : 0; // min z grid index
     
      FLT x1=(FLT)i1-xj;           // shift of ker center, in [-w/2,-w/2+1]
      FLT x2= (ndims > 1) ? (FLT)i2-yj : 0 ;
      FLT x3= (ndims > 2)? (FLT)i3-zj : 0;

      // eval kernel values patch and use to interpolate from uniform data...
      if (!(opts.flags & TF_OMIT_SPREADING)) {

	  if (opts.kerevalmeth==0) {               // choose eval method
	    set_kernel_args(kernel_args, x1, opts);
	    if(ndims > 1)  set_kernel_args(kernel_args+ns, x2, opts);
	    if(ndims > 2)  set_kernel_args(kernel_args+2*ns, x3, opts);
	    
	    evaluate_kernel_vector(kernel_values, kernel_args, opts, ndims*ns);
	  }

	  else{
	    eval_kernel_vec_Horner(ker1,x1,ns,opts);
	    if (ndims > 1) eval_kernel_vec_Horner(ker2,x2,ns,opts);  
	    if (ndims > 2) eval_kernel_vec_Horner(ker3,x3,ns,opts);
	  }

	  switch(ndims){
	  case 1:
	    interp_line(target,data_uniform,ker1,i1,N1,ns);
	    break;
	  case 2:
	    interp_square(target,data_uniform,ker1,ker2,i1,i2,N1,N2,ns);
	    break;
	  case 3:
	    interp_cube(target,data_uniform,ker1,ker2,ker3,i1,i2,i3,N1,N2,N3,ns);
	    break;
	  default: //can't get here
	    break;
	     
	  }	 
      }
    } // end loop over targets in chunk
        
    // Copy result buffer to output array
    for (int ibuf=0; ibuf<bufsize; ibuf++) {
      BIGINT j = jlist[ibuf];
      data_nonuniform[2*j] = outbuf[2*ibuf];
      data_nonuniform[2*j+1] = outbuf[2*ibuf+1];              
    }         
        
      } // end NU targ loop
  } // end parallel section
  if (opts.debug) printf("\tt2 spreading loop: \t%.3g s\n",timer.elapsedsec());
  return 0;
};



///////////////////////////////////////////////////////////////////////////

int setup_spreader(finufft_spread_opts &opts, FLT eps, double upsampfac,
                   int kerevalmeth, int debug, int showwarn, int dim)
/* Initializes spreader kernel parameters given desired NUFFT tolerance eps,
   upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), ker eval meth
   (either 0:exp(sqrt()), 1: Horner ppval), and some debug-level flags.
   Also sets all default options in finufft_spread_opts. See finufft_spread_opts.h for opts.
   dim is spatial dimension (1,2, or 3).
   See finufft.cpp:finufft_plan() for where upsampfac is set.
   Must call this before any kernel evals done, otherwise segfault likely.
   Returns:
     0  : success
     FINUFFT_WARN_EPS_TOO_SMALL : requested eps cannot be achieved, but proceed with
                          best possible eps
     otherwise : failure (see codes in defs.h); spreading must not proceed
   Barnett 2017. debug, loosened eps logic 6/14/20.
*/
{
  if (upsampfac!=2.0 && upsampfac!=1.25) {   // nonstandard sigma
    if (kerevalmeth==1) {
      fprintf(stderr,"FINUFFT setup_spreader: nonstandard upsampfac=%.3g cannot be handled by kerevalmeth=1\n",upsampfac);
      return FINUFFT_ERR_HORNER_WRONG_BETA;
    }
    if (upsampfac<=1.0) {       // no digits would result
      fprintf(stderr,"FINUFFT setup_spreader: error, upsampfac=%.3g is <=1.0\n",upsampfac);
      return FINUFFT_ERR_UPSAMPFAC_TOO_SMALL;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (showwarn && upsampfac>4.0)
      fprintf(stderr,"FINUFFT setup_spreader warning: upsampfac=%.3g way too large to be beneficial.\n",upsampfac);
  }
    
  // write out default finufft_spread_opts (some overridden in setup_spreader_for_nufft)
  opts.spread_direction = 0;    // user should always set to 1 or 2 as desired
  opts.sort = 2;                // 2:auto-choice
  opts.kerpad = 0;              // affects only evaluate_kernel_vector
  opts.kerevalmeth = kerevalmeth;
  opts.upsampfac = upsampfac;
  opts.nthreads = 0;            // all avail
  opts.sort_threads = 0;        // 0:auto-choice
  // heuristic dir=1 chunking for nthr>>1, typical for intel i7 and skylake...
  opts.max_subproblem_size = (dim==1) ? 10000 : 100000;
  opts.flags = 0;               // 0:no timing flags (>0 for experts only)
  opts.debug = 0;               // 0:no debug output
  // heuristic nthr above which switch OMP critical to atomic (add_wrapped...):
  opts.atomic_threshold = 10;   // R Blackwell's value

  int ns, ier = 0;  // Set kernel width w (aka ns, nspread) then copy to opts...
  if (eps<EPSILON) {            // safety; there's no hope of beating e_mach
    if (showwarn)
      fprintf(stderr,"%s warning: increasing tol=%.3g to eps_mach=%.3g.\n",__func__,(double)eps,(double)EPSILON);
    eps = EPSILON;              // only changes local copy (not any opts)
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }
  if (upsampfac==2.0)           // standard sigma (see SISC paper)
    ns = std::ceil(-log10(eps/(FLT)10.0));          // 1 digit per power of 10
  else                          // custom sigma
    ns = std::ceil(-log(eps) / (PI*sqrt(1.0-1.0/upsampfac)));  // formula, gam=1
  ns = max(2,ns);               // (we don't have ns=1 version yet)
  if (ns>MAX_NSPREAD) {         // clip to fit allocated arrays, Horner rules
    if (showwarn)
      fprintf(stderr,"%s warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d.\n",__func__,
              upsampfac,(double)eps,ns,MAX_NSPREAD);
    ns = MAX_NSPREAD;
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }
  opts.nspread = ns;
  // setup for reference kernel eval (via formula): select beta width param...
  // (even when kerevalmeth=1, this ker eval needed for FTs in onedim_*_kernel)
  opts.ES_halfwidth=(double)ns/2;   // constants to help (see below routines)
  opts.ES_c = 4.0/(double)(ns*ns);
  double betaoverns = 2.30;         // gives decent betas for default sigma=2.0
  if (ns==2) betaoverns = 2.20;  // some small-width tweaks...
  if (ns==3) betaoverns = 2.26;
  if (ns==4) betaoverns = 2.38;
  if (upsampfac!=2.0) {          // again, override beta for custom sigma
    FLT gamma=0.97;              // must match devel/gen_all_horner_C_code.m !
    betaoverns = gamma*PI*(1.0-1.0/(2*upsampfac));  // formula based on cutoff
  }
  opts.ES_beta = betaoverns * ns;   // set the kernel beta parameter
  if (debug)
    printf("%s (kerevalmeth=%d) eps=%.3g sigma=%.3g: chose ns=%d beta=%.3g\n",__func__,kerevalmeth,(double)eps,upsampfac,ns,opts.ES_beta);
  
  return ier;
}

FLT evaluate_kernel(FLT x, const finufft_spread_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg finufft/onedim_* 2/17/17
*/
{
  if (abs(x)>=(FLT)opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp((FLT)opts.ES_beta * sqrt((FLT)1.0 - (FLT)opts.ES_c*x*x));
}

static inline void set_kernel_args(FLT *args, FLT x, const finufft_spread_opts& opts) noexcept
// Fills vector args[] with kernel arguments x, x+1, ..., x+ns-1.
// needed for the vectorized kernel eval of Ludvig af K.
{
  int ns=opts.nspread;
  for (int i=0; i<ns; i++)
    args[i] = x + (FLT) i;
}

static inline void evaluate_kernel_vector(FLT *ker, FLT *args, const finufft_spread_opts& opts, const int N) noexcept
/* Evaluate ES kernel for a vector of N arguments; by Ludvig af K.
   If opts.kerpad true, args and ker must be allocated for Npad, and args is
   written to (to pad to length Npad), only first N outputs are correct.
   Barnett 4/24/18 option to pad to mult of 4 for better SIMD vectorization.

   Obsolete (replaced by Horner), but keep around for experimentation since
   works for arbitrary beta. Formula must match reference implementation. */
{
  FLT b = (FLT)opts.ES_beta;
  FLT c = (FLT)opts.ES_c;
  if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
    // Note (by Ludvig af K): Splitting kernel evaluation into two loops
    // seems to benefit auto-vectorization.
    // gcc 5.4 vectorizes first loop; gcc 7.2 vectorizes both loops
    int Npad = N;
    if (opts.kerpad) {        // since always same branch, no speed hit
      Npad = 4*(1+(N-1)/4);   // pad N to mult of 4; help i7 GCC, not xeon
      for (int i=N;i<Npad;++i)    // pad with 1-3 zeros for safe eval
	args[i] = 0.0;
    }
    for (int i = 0; i < Npad; i++) { // Loop 1: Compute exponential arguments
      ker[i] = b * sqrt((FLT)1.0 - c*args[i]*args[i]);  // care! 1.0 is double
    }
    if (!(opts.flags & TF_OMIT_EVALUATE_EXPONENTIAL))
      for (int i = 0; i < Npad; i++) // Loop 2: Compute exponentials
	ker[i] = exp(ker[i]);
  } else {
    for (int i = 0; i < N; i++)             // dummy for timing only
      ker[i] = 1.0;
  }
  // Separate check from arithmetic (Is this really needed? doesn't slow down)
  for (int i = 0; i < N; i++)
    if (abs(args[i])>=(FLT)opts.ES_halfwidth) ker[i] = 0.0;
}

template<uint16_t w> // aka ns
static inline void eval_kernel_vec_Horner(FLT *__restrict__ ker, const FLT x,
                       const finufft_spread_opts &opts) noexcept
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
This is the current evaluation method, since it's faster (except i7 w=16).
Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
  const FLT z = std::fma(FLT(2.0),x, FLT(w-1)); // scale so local grid offset z in [-1,1]
  // insert the auto-generated code which expects z, w args, writes to ker...
  if (opts.upsampfac==2.0) {     // floating point equality is fine here
#include "ker_horner_allw_loop_constexpr.c"
  } else if (opts.upsampfac==1.25) {
#include "ker_lowupsampfac_horner_allw_loop_constexpr.c"
  }
}

static inline void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w,
					  const finufft_spread_opts &opts) noexcept
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
  if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
    const FLT z = std::fma(FLT(2.0),x, FLT(w-1)); // scale so local grid offset z in [-1,1]
    // insert the auto-generated code which expects z, w args, writes to ker...
    if (opts.upsampfac==2.0) {     // floating point equality is fine here
#include "ker_horner_allw_loop.c"
    } else if (opts.upsampfac==1.25) {
#include "ker_lowupsampfac_horner_allw_loop.c"
    } else
      fprintf(stderr,"%s: unknown upsampfac, failed!\n",__func__);
  }
}

void interp_line(FLT *target,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns)
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
*/
{
  FLT out[] = {0.0, 0.0};
  BIGINT j = i1;
  if (i1<0) {                               // wraps at left
    j+=N1;
    for (int dx=0; dx<-i1; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
    j-=N1;
    for (int dx=-i1; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  } else if (i1+ns>=N1) {                    // wraps at right
    for (int dx=0; dx<N1-i1; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
    j-=N1;
    for (int dx=N1-i1; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  } else {                                     // doesn't wrap
    for (int dx=0; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

void interp_square(FLT *target,FLT *du, FLT *ker1, FLT *ker2, BIGINT i1,BIGINT i2,BIGINT N1,BIGINT N2,int ns)
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
*/
{
  FLT out[] = {0.0, 0.0};
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2) {  // no wrapping: avoid ptrs
    FLT line[2*MAX_NSPREAD];   // store a horiz line (interleaved real,imag)
    // block for first y line, to avoid explicitly initializing line with zeros
    {
      const FLT *lptr = du + 2*(N1*i2 + i1);   // ptr to horiz line start in du
      for (int l=0; l<2*ns; l++) {    // l is like dx but for ns interleaved
        line[l] = ker2[0]*lptr[l];
      }
    }
    // add remaining const-y lines to the line (expensive inner loop)
    for (int dy=1; dy<ns; dy++) {
      const FLT *lptr = du + 2*(N1*(i2+dy) + i1);  // (see above)
      for (int l=0; l<2*ns; ++l) {
        line[l] += ker2[dy]*lptr[l];
      }
    }
    // apply x kernel to the (interleaved) line and add together
    for (int dx=0; dx<ns; dx++) {
      out[0] += line[2*dx]   * ker1[dx];
      out[1] += line[2*dx+1] * ker1[dx];
    }
  } else {                         // wraps somewhere: use ptr list
    // this is slower than above, but occurs much less often, with fractional
    // rate O(ns/min(N1,N2)). Thus this code doesn't need to be so optimized.
    BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD];   // 1d ptr lists
    BIGINT x=i1, y=i2;                 // initialize coords
    for (int d=0; d<ns; d++) {         // set up ptr lists
      if (x<0) x+=N1;
      if (x>=N1) x-=N1;
      j1[d] = x++;
      if (y<0) y+=N2;
      if (y>=N2) y-=N2;
      j2[d] = y++;
    }
    for (int dy=0; dy<ns; dy++) {      // use the pts lists
      BIGINT oy = N1*j2[dy];           // offset due to y
      for (int dx=0; dx<ns; dx++) {
	FLT k = ker1[dx]*ker2[dy];
	BIGINT j = oy + j1[dx];
	out[0] += du[2*j] * k;
	out[1] += du[2*j+1] * k;
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];  
}

void interp_cube(FLT *target,FLT *du, FLT *ker1, FLT *ker2, FLT *ker3,
		 BIGINT i1,BIGINT i2,BIGINT i3, BIGINT N1,BIGINT N2,BIGINT N3,int ns)
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
*/
{
  FLT out[] = {0.0, 0.0};  
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2 && i3>=0 && i3+ns<=N3) {
    // no wrapping: avoid ptrs (by far the most common case)
    FLT line[2*MAX_NSPREAD];       // store a horiz line (interleaved real,imag)
    // initialize line with zeros; hard to avoid here, but overhead small in 3D
    for (int l=0; l<2*ns; l++) {
      line[l] = 0;
    }
    // co-add y and z contributions to line in x; do not apply x kernel yet
    // This is expensive innermost loop
    for (int dz=0; dz<ns; dz++) {
      BIGINT oz = N1*N2*(i3+dz);        // offset due to z
      for (int dy=0; dy<ns; dy++) {
        const FLT *lptr = du + 2*(oz + N1*(i2+dy) + i1);  // ptr start of line
        FLT ker23 = ker2[dy]*ker3[dz];
        for (int l=0; l<2*ns; ++l) {    // loop over ns interleaved (R,I) pairs
          line[l] += lptr[l]*ker23;
        }
      }
    }
    // apply x kernel to the (interleaved) line and add together (cheap)
    for (int dx=0; dx<ns; dx++) {
      out[0] += line[2*dx]   * ker1[dx];
      out[1] += line[2*dx+1] * ker1[dx];
    }
  } else {                         // wraps somewhere: use ptr list
    // ...can be slower since this case only happens with probability
    // O(ns/min(N1,N2,N3))
    BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD], j3[MAX_NSPREAD];   // 1d ptr lists
    BIGINT x=i1, y=i2, z=i3;         // initialize coords
    for (int d=0; d<ns; d++) {          // set up ptr lists
      if (x<0) x+=N1;
      if (x>=N1) x-=N1;
      j1[d] = x++;
      if (y<0) y+=N2;
      if (y>=N2) y-=N2;
      j2[d] = y++;
      if (z<0) z+=N3;
      if (z>=N3) z-=N3;
      j3[d] = z++;
    }
    for (int dz=0; dz<ns; dz++) {             // use the pts lists
      BIGINT oz = N1*N2*j3[dz];               // offset due to z
      for (int dy=0; dy<ns; dy++) {
	BIGINT oy = oz + N1*j2[dy];           // offset due to y & z
	FLT ker23 = ker2[dy]*ker3[dz];	
	for (int dx=0; dx<ns; dx++) {
	  FLT k = ker1[dx]*ker23;
	  BIGINT j = oy + j1[dx];
	  out[0] += du[2*j] * k;
	  out[1] += du[2*j+1] * k;
	}
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];  
}

template<uint16_t ns, bool kerevalmeth>
void spread_subproblem_1d_kernel(const BIGINT off1, const BIGINT size1, FLT * __restrict__ du, const BIGINT M,
                              const FLT * const kx, const FLT * const dd, const finufft_spread_opts& opts) noexcept {
/* 1D spreader from nonuniform to uniform subproblem grid, without wrapping.
   Inputs:
   off1 - integer offset of left end of du subgrid from that of overall fine
          periodized output grid {0,1,..N-1}.
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
*/

  static constexpr auto ns2 = ns * FLT(0.5);          // half spread width
  std::fill(du, du + 2 * size1, 0);           // zero output
  FLT ker[MAX_NSPREAD];

  for (BIGINT i = 0; i < M; i++) {           // loop over NU pts
    const auto re0 = dd[2 * i];
    const auto im0 = dd[2 * i + 1];

    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = (BIGINT) std::ceil(kx[i] - ns2);    // fine grid start index
    auto x1 = (FLT) i1 - kx[i];            // x1 in [-w/2,-w/2+1], up to rounding
    // However if N1*epsmach>O(1) then can cause O(1) errors in x1, hence ppoly
    // kernel evaluation will fall outside their designed domains, >>1 errors.
    // This can only happen if the overall error would be O(1) anyway. Clip x1??
    if (x1 < -ns2) x1 = -ns2; // why the wrapping only in 1D ?
    if (x1 > -ns2 + 1) x1 = -ns2 + 1;   // ***
    if constexpr (kerevalmeth) {          // faster Horner poly method
      eval_kernel_vec_Horner<ns>(ker, x1, opts);
   } else {
      FLT kernel_args[ns];
      set_kernel_args(kernel_args, x1, opts);
      evaluate_kernel_vector(ker, kernel_args, opts, ns);
    }

    const auto j = i1 - off1;    // offset rel to subgrid, starts the output indices
    auto* __restrict__ trg = du + 2 * j;
    // critical inner loop:
    for (auto dx=0; dx<ns; ++dx) {
      const auto k = ker[dx];
      trg[2*dx] = std::fma(re0, k, trg[2*dx]);
      trg[2*dx+1] = std::fma(im0, k, trg[2*dx+1]);
    }
  }
}

template<uint16_t NS>
FINUFFT_ALWAYS_INLINE
static void spread_subproblem_1d_dispatch(const BIGINT off1, const BIGINT size1, FLT *__restrict__ du, const BIGINT M,
                                   const FLT *kx, const FLT *dd,
                                   const finufft_spread_opts &opts) noexcept {
  static_assert(MIN_NSPREAD <= NS <= MAX_NSPREAD, "NS must be in the range (MIN_NSPREAD, MAX_NSPREAD)");
  if constexpr (NS == MIN_NSPREAD) { // Base case
    if (opts.kerevalmeth)
      return spread_subproblem_1d_kernel<MIN_NSPREAD, true>(off1, size1, du, M, kx, dd, opts);
    else {
      return spread_subproblem_1d_kernel<MIN_NSPREAD, false>(off1, size1, du, M, kx, dd, opts);
    }
  } else {
    if (opts.nspread == NS ){
      if (opts.kerevalmeth) {
        return spread_subproblem_1d_kernel<NS, true>(off1, size1, du, M, kx, dd, opts);
      } else {
        return spread_subproblem_1d_kernel<NS, false>(off1, size1, du, M, kx, dd, opts);
      }
    } else {
      return spread_subproblem_1d_dispatch< NS - 1 >(off1, size1, du, M, kx, dd, opts);
    }
  }
}

void spread_subproblem_1d(BIGINT off1, BIGINT size1,FLT *du,BIGINT M,
                          FLT *kx,FLT *dd, const finufft_spread_opts& opts) noexcept
/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
See above docs/notes for spread_subproblem_2d.
kx,ky,kz (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in each dim.
dd (size M complex) are complex source strengths
du (size size1*size2*size3) is uniform complex output array
*/
{
  spread_subproblem_1d_dispatch<MAX_NSPREAD>(off1, size1, du, M, kx, dd, opts);
}

template<uint16_t ns, bool kerevalmeth>
static void spread_subproblem_2d_kernel(const BIGINT off1, const BIGINT off2, const BIGINT size1, const BIGINT size2,
                          FLT * __restrict__ du, const BIGINT M, const FLT *kx, const FLT *ky, const FLT *dd,
                          const finufft_spread_opts &opts) noexcept
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array
*/
{
  static constexpr auto padding = get_padding<FLT, 2 * ns>();
  using batch_t = typename xsimd::make_sized_batch<FLT, GetPaddedSIMDSize<FLT, 2 * ns>()>::type;
  using arch_t = typename batch_t::arch_type;
  static constexpr auto avx_size = batch_t::size;
  static constexpr size_t alignment = batch_t::arch_type::alignment();

  static constexpr auto ns2 = ns * FLT(0.5);          // half spread width
  std::fill(du, du + 2 * size1 * size2, 0);
  alignas(alignment) FLT ker1val[2 * ns + padding] = {0};
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in all three directions in a single kernel evaluation call.
  alignas(alignment) FLT kernel_values[2 * MAX_NSPREAD];
  auto *ker1 = kernel_values;
  auto *ker2 = kernel_values + ns;
  for (BIGINT pt = 0; pt < M; pt++) {           // loop over NU pts
    const auto re0 = dd[2 * pt];
    const auto im0 = dd[2 * pt + 1];
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = (BIGINT) std::ceil(kx[pt] - ns2);   // fine grid start indices
    const auto i2 = (BIGINT) std::ceil(ky[pt] - ns2);
    const auto x1 = (FLT) i1 - kx[pt];
    const auto x2 = (FLT) i2 - ky[pt];
    if constexpr (kerevalmeth) {          // faster Horner poly method
      eval_kernel_vec_Horner<ns>(ker1, x1, opts);
      eval_kernel_vec_Horner<ns>(ker2, x2, opts);
    } else {
      alignas(alignment) FLT kernel_args[3 * ns];
      set_kernel_args(kernel_args, x1, opts);
      set_kernel_args(kernel_args + ns, x2, opts);
      evaluate_kernel_vector(kernel_values, kernel_args, opts, 3 * ns);
    }
    // Combine kernel with complex source value to simplify inner loop
    // here 2* is because of complex
    // initialized to 0 due to the padding

    for (auto i = 0; i < ns; i++) {
      ker1val[2 * i] = re0 * ker1[i];
      ker1val[2 * i + 1] = im0 * ker1[i];
    }
    // critical inner loop:
    for (auto dy = 0; dy < ns; ++dy) {
      const auto j = size1 * (i2 - off2 + dy) + i1 - off1;   // should be in subgrid
      auto *__restrict__ trg = du + 2 * j;
      const auto kerval = ker2[dy];
      const batch_t kerval_batch(kerval);
      for (auto dx = 0; dx < 2 * ns; dx += avx_size) {
        const auto ker1val_batch = xsimd::load_aligned<arch_t>(ker1val + dx);
        const auto trg_batch = xsimd::load_unaligned<arch_t>(trg + dx);
        const auto result = xsimd::fma(kerval_batch, ker1val_batch, trg_batch);
        result.store_unaligned(trg + dx);
      }
    }
  }
}

template<uint16_t NS>
FINUFFT_ALWAYS_INLINE
static void spread_subproblem_2d_dispatch(const BIGINT off1, const BIGINT off2, const BIGINT size1, const BIGINT size2,
                                 FLT * __restrict__ du, const BIGINT M, const FLT *kx, const FLT *ky, const FLT *dd,
                                 const finufft_spread_opts &opts) {
  static_assert(MIN_NSPREAD <= NS <= MAX_NSPREAD, "NS must be in the range (MIN_NSPREAD, MAX_NSPREAD)");
  if constexpr (NS == MIN_NSPREAD) { // Base case
    if (opts.kerevalmeth)
      return spread_subproblem_2d_kernel<MIN_NSPREAD, true>(off1, off2, size1, size2, du, M, kx, ky, dd, opts);
    else {
      return spread_subproblem_2d_kernel<MIN_NSPREAD, false>(off1, off2, size1, size2, du, M, kx, ky, dd, opts);
    }
  } else {
    if (opts.nspread == NS ){
      if (opts.kerevalmeth) {
        return spread_subproblem_2d_kernel<NS, true>(off1, off2, size1, size2, du, M, kx, ky, dd, opts);
      } else {
        return spread_subproblem_2d_kernel<NS, false>(off1, off2, size1, size2, du, M, kx, ky, dd, opts);
      }
    } else {
      return spread_subproblem_2d_dispatch < NS - 1 >
          (off1, off2, size1, size2, du, M, kx, ky, dd, opts);
    }
  }
}

void spread_subproblem_2d(const BIGINT off1, const BIGINT off2, const BIGINT size1, const BIGINT size2,
                          FLT * __restrict__ du, const BIGINT M, const FLT *kx, const FLT *ky, const FLT *dd,
                          const finufft_spread_opts &opts) noexcept
/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
See above docs/notes for spread_subproblem_2d.
kx,ky,kz (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in each dim.
dd (size M complex) are complex source strengths
du (size size1*size2*size3) is uniform complex output array
*/
{
  spread_subproblem_2d_dispatch<MAX_NSPREAD>(off1, off2, size1, size2, du, M, kx, ky, dd, opts);
}


template<uint16_t ns, bool kerevalmeth>
static void spread_subproblem_3d_kernel(const BIGINT off1, const BIGINT off2, const BIGINT off3, const BIGINT size1,
                                 const BIGINT size2, const BIGINT size3, FLT *__restrict__ du, const BIGINT M,
                                 const FLT *kx, const FLT *ky, const FLT *kz, const FLT *dd,
                                 const finufft_spread_opts &opts) noexcept {
  static constexpr auto padding = get_padding<FLT, 2 * ns>();
  using batch_t = PaddedSIMD<FLT, 2 * ns>;
  using arch_t = typename batch_t::arch_type;
  static constexpr auto avx_size = batch_t::size;
  static constexpr size_t alignment = batch_t::arch_type::alignment();

  static constexpr auto ns2 = ns * FLT(0.5);          // half spread width
  std::fill(du, du + 2 * size1 * size2 * size3, 0);
  // initialized to 0 due to the padding
  alignas(alignment) FLT ker1val[2 * ns + padding]={0};
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in all three directions in a single kernel evaluation call.
  alignas(alignment) FLT kernel_values[3 * MAX_NSPREAD];
  auto * ker1 = kernel_values;
  auto * ker2 = kernel_values + ns;
  auto * ker3 = kernel_values + 2 * ns;
  for (BIGINT pt = 0; pt < M; pt++) {           // loop over NU pts
    const auto re0 = dd[2 * pt];
    const auto im0 = dd[2 * pt + 1];
    // ceil offset, hence rounding, must match that in get_subgrid...
    const auto i1 = (BIGINT) std::ceil(kx[pt] - ns2);   // fine grid start indices
    const auto i2 = (BIGINT) std::ceil(ky[pt] - ns2);
    const auto i3 = (BIGINT) std::ceil(kz[pt] - ns2);
    const auto x1= std::ceil(kx[pt] - ns2) - kx[pt];
    const auto x2= std::ceil(ky[pt] - ns2) - ky[pt];
    const auto x3= std::ceil(kz[pt] - ns2) - kz[pt];
    if constexpr (kerevalmeth) {          // faster Horner poly method
      eval_kernel_vec_Horner<ns>(ker1, x1, opts);
      eval_kernel_vec_Horner<ns>(ker2, x2, opts);
      eval_kernel_vec_Horner<ns>(ker3, x3, opts);
    } else {
      alignas(alignment) FLT kernel_args[3 * ns];
      set_kernel_args(kernel_args, x1, opts);
      set_kernel_args(kernel_args + ns, x2, opts);
      set_kernel_args(kernel_args + 2 * ns, x3, opts);
      evaluate_kernel_vector(kernel_values, kernel_args, opts, 3 * ns);
    }
    // Combine kernel with complex source value to simplify inner loop
    // here 2* is because of complex
    for (auto i = 0; i < ns; i++) {
      ker1val[2 * i] = re0 * ker1[i];
      ker1val[2 * i + 1] = im0 * ker1[i];
    }
    // critical inner loop:
    for (auto dz = 0; dz < ns; ++dz) {
      const auto oz = size1 * size2 * (i3 - off3 + dz);        // offset due to z
      for (auto dy = 0; dy < ns; ++dy) {
        const auto j = oz + size1 * (i2 - off2 + dy) + i1 - off1;   // should be in subgrid
        auto * __restrict__ trg = du + 2 * j;
        const auto kerval = ker2[dy] * ker3[dz];
        const batch_t kerval_batch(kerval);
        for (auto dx = 0; dx < 2 * ns; dx += avx_size) {
          const auto ker1val_batch = xsimd::load_aligned<arch_t>(ker1val + dx);
          const auto trg_batch = xsimd::load_unaligned<arch_t>(trg + dx);
          const auto result = xsimd::fma(kerval_batch, ker1val_batch, trg_batch);
          result.store_unaligned(trg + dx);
        }
      }
    }
  }
}

template<int NS>
FINUFFT_ALWAYS_INLINE
static void spread_subproblem_3d_dispatch(BIGINT off1, BIGINT off2, BIGINT off3, BIGINT size1, BIGINT size2, BIGINT size3,
                                   FLT *du, BIGINT M, const FLT *kx, const FLT *ky, const FLT *kz, const FLT *dd,
                                   const finufft_spread_opts &opts) noexcept {
  static_assert(MIN_NSPREAD <= NS <= MAX_NSPREAD, "NS must be in the range (MIN_NSPREAD, MAX_NSPREAD)");
  if constexpr (NS == MIN_NSPREAD) { // Base case
    if (opts.kerevalmeth)
      return spread_subproblem_3d_kernel<MIN_NSPREAD, true>(off1, off2, off3, size1, size2, size3, du, M, kx, ky, kz, dd, opts);
    else {
      return spread_subproblem_3d_kernel<MIN_NSPREAD, false>(off1, off2, off3, size1, size2, size3, du, M, kx, ky, kz, dd, opts);
    }
  } else {
    if (opts.nspread == NS ){
      if (opts.kerevalmeth) {
        return spread_subproblem_3d_kernel<NS, true>(off1, off2, off3, size1, size2, size3, du, M, kx, ky, kz, dd, opts);
      } else {
        return spread_subproblem_3d_kernel<NS, false>(off1, off2, off3, size1, size2, size3, du, M, kx, ky, kz, dd, opts);
      }
    } else {
      return spread_subproblem_3d_dispatch < NS - 1 >
      (off1, off2, off3, size1, size2, size3, du, M, kx, ky, kz, dd, opts);
    }
  }
}

void spread_subproblem_3d(BIGINT off1, BIGINT off2, BIGINT off3, BIGINT size1,
                          BIGINT size2, BIGINT size3, FLT *du, BIGINT M,
                          FLT *kx, FLT *ky, FLT *kz, FLT *dd,
                          const finufft_spread_opts &opts) noexcept
/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
See above docs/notes for spread_subproblem_2d.
kx,ky,kz (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in each dim.
dd (size M complex) are complex source strengths
du (size size1*size2*size3) is uniform complex output array
*/
{
  spread_subproblem_3d_dispatch<MAX_NSPREAD>(off1, off2, off3, size1, size2, size3, du, M, kx, ky, kz, dd, opts);
}


template<bool thread_safe>
void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3, BIGINT padded_size1,
                         BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
                         BIGINT N2, BIGINT N3, FLT *__restrict__ data_uniform,
                         const FLT *const du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   padded_size1,2,3 give the size of subgrid.
   Works in all dims. Thread-safe variant of the above routine,
   using atomic writes (R Blackwell, Nov 2020).
*/
{
  std::vector<BIGINT> o2(size2), o3(size3);
  static auto accumulate = [](FLT& a, FLT b) {
    if constexpr (thread_safe) { // NOLINT(*-branch-clone)
#pragma omp atomic
      a += b;
    } else {
      a += b;
    }
  };

  BIGINT y = offset2, z = offset3;    // fill wrapped ptr lists in slower dims y,z...
  for (int i = 0; i < size2; ++i) {
    if (y < 0) y += N2;
    if (y >= N2) y -= N2;
    o2[i] = y++;
  }
  for (int i = 0; i < size3; ++i) {
    if (z < 0) z += N3;
    if (z >= N3) z -= N3;
    o3[i] = z++;
  }
  BIGINT nlo = (offset1 < 0) ? -offset1 : 0;          // # wrapping below in x
  BIGINT nhi = (offset1 + size1 > N1) ? offset1 + size1 - N1 : 0;    // " above in x
  // this triple loop works in all dims
  for (int dz = 0; dz < size3; dz++) {       // use ptr lists in each axis
    const auto oz = N1 * N2 * o3[dz];            // offset due to z (0 in <3D)
    for (int dy = 0; dy < size2; dy++) {
      const auto oy = N1 * o2[dy] + oz;        // off due to y & z (0 in 1D)
      auto * __restrict__ out = data_uniform + 2 * oy;
      const auto in = du0 + 2 * padded_size1 * (dy + size2 * dz);   // ptr to subgrid array
      auto o = 2 * (offset1 + N1);         // 1d offset for output
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


void bin_sort_singlethread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,
	      double bin_size_x,double bin_size_y,double bin_size_z, int debug)
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
  const auto isky = (N2 > 1), iskz = (N3 > 1);  // ky,kz avail? (cannot access if not)
  // here the +1 is needed to allow round-off error causing i1=N1/bin_size_x,
  // for kx near +pi, ie foldrescale gives N1 (exact arith would be 0 to N1-1).
  // Note that round-off near kx=-pi stably rounds negative to i1=0.
  const auto nbins1 = BIGINT(FLT(N1) / bin_size_x + 1);
  const auto nbins2 = isky ? BIGINT(FLT(N2) / bin_size_y + 1) : 1;
  const auto nbins3 = iskz ? BIGINT(FLT(N3) / bin_size_z + 1) : 1;
  const auto nbins = nbins1 * nbins2 * nbins3;
  const auto inv_bin_size_x = FLT(1.0 / bin_size_x);
  const auto inv_bin_size_y = FLT(1.0 / bin_size_y);
  const auto inv_bin_size_z = FLT(1.0 / bin_size_z);
  // count how many pts in each bin
  std::vector<BIGINT> counts(nbins, 0);

  for (auto i=0; i<M; i++) {
    // find the bin index in however many dims are needed
    const auto i1 = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2 = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3 = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1+nbins1*(i2+nbins2*i3);
    ++counts[bin];
  }

  // compute the offsets directly in the counts array (no offset array)
  BIGINT current_offset=0;
  for (BIGINT i=0; i<nbins; i++) {
    BIGINT tmp = counts[i];
    counts[i] = current_offset;   // Reinecke's cute replacement of counts[i]
    current_offset += tmp;
  }              // (counts now contains the index offsets for each bin)

  for (auto i=0; i<M; i++) {
    // find the bin index (again! but better than using RAM)
    const auto i1 = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
    const auto i2 = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
    const auto i3 = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
    const auto bin = i1+nbins1*(i2+nbins2*i3);
    ret[counts[bin]] = BIGINT(i);      // fill the inverse map on the fly
    ++counts[bin];             // update the offsets
  }
}

void bin_sort_multithread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,
              double bin_size_x,double bin_size_y,double bin_size_z, int debug,
              int nthr)
/* Mostly-OpenMP'ed version of bin_sort.
   For documentation see: bin_sort_singlethread.
   Caution: when M (# NU pts) << N (# U pts), is SLOWER than single-thread.
   Originally by Barnett 2/8/18
   Explicit #threads control argument 7/20/20.
   Improved by Martin Reinecke, 6/19/23 (up to 50% faster at 1 thr/core).
   Todo: if debug, print timing breakdowns.
 */
{
  bool isky=(N2>1), iskz=(N3>1);  // ky,kz avail? (cannot access if not)
  BIGINT nbins1=N1/bin_size_x+1, nbins2, nbins3;  // see above note on why +1
  nbins2 = isky ? N2/bin_size_y+1 : 1;
  nbins3 = iskz ? N3/bin_size_z+1 : 1;
  BIGINT nbins = nbins1*nbins2*nbins3;
  if (nthr==0)                      // should never happen in spreadinterp use
    fprintf(stderr,"[%s] nthr (%d) must be positive!\n",__func__,nthr);
  int nt = min(M,(BIGINT)nthr);     // handle case of less points than threads
  std::vector<BIGINT> brk(nt+1);    // list of start NU pt indices per thread

  // distribute the NU pts to threads once & for all...
  for (int t=0; t<=nt; ++t)
    brk[t] = (BIGINT)(0.5 + M*t/(double)nt);   // start index for t'th chunk

  // set up 2d array (nthreads * nbins), just its pointers for now
  // (sub-vectors will be initialized later)
  std::vector< std::vector<BIGINT> > counts(nt);
    
#pragma omp parallel num_threads(nt)
  {  // parallel binning to each thread's count. Block done once per thread
    int t = MY_OMP_GET_THREAD_NUM();     // (we assume all nt threads created)
    auto &my_counts(counts[t]);          // name for counts[t]
    my_counts.resize(nbins,0);  // allocate counts[t], now in parallel region
    for (BIGINT i=brk[t]; i<brk[t+1]; i++) {
      // find the bin index in however many dims are needed
      BIGINT i1= fold_rescale(kx[i], N1) / bin_size_x, i2=0, i3=0;
      if (isky) i2 = fold_rescale(ky[i], N2) / bin_size_y;
      if (iskz) i3 = fold_rescale(kz[i], N3) / bin_size_z;
      BIGINT bin = i1+nbins1*(i2+nbins2*i3);
      ++my_counts[bin];               // no clash btw threads
    }
  }
  
  // inner sum along both bin and thread (inner) axes to get global offsets
  BIGINT current_offset = 0;
  for (BIGINT b=0; b<nbins; ++b)   // (not worth omp)
    for (int t=0; t<nt; ++t) {
      BIGINT tmp = counts[t][b];
      counts[t][b] = current_offset;
      current_offset += tmp;
    }   // counts[t][b] is now the index offset as if t ordered fast, b slow
  
#pragma omp parallel num_threads(nt)
  {
    int t = MY_OMP_GET_THREAD_NUM();
    auto &my_counts(counts[t]);
    for (BIGINT i=brk[t]; i<brk[t+1]; i++) {
      // find the bin index (again! but better than using RAM)
      BIGINT i1= fold_rescale(kx[i], N1) / bin_size_x, i2=0, i3=0;
      if (isky) i2 = fold_rescale(ky[i], N2) / bin_size_y;
      if (iskz) i3 = fold_rescale(kz[i], N3) / bin_size_z;
      BIGINT bin = i1+nbins1*(i2+nbins2*i3);
      ret[my_counts[bin]] = i;   // inverse is offset for this NU pt and thread
      ++my_counts[bin];          // update the offsets; no thread clash
    }
  }
}

void get_subgrid(BIGINT &offset1, BIGINT &offset2, BIGINT &offset3, BIGINT &padded_size1, BIGINT &size1, BIGINT &size2, BIGINT &size3, BIGINT M, FLT* kx, FLT* ky, FLT* kz, int ns, int ndims)
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
  FLT ns2 = (FLT)ns/2;
  FLT min_kx,max_kx;   // 1st (x) dimension: get min/max of nonuniform points
  arrayrange(M,kx,&min_kx,&max_kx);
  offset1 = (BIGINT)std::ceil(min_kx-ns2);   // min index touched by kernel
  size1 = (BIGINT)std::ceil(max_kx - ns2) - offset1 + ns;  // int(ceil) first!
  padded_size1 = size1+get_padding<FLT>(2 * ns)/2;
  if (ndims>1) {
    FLT min_ky,max_ky;   // 2nd (y) dimension: get min/max of nonuniform points
    arrayrange(M,ky,&min_ky,&max_ky);
    offset2 = (BIGINT)std::ceil(min_ky-ns2);
    size2 = (BIGINT)std::ceil(max_ky-ns2) - offset2 + ns;
  } else {
    offset2 = 0;
    size2 = 1;
  }
  if (ndims>2) {
    FLT min_kz,max_kz;   // 3rd (z) dimension: get min/max of nonuniform points
    arrayrange(M,kz,&min_kz,&max_kz);
    offset3 = (BIGINT)std::ceil(min_kz-ns2);
    size3 = (BIGINT)std::ceil(max_kz-ns2) - offset3 + ns;
  } else {
    offset3 = 0;
    size3 = 1;
  }
}
/* local NU coord fold+rescale macro: does the following affine transform to x:
     when p=true:   (x+PI) mod PI    each to [0,N)
     otherwise,     x mod N          each to [0,N)
   Note: folding big numbers can cause numerical inaccuracies
   Martin Reinecke, 8.5.2024 used floor to speedup the function and removed the range limitation
   Marco Barbone, 8.5.2024 Changed it from a Macro to an inline function
*/
FINUFFT_ALWAYS_INLINE FLT fold_rescale(const FLT x, const BIGINT N) noexcept {
  static constexpr const FLT x2pi = FLT(M_1_2PI);
  const FLT result = x * x2pi + FLT(0.5);
  return (result-floor(result)) * FLT(N);
}

// Below there is some template metaprogramming magic to find the best SIMD type
// for the given number of elements. The code is based on the xsimd library


// this finds the largest SIMD instruction set that can handle N elements
// void otherwise -> compile error
template<class T, uint16_t N, uint16_t K>
static constexpr auto BestSIMDHelper() {
  if constexpr (N % K == 0) { // returns void in the worst case
    return xsimd::make_sized_batch<T, K>{};
  } else {
    return BestSIMDHelper<T, N, (K>>1)>();
  }
}

template<class T, uint16_t N>
static constexpr uint16_t min_batch_size() {
  if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
    return min_batch_size<T, N*2>();
  } else {
    return N;
  }
};


template<class T, uint16_t N, uint16_t batch_size, uint16_t min_iterations, uint16_t optimal_batch_size>
constexpr uint16_t find_optimal_batch_size() {
  if constexpr (batch_size > xsimd::batch<T>::size) {
    return optimal_batch_size;
  } else {
    constexpr uint16_t iterations = (N + batch_size - 1) / batch_size;
    if constexpr (iterations < min_iterations) {
      return find_optimal_batch_size<T, N, batch_size * 2, iterations, batch_size>();
    } else {
      return find_optimal_batch_size<T, N, batch_size * 2, min_iterations, optimal_batch_size>();
    }
  }
}

template<class T, uint16_t N>
static constexpr auto GetPaddedSIMDSize() {
  static_assert(N < 128);
  return xsimd::make_sized_batch<T, find_optimal_batch_size<T, N>()>::type::size;
}

template<class T, uint16_t ns>
static constexpr auto get_padding() {
  constexpr uint16_t width = GetPaddedSIMDSize<T, ns>();
  return ns % width == 0 ? 0 : width - (ns % width);
}

template<class T, uint16_t ns>
static constexpr auto get_padding_helper(uint16_t runtime_ns) {
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

template<class T>
static uint16_t get_padding(uint16_t ns) {
  return get_padding_helper<T, 2*MAX_NSPREAD>(ns);
}

}   // namespace
}   // namespace
