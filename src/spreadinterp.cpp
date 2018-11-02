#include "spreadinterp.h"
#include <stdlib.h>
#include <vector>
#include <math.h>

// declarations of internal functions...
static inline void set_kernel_args(FLT *args, FLT x, const spread_opts& opts);
static inline void evaluate_kernel_vector(FLT *ker, FLT *args, const spread_opts& opts, const int N);
static inline void eval_kernel_vec_Horner(FLT *ker, const FLT z, const int w, const spread_opts &opts);
void interp_line(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns);
void interp_square(FLT *out,FLT *du, FLT *ker1, FLT *ker2, BIGINT i1,BIGINT i2,BIGINT N1,BIGINT N2,int ns);
void interp_cube(FLT *out,FLT *du, FLT *ker1, FLT *ker2, FLT *ker3,
		 BIGINT i1,BIGINT i2,BIGINT i3,BIGINT N1,BIGINT N2,BIGINT N3,int ns);
void spread_subproblem_1d(BIGINT N1,FLT *du0,BIGINT M0,FLT *kx0,FLT *dd0,
			  const spread_opts& opts);
void spread_subproblem_2d(BIGINT N1,BIGINT N2,FLT *du0,BIGINT M0,
			  FLT *kx0,FLT *ky0,FLT *dd0,const spread_opts& opts);
void spread_subproblem_3d(BIGINT N1,BIGINT N2,BIGINT N3,FLT *du0,BIGINT M0,
			  FLT *kx0,FLT *ky0,FLT *kz0,FLT *dd0,
			  const spread_opts& opts);
void add_wrapped_subgrid(BIGINT offset1,BIGINT offset2,BIGINT offset3,
			 BIGINT size1,BIGINT size2,BIGINT size3,BIGINT N1,
			 BIGINT N2,BIGINT N3,FLT *data_uniform, FLT *du0);
void bin_sort_singlethread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
	      double bin_size_x,double bin_size_y,double bin_size_z, int debug);
void bin_sort_multithread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
	      double bin_size_x,double bin_size_y,double bin_size_z, int debug);
void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,
		 BIGINT &size2,BIGINT &size3,BIGINT M0,FLT* kx0,FLT* ky0,
		 FLT* kz0,int ns, int ndims);


int spreadinterp(
        BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
        BIGINT M, FLT *kx, FLT *ky, FLT *kz, FLT *data_nonuniform,
        spread_opts opts)
/* Spreader/interpolator for 1, 2, or 3 dimensions.
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
   For 1D set N2=N3=1; for 2D set N3=1; for 3D set N1,N2,N3>0.

   Notes:
   No particular normalization of the spreading kernel is assumed.
   Uniform (U) points are centered at coords
   [0,1,...,N1-1] in 1D, analogously in 2D and 3D. They are stored in x
   fastest, y medium, z slowest ordering, up to however many
   dimensions are relevant; note that this is Fortran-style ordering for an
   array f(x,y,z), but C style for f[z][y][x]. This is to match the fortran
   interface of the original CMCL libraries.
   Non-uniform (NU) points kx,ky,kz are real.
   If pirange=0, should be in the range [0,N1] in 1D, analogously in 2D and 3D.
   If pirange=1, the range is instead [-pi,pi] for each coord.
   The spread_opts struct must have been set up already by calling setup_kernel.
   It is assumed that 2*opts.nspread < min(N1,N2,N3), so that the kernel
   only ever wraps once when falls below 0 or off the top of a uniform grid
   dimension.

   Inputs:
   N1,N2,N3 - grid sizes in x (fastest), y (medium), z (slowest) respectively.
              If N2==0, 1D spreading is done. If N3==0, 2D spreading.
	      Otherwise, 3D.
   M - number of NU pts.
   kx, ky, kz - length-M real arrays of NU point coordinates (only kz used in
                1D, only kx and ky used in 2D).
		These should lie in the box 0<=kx<=N1 etc (if pirange=0),
                or -pi<=kx<=pi (if pirange=1). However, points up to +-1 period
                outside this domain are also correctly folded back into this
                domain, but pts beyond this either raise an error (if chkbnds=1)
                or a crash (if chkbnds=0).
   opts - object controlling spreading method and text output, has fields
          including:
        spread_direction=1, spreads from nonuniform input to uniform output, or
        spread_direction=2, interpolates ("spread transpose") from uniform input
                            to nonuniform output.
	pirange = 0: kx,ky,kz coords in [0,N]. 1: coords in [-pi,pi].
                (due to +-1 box folding these can be out to [-N,2N] and
                [-3pi/2,3pi/2] respectively).
	sort = 0,1,2: whether to sort NU points using natural yz-grid
	       ordering. 0: don't, 1: do, 2: use heuristic choice (default)
        sort_threads = 0, 1,... : if >0, set # sorting threads; if 0
                   allow heuristic choice (either single or all avail).
	kerpad = 0,1: whether pad to next mult of 4, helps SIMD (kerevalmeth=0).
	kerevalmeth = 0: direct exp(sqrt(..)) eval; 1: Horner piecewise poly.
	debug = 0: no text output, 1: some openmp output, 2: mega output
	           (each NU pt)
	chkbnds = 0: don't check incoming NU pts for bounds (but still fold +-1)
                  1: do, and stop with error if any found outside valid bnds
	flags = integer with binary bits determining various timing options
                (set to 0 unless expert; see cnufftspread.h)

   Inputs/Outputs:
   data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
   data_nonuniform - input strengths of the sources (dir=1)
                     OR output values at targets (dir=2)
   Ouputs:
   returned value - 0 indicates success; other values as follows
      (see utils.h and ../docs/usage.rst for error codes)
      3 : one or more non-trivial box dimensions is less than 2.nspread.
      4 : nonuniform points outside [-Nm,2*Nm] or [-3pi,3pi] in at least one
          dimension m=1,2,3.
      6 : invalid opts.spread_direction

   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
   error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17. parallel sort 2/9/18
   No separate subprob indices in t-1 2/11/18.
   sort_threads (since for M<<N, multithread sort slower than single) 3/27/18
   kereval, kerpad 4/24/18
   Melody Shih split into 3 routines: check, sort, spread. Jun 2018, making
   this routine just a caller to them. Name change, Barnett 7/27/18
*/
{
  int ier_spreadcheck = spreadcheck(N1, N2, N3, M, kx, ky, kz, opts);
  if (ier_spreadcheck != 0) return ier_spreadcheck;

  BIGINT* sort_indices = (BIGINT*)malloc(sizeof(BIGINT)*M);
  int did_sort = spreadsort(sort_indices, N1, N2, N3, M, kx, ky, kz, opts);
  int ier_spread = spreadwithsortidx(sort_indices, N1, N2, N3, data_uniform,
                                           M, kx, ky, kz, data_nonuniform,
                                           opts,did_sort);
  if (ier_spread != 0) return ier_spread;

  free(sort_indices);
  return 0;
}

int ndims_from_Ns(BIGINT N1, BIGINT N2, BIGINT N3)
/* rule for getting number of spreading dimensions from the list of Ns per dim.
   Split out, Barnett 7/26/18
*/
{
  int ndims = 1;                // decide ndims: 1,2 or 3
  if (N2>1) ++ndims;
  if (N3>1) ++ndims;
  return ndims;
}

int spreadcheck(BIGINT N1, BIGINT N2, BIGINT N3,
		BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		spread_opts opts)
/* This does just the input checking and reporting for the spreader.
   See cnufftspread() for input arguments.
   Return value is that returned by cnufftspread.
   Split out by Melody Shih, Jun 2018. Finiteness chk 7/30/18.
*/
{
  CNTime timer;
  // INPUT CHECKING & REPORTING .... cuboid not too small for spreading?
  int minN = 2*opts.nspread;
  if (N1<minN || (N2>1 && N2<minN) || (N3>1 && N3<minN)) {
    fprintf(stderr,"error: one or more non-trivial box dims is less than 2.nspread!\n");
    return ERR_SPREAD_BOX_SMALL;
  }
  if (opts.spread_direction!=1 && opts.spread_direction!=2) {
    fprintf(stderr,"error: opts.spread_direction must be 1 or 2!\n");
    return ERR_SPREAD_DIR;
  }
  int ndims = ndims_from_Ns(N1,N2,N3);
  if (opts.debug)
    printf("starting spread %dD (dir=%d. M=%lld; N1=%lld,N2=%lld,N3=%lld; pir=%d), %d threads\n",ndims,opts.spread_direction,(long long)M,(long long)N1,(long long)N2,(long long)N3,opts.pirange,MY_OMP_GET_MAX_THREADS());
  
  // BOUNDS CHECKING .... check NU pts are valid (incl +-1 box), exit gracefully
  if (opts.chkbnds) {
    timer.start();
    for (BIGINT i=0; i<M; ++i) {
      FLT x=RESCALE(kx[i],N1,opts.pirange);  // this includes +-1 box folding
      if (x<0 || x>N1 || !isfinite(x)) {     // note isfinite() breaks with -Ofast
        fprintf(stderr,"NU pt not in valid range (central three periods): kx=%g, N1=%lld (pirange=%d)\n",x,(long long)N1,opts.pirange);
        return ERR_SPREAD_PTS_OUT_RANGE;
      }
    }
    if (ndims>1)
      for (BIGINT i=0; i<M; ++i) {
        FLT y=RESCALE(ky[i],N2,opts.pirange);
        if (y<0 || y>N2 || !isfinite(y)) {
          fprintf(stderr,"NU pt not in valid range (central three periods): ky=%g, N2=%lld (pirange=%d)\n",y,(long long)N2,opts.pirange);
          return ERR_SPREAD_PTS_OUT_RANGE;
        }
      }
    if (ndims>2)
      for (BIGINT i=0; i<M; ++i) {
        FLT z=RESCALE(kz[i],N3,opts.pirange);
        if (z<0 || z>N3 || !isfinite(z)) {
          fprintf(stderr,"NU pt not in valid range (central three periods): kz=%g, N3=%lld (pirange=%d)\n",z,(long long)N3,opts.pirange);
          return ERR_SPREAD_PTS_OUT_RANGE;
        }
      }
    if (opts.debug) printf("\tNU bnds check:\t\t%.3g s\n",timer.elapsedsec());
  }
  return 0; 
}

int spreadsort(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, 
               FLT *kx, FLT *ky, FLT *kz, spread_opts opts)
/* This makes a decision whether to sort the NU pts, and if so, calls either
   single- or multi-threaded bin sort, writing reordered index list to sort_indices.
   See cnufftspread() for input arguments.
   Return value is whether a sort was done.
   Split out by Melody Shih, Jun 2018.
*/
{
  CNTime timer;
  int ndims = ndims_from_Ns(N1,N2,N3);
  BIGINT N=N1*N2*N3;            // output array size
  
  // NONUNIFORM POINT SORTING .....
  // heuristic binning box size for U grid... affects performance:
  double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;
  // put in heuristics based on L3 size (only useful for single-thread) ?
  int better_to_sort = !(ndims==1 && (opts.spread_direction==2 || (M > 1000*N1))); // 1D small-N or dir=2 case: don't sort

  timer.start();                 // if needed, sort all the NU pts...
  int did_sort=0;
  if (opts.sort==1 || (opts.sort==2 && better_to_sort)) {
    // store a good permutation ordering of all NU pts (dim=1,2 or 3)
    int sort_debug = (opts.debug>=2);    // show timing output?
    int sort_nthr = opts.sort_threads;   // choose # threads for sorting
    if (sort_nthr==0)   // auto choice: when N>>M, one thread is better!
      sort_nthr = (10*M>N) ? MY_OMP_GET_MAX_THREADS() : 1;      // heuristic
    if (sort_nthr==1)
      bin_sort_singlethread(sort_indices,M,kx,ky,kz,N1,N2,N3,opts.pirange,bin_size_x,bin_size_y,bin_size_z,sort_debug);
    else
      bin_sort_multithread(sort_indices,M,kx,ky,kz,N1,N2,N3,opts.pirange,bin_size_x,bin_size_y,bin_size_z,sort_debug);
    if (opts.debug) 
      printf("\tsorted (%d threads):\t%.3g s\n",sort_nthr,timer.elapsedsec());
    did_sort=1;
  } else {
#pragma omp parallel for schedule(static,1000000)
    for (BIGINT i=0; i<M; i++)                // omp helps xeon, hinders i7
      sort_indices[i]=i;                      // the identity permutation
    if (opts.debug)
      printf("\tnot sorted (sort=%d): \t%.3g s\n",(int)opts.sort,timer.elapsedsec());
  }
  return did_sort;
}

int spreadwithsortidx(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, spread_opts opts, int did_sort)
/* The main spreading (dir=1) and interpolation (dir=2) routines.
   See cnufftspread() above for inputs arguments and definitions.
   Return value should always be 0.
   Split out by Melody Shih, Jun 2018.
*/
{
  CNTime timer;
  int ndims = ndims_from_Ns(N1,N2,N3);
  BIGINT N=N1*N2*N3;            // output array size
  int ns=opts.nspread;          // abbrev. for w, kernel width
  FLT ns2 = (FLT)ns/2;          // half spread width, used as stencil shift

  if (opts.spread_direction==1) { // ========= direction 1 (spreading) =======

    timer.start();
    for (BIGINT i=0; i<2*N; i++) // zero the output array. std::fill is no faster
      data_uniform[i]=0.0;
    if (opts.debug) printf("\tzero output array\t%.3g s\n",timer.elapsedsec());
    if (M==0)                     // no NU pts, we're done
      return 0;

    int spread_single = (M*100<N);     // low-density heuristic?
    spread_single = 0;                 // for now
    timer.start();
    if (spread_single) {    // ------- Basic single-core t1 spreading ------
      for (BIGINT j=0; j<M; j++) {
        // todo, not urgent
      }
      if (opts.debug) printf("\tt1 simple spreading:\t%.3g s\n",timer.elapsedsec());

    } else {               // ------- Fancy multi-core blocked t1 spreading ----
      // Split sorted inds (jfm's advanced2), could double RAM
      int nb = MIN(4*MY_OMP_GET_MAX_THREADS(),M);     // Choose # subprobs
      if (nb*opts.max_subproblem_size<M)
        nb = (M+opts.max_subproblem_size-1)/opts.max_subproblem_size;  // int div
      if (M*1000<N) {         // low-density heuristic: one thread per NU pt!
        nb = M;
        if (opts.debug) printf("\tusing low-density speed rescue nb=M...\n");
      }
      if (!did_sort && MY_OMP_GET_MAX_THREADS()==1) {
        nb = 1;
        if (opts.debug) printf("\tforcing single subproblem...\n");
      }
      std::vector<BIGINT> brk(nb+1); // NU index breakpoints defining subproblems
      for (int p=0;p<=nb;++p)
        brk[p] = (BIGINT)(0.5 + M*p/(double)nb);
      
#pragma omp parallel for schedule(dynamic,1)
      for (int isub=0; isub<nb; isub++) {    // Main loop through the subproblems
        BIGINT M0 = brk[isub+1]-brk[isub];   // # NU pts in this subproblem
        // copy the location and data vectors for the nonuniform points
        FLT *kx0=(FLT*)malloc(sizeof(FLT)*M0), *ky0=NULL, *kz0=NULL;
        if (N2>1)
          ky0=(FLT*)malloc(sizeof(FLT)*M0);
        if (N3>1)
          kz0=(FLT*)malloc(sizeof(FLT)*M0);
        FLT *dd0=(FLT*)malloc(sizeof(FLT)*M0*2);    // complex strength data
        for (BIGINT j=0; j<M0; j++) {           // todo: can avoid this copying?
          BIGINT kk=sort_indices[j+brk[isub]];  // NU pt from subprob index list
          kx0[j]=RESCALE(kx[kk],N1,opts.pirange);
          if (N2>1) ky0[j]=RESCALE(ky[kk],N2,opts.pirange);
          if (N3>1) kz0[j]=RESCALE(kz[kk],N3,opts.pirange);
          dd0[j*2]=data_nonuniform[kk*2];     // real part
          dd0[j*2+1]=data_nonuniform[kk*2+1]; // imag part
        }
        // get the subgrid which will include padding by roughly nspread/2
        BIGINT offset1,offset2,offset3,size1,size2,size3; // get_subgrid sets
        get_subgrid(offset1,offset2,offset3,size1,size2,size3,M0,kx0,ky0,kz0,ns,ndims);  // sets offsets and sizes
        if (opts.debug>1) { // verbose
          if (ndims==1)
            printf("\tsubgrid: off %lld\t siz %lld\t #NU %lld\n",(long long)offset1,(long long)size1,(long long)M0);
          else if (ndims==2)
            printf("\tsubgrid: off %lld,%lld\t siz %lld,%lld\t #NU %lld\n",(long long)offset1,(long long)offset2,(long long)size1,(long long)size2,(long long)M0);
          else
            printf("\tsubgrid: off %lld,%lld,%lld\t siz %lld,%lld,%lld\t #NU %lld\n",(long long)offset1,(long long)offset2,(long long)offset3,(long long)size1,(long long)size2,(long long)size3,(long long)M0);
	}
        for (BIGINT j=0; j<M0; j++) {
          kx0[j]-=offset1;  // now kx0 coords are relative to corner of subgrid
          if (N2>1) ky0[j]-=offset2;  // only accessed if 2D or 3D
          if (N3>1) kz0[j]-=offset3;  // only access if 3D
        }
        // allocate output data for this subgrid
        FLT *du0=(FLT*)malloc(sizeof(FLT)*2*size1*size2*size3); // complex
        
        // Spread to subgrid without need for bounds checking or wrapping
        if (!(opts.flags & TF_OMIT_SPREADING)) {
          if (ndims==1)
            spread_subproblem_1d(size1,du0,M0,kx0,dd0,opts);
          else if (ndims==2)
            spread_subproblem_2d(size1,size2,du0,M0,kx0,ky0,dd0,opts);
          else
            spread_subproblem_3d(size1,size2,size3,du0,M0,kx0,ky0,kz0,dd0,opts);
	}
        
#pragma omp critical
        {  // do the adding of subgrid to output; only here threads cannot clash
          if (!(opts.flags & TF_OMIT_WRITE_TO_GRID))
            add_wrapped_subgrid(offset1,offset2,offset3,size1,size2,size3,N1,N2,N3,data_uniform,du0);
        }  // end critical block

        // free up stuff from this subprob... (that was malloc'ed by hand)
        free(dd0);
        free(du0);
        free(kx0);
        if (N2>1) free(ky0);
        if (N3>1) free(kz0); 
      }     // end main loop over subprobs
      if (opts.debug) printf("\tt1 fancy spread: \t%.3g s (%d subprobs)\n",timer.elapsedsec(), nb);
    }   // end of choice of which t1 spread type to use
    
  } else {          // ================= direction 2 (interpolation) ===========
    timer.start();
#pragma omp parallel
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
#pragma omp for schedule(dynamic) // assign threads to NU targ pts:
      for (BIGINT i=0; i<M; i+=CHUNKSIZE)  // main loop over NU targs, interp each from U
      {
        // Setup buffers for this chunk
        int bufsize = (i+CHUNKSIZE > M) ? M-i : CHUNKSIZE;
        for (int ibuf=0; ibuf<bufsize; ibuf++) {
          BIGINT j = sort_indices[i+ibuf];
          jlist[ibuf] = j;
          xjlist[ibuf] = RESCALE(kx[j],N1,opts.pirange);
        }
        if (ndims==3) {
          for (int ibuf=0; ibuf<bufsize; ibuf++) {
            BIGINT j = jlist[ibuf];
            yjlist[ibuf] = RESCALE(ky[j],N2,opts.pirange);
            zjlist[ibuf] = RESCALE(kz[j],N3,opts.pirange);                              
          }
        } else if (ndims==2) {
          for (int ibuf=0; ibuf<bufsize; ibuf++) {
            BIGINT j = jlist[ibuf];
            yjlist[ibuf] = RESCALE(ky[j],N2,opts.pirange);
          }
        }
        // Loop over targets in chunk
        for (int ibuf=0; ibuf<bufsize; ibuf++) {
          FLT xj = xjlist[ibuf];
          FLT *target = outbuf+2*ibuf;
        
          // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
          BIGINT i1=(BIGINT)std::ceil(xj-ns2); // leftmost grid index
          FLT x1=(FLT)i1-xj;           // shift of ker center, in [-w/2,-w/2+1]
          // eval kernel values patch and use to interpolate from uniform data...
          if (!(opts.flags & TF_OMIT_SPREADING)) {
            if (ndims==1) {                                          // 1D
              if (opts.kerevalmeth==0) {               // choose eval method
                set_kernel_args(kernel_args, x1, opts);
                evaluate_kernel_vector(kernel_values, kernel_args, opts, ns);
              } else
                eval_kernel_vec_Horner(ker1,x1,ns,opts);
              interp_line(target,data_uniform,ker1,i1,N1,ns);

            } else if (ndims==2) {                                   // 2D
              FLT yj=yjlist[ibuf];
              BIGINT i2=(BIGINT)std::ceil(yj-ns2); // min y grid index
              FLT x2=(FLT)i2-yj;
              if (opts.kerevalmeth==0) {               // choose eval method
                set_kernel_args(kernel_args, x1, opts);
                set_kernel_args(kernel_args+ns, x2, opts);
                evaluate_kernel_vector(kernel_values, kernel_args, opts, 2*ns);
              } else {
                eval_kernel_vec_Horner(ker1,x1,ns,opts);
                eval_kernel_vec_Horner(ker2,x2,ns,opts);
              }
              interp_square(target,data_uniform,ker1,ker2,i1,i2,N1,N2,ns);
            } else {                                                 // 3D
              FLT yj=yjlist[ibuf];
              FLT zj=zjlist[ibuf];
              BIGINT i2=(BIGINT)std::ceil(yj-ns2); // min y grid index
              BIGINT i3=(BIGINT)std::ceil(zj-ns2); // min z grid index
              FLT x2=(FLT)i2-yj;
              FLT x3=(FLT)i3-zj;
              if (opts.kerevalmeth==0) {               // choose eval method
                set_kernel_args(kernel_args, x1, opts);
                set_kernel_args(kernel_args+ns, x2, opts);
                set_kernel_args(kernel_args+2*ns, x3, opts);
                evaluate_kernel_vector(kernel_values, kernel_args, opts, 3*ns);
              } else {
                eval_kernel_vec_Horner(ker1,x1,ns,opts);
                eval_kernel_vec_Horner(ker2,x2,ns,opts);
                eval_kernel_vec_Horner(ker3,x3,ns,opts);
              }
              interp_cube(target,data_uniform,ker1,ker2,ker3,i1,i2,i3,N1,N2,N3,ns);
            }
	  }
        } // end loop over targets in chunk
        
        // Copy result buffer to output array
        for (int ibuf=0; ibuf<bufsize; ibuf++) {
          BIGINT j = jlist[ibuf];
          data_nonuniform[2*j] = outbuf[2*ibuf];
          data_nonuniform[2*j+1] = outbuf[2*ibuf+1];              
        }         
        
      }    // end NU targ loop
    } // end parallel section
    if (opts.debug) printf("\tt2 spreading loop: \t%.3g s\n",timer.elapsedsec());
  }                           // ================= end direction choice ========
  return 0;
}

///////////////////////////////////////////////////////////////////////////

int setup_spreader(spread_opts &opts,FLT eps,FLT upsampfac, int kerevalmeth)
// Initializes spreader kernel parameters given desired NUFFT tolerance eps,
// upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), and ker eval meth
// (etiher 0:exp(sqrt()), 1: Horner ppval).
// Also sets all default options in spread_opts. See cnufftspread.h for opts.
// Must call before any kernel evals done.
// Returns: 0 success, >0 failure (see error codes in utils.h)
{
  if (eps<0.5*EPSILON) {     // factor there since fortran wants 1e-16 to be ok
    fprintf(stderr,"setup_spreader: error, requested eps (%.3g) is too small (<%.3g)\n",(double)eps,0.5*EPSILON);
    return ERR_EPS_TOO_SMALL;
  }
  if (upsampfac!=2.0 && upsampfac!=1.25) {   // nonstandard sigma
    if (kerevalmeth==1) {
      fprintf(stderr,"setup_spreader: nonstandard upsampfac=%.3g cannot be handled by kerevalmeth=1\n",(double)upsampfac);
      return HORNER_WRONG_BETA;
    }
    if (upsampfac<=1.0) {
      fprintf(stderr,"setup_spreader: error, upsampfac=%.3g is <=1.0\n",(double)upsampfac);
      return ERR_UPSAMPFAC_TOO_SMALL;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (upsampfac>4.0)
      fprintf(stderr,"setup_spreader: warning, upsampfac=%.3g is too large to be beneficial.\n",(double)upsampfac);
  }
    
  // defaults... (user can change after this function called)
  opts.spread_direction = 1;    // user should always set to 1 or 2 as desired
  opts.pirange = 1;             // user also should always set this
  opts.chkbnds = 1;
  opts.sort = 2;                // 2:auto-choice
  opts.kerpad = 0;              // affects only evaluate_kernel_vector
  opts.kerevalmeth = kerevalmeth;
  opts.upsampfac = upsampfac;
  opts.sort_threads = 0;        // 0:auto-choice
  opts.max_subproblem_size = (BIGINT)1e4;  // was larger (1e5, slightly worse)
  opts.flags = 0;               // 0:no timing flags
  opts.debug = 0;               // 0:no debug output

  // Set kernel width w (aka ns) and ES kernel beta parameter, in opts...
  int ns = std::ceil(-log10(eps/(FLT)10.0));   // 1 digit per power of ten
  // (FLT)10.0 is cosmetic: sng & dbl desire same ns for bdry cases eps=10^{-n}
  if (upsampfac!=(FLT)2.0)           // override ns when custom sigma
    ns = std::ceil(-log(eps) / (PI*sqrt(1-1/upsampfac)));  // formula, gamma=1
  ns = max(2,ns);               // (we don't have ns=1 version yet)
  if (ns>MAX_NSPREAD) {         // clip to match allocated arrays
    fprintf(stderr,"setup_spreader: warning, kernel width ns=%d clipped to max %d; will not match requested eps!\n",ns,MAX_NSPREAD);
    ns = MAX_NSPREAD;
  }
  opts.nspread = ns;
  opts.ES_halfwidth=(FLT)ns/2;   // constants to help ker eval (except Horner)
  opts.ES_c = 4.0/(FLT)(ns*ns);

  FLT betaoverns = 2.30;         // gives decent betas for default sigma=2.0
  if (ns==2) betaoverns = 2.20;  // some small-width tweaks...
  if (ns==3) betaoverns = 2.26;
  if (ns==4) betaoverns = 2.38;
  if (upsampfac!=2.0) {          // again, override beta for custom sigma
    FLT gamma=0.97;              // must match devel/gen_all_horner_C_code.m
    betaoverns = gamma*PI*(1-1/(2*upsampfac));  // formula based on cutoff
  }
  opts.ES_beta = betaoverns * (FLT)ns;    // set the kernel beta parameter
  //fprintf(stderr,"setup_spreader: eps=%.3g sigma=%.6f, chose ns=%d beta=%.6f\n",(double)eps,(double)upsampfac,ns,(double)opts.ES_beta); // user hasn't set debug yet
  return 0;
}

FLT evaluate_kernel(FLT x, const spread_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{
  if (abs(x)>=opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp(opts.ES_beta * sqrt(1.0 - opts.ES_c*x*x));
}

FLT evaluate_kernel_noexp(FLT x, const spread_opts &opts)
// Version of the above just for timing purposes - gives wrong answer!!!
{
  if (abs(x)>=opts.ES_halfwidth)
    return 0.0;
  else {
    FLT s = sqrt(1.0 - opts.ES_c*x*x);
    //  return sinh(opts.ES_beta * s)/s; // roughly, backward K-B kernel of NFFT
        return opts.ES_beta * s;
  }
}

static inline void set_kernel_args(FLT *args, FLT x, const spread_opts& opts)
// Fills vector args[] with kernel arguments x, x+1, ..., x+ns-1.
// needed for the vectorized kernel eval of Ludvig af K.
{
  int ns=opts.nspread;
  for (int i=0; i<ns; i++)
    args[i] = x + (FLT) i;
}

static inline void evaluate_kernel_vector(FLT *ker, FLT *args, const spread_opts& opts, const int N)
/* Evaluate ES kernel for a vector of N arguments; by Ludvig af K.
   If opts.kerpad true, args and ker must be allocated for Npad, and args is
   written to (to pad to length Npad), only first N outputs are correct.
   Barnett 4/24/18 option to pad to mult of 4 for better SIMD vectorization.

   Obsolete (replaced by Horner), but keep around for experimentation since
   works for arbitrary beta. Formula must match reference implementation. */
{
  FLT b = opts.ES_beta;
  FLT c = opts.ES_c;
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
      ker[i] = b * sqrt(1.0 - c*args[i]*args[i]);
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
    if (abs(args[i])>=opts.ES_halfwidth) ker[i] = 0.0;
}

static inline void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w,
					  const spread_opts &opts)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
  if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
    FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
    // insert the auto-generated code which expects z, w args, writes to ker...
    if (opts.upsampfac==2.0) {     // floating point equality is fine here
#include "ker_horner_allw_loop.c"
    } else if (opts.upsampfac==1.25) {
#include "ker_lowupsampfac_horner_allw_loop.c"
    } else
      fprintf(stderr,"eval_kernel_vec_Horner: unknown upsampfac, failed!\n");
  }
}

void interp_line(FLT *target,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns)
// 1D interpolate complex values from du array to out, using real weights
// ker[0] through ker[ns-1]. out must be size 2 (real,imag), and du
// of size 2*N1 (alternating real,imag). i1 is the left-most index in [0,N1)
// Periodic wrapping in the du array is applied, assuming N1>=ns.
// dx is index into ker array, j index in complex du (data_uniform) array.
// Barnett 6/15/17
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
// 2D interpolate complex values from du (uniform grid data) array to out value,
// using ns*ns square of real weights
// in ker. out must be size 2 (real,imag), and du
// of size 2*N1*N2 (alternating real,imag). i1 is the left-most index in [0,N1)
// and i2 the bottom index in [0,N2).
// Periodic wrapping in the du array is applied, assuming N1,N2>=ns.
// dx,dy indices into ker array, j index in complex du array.
// Barnett 6/16/17
{
  FLT out[] = {0.0, 0.0};
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2) {  // no wrapping: avoid ptrs
    for (int dy=0; dy<ns; dy++) {
      BIGINT j = N1*(i2+dy) + i1;
      for (int dx=0; dx<ns; dx++) {
	FLT k = ker1[dx]*ker2[dy];
	out[0] += du[2*j] * k;
	out[1] += du[2*j+1] * k;
	++j;
      }
    }
  } else {                         // wraps somewhere: use ptr list (slower)
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
// 3D interpolate complex values from du (uniform grid data) array to out value,
// using ns*ns*ns cube of real weights
// in ker. out must be size 2 (real,imag), and du
// of size 2*N1*N2*N3 (alternating real,imag). i1 is the left-most index in
// [0,N1), i2 the bottom index in [0,N2), i3 lowest in [0,N3).
// Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
// dx,dy,dz indices into ker array, j index in complex du array.
// Barnett 6/16/17
{
  FLT out[] = {0.0, 0.0};  
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2 && i3>=0 && i3+ns<=N3) {
    // no wrapping: avoid ptrs
    for (int dz=0; dz<ns; dz++) {
      BIGINT oz = N1*N2*(i3+dz);        // offset due to z
      for (int dy=0; dy<ns; dy++) {
	BIGINT j = oz + N1*(i2+dy) + i1;
	FLT ker23 = ker2[dy]*ker3[dz];
	for (int dx=0; dx<ns; dx++) {
	  FLT k = ker1[dx]*ker23;
	  out[0] += du[2*j] * k;
	  out[1] += du[2*j+1] * k;
	  ++j;
	}
      }
    }
  } else {                         // wraps somewhere: use ptr list (slower)
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

void spread_subproblem_1d(BIGINT N1,FLT *du,BIGINT M,
			  FLT *kx,FLT *dd,
			  const spread_opts& opts)
/* spreader from dd (NU) to du (uniform) in 1D without wrapping.
   kx (size M) are NU locations in [0,N1]
   dd (size M complex) are source strengths
   du (size N1) is uniform output array.

   This a naive loop w/ Ludvig's eval_ker_vec.
*/
{
  int ns=opts.nspread;
  FLT ns2 = (FLT)ns/2;          // half spread width
  for (BIGINT i=0;i<2*N1;++i)
    du[i] = 0.0;
  FLT kernel_args[MAX_NSPREAD];
  FLT ker[MAX_NSPREAD];
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];            // x1 in [-w/2,-w/2+1]
    if (opts.kerevalmeth==0) {
      set_kernel_args(kernel_args, x1, opts);
      evaluate_kernel_vector(ker, kernel_args, opts, ns);
    } else
      eval_kernel_vec_Horner(ker,x1,ns,opts);
    // critical inner loop: 
    BIGINT j=i1;
    for (int dx=0; dx<ns; ++dx) {
      FLT k = ker[dx];
      du[2*j] += re0*k;
      du[2*j+1] += im0*k;
      ++j;
    }
  }
}

void spread_subproblem_2d(BIGINT N1,BIGINT N2,FLT *du,BIGINT M,
			  FLT *kx,FLT *ky,FLT *dd,
			  const spread_opts& opts)
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   kx,ky (size M) are NU locations in [0,N1],[0,N2]
   dd (size M complex) are source strengths
   du (size N1*N2) is uniform output array
 */
{
  int ns=opts.nspread;
  FLT ns2 = (FLT)ns/2;          // half spread width
  for (BIGINT i=0;i<2*N1*N2;++i)
    du[i] = 0.0;
  FLT kernel_args[2*MAX_NSPREAD];
  FLT kernel_values[2*MAX_NSPREAD];
  FLT *ker1 = kernel_values;
  FLT *ker2 = kernel_values + ns;  
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];
    FLT x2 = (FLT)i2 - ky[i];
    if (opts.kerevalmeth==0) {
      set_kernel_args(kernel_args, x1, opts);
      set_kernel_args(kernel_args+ns, x2, opts);
      evaluate_kernel_vector(kernel_values, kernel_args, opts, 2*ns);
    } else {
      eval_kernel_vec_Horner(ker1,x1,ns,opts);
      eval_kernel_vec_Horner(ker2,x2,ns,opts);
    }
    // Combine kernel with complex source value to simplify inner loop
    FLT ker1val[2*MAX_NSPREAD];
    for (int i = 0; i < ns; i++) {
      ker1val[2*i] = re0*ker1[i];
      ker1val[2*i+1] = im0*ker1[i];	
    }    
    // critical inner loop:
    for (int dy=0; dy<ns; ++dy) {
      BIGINT j = N1*(i2+dy) + i1;
      FLT kerval = ker2[dy];
      FLT *trg = du+2*j;
      for (int dx=0; dx<2*ns; ++dx) {
	trg[dx] += kerval*ker1val[dx];
      }	
    }
  }
}

void spread_subproblem_3d(BIGINT N1,BIGINT N2,BIGINT N3,FLT *du,BIGINT M,
			  FLT *kx,FLT *ky,FLT *kz,FLT *dd,
			  const spread_opts& opts)
/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
   kx,ky,kz (size M) are NU locations in [0,N1],[0,N2],[0,N3]
   dd (size M complex) are source strengths
   du (size N1*N2*N3) is uniform output array
 */
{
  int ns=opts.nspread;
  FLT ns2 = (FLT)ns/2;          // half spread width
  for (BIGINT i=0;i<2*N1*N2*N3;++i)
    du[i] = 0.0;
  FLT kernel_args[3*MAX_NSPREAD];
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in all three directions in a single kernel evaluation call.
  FLT kernel_values[3*MAX_NSPREAD];
  FLT *ker1 = kernel_values;
  FLT *ker2 = kernel_values + ns;
  FLT *ker3 = kernel_values + 2*ns;  
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
    BIGINT i3 = (BIGINT)std::ceil(kz[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];
    FLT x2 = (FLT)i2 - ky[i];
    FLT x3 = (FLT)i3 - kz[i];
    if (opts.kerevalmeth==0) {
      set_kernel_args(kernel_args, x1, opts);
      set_kernel_args(kernel_args+ns, x2, opts);
      set_kernel_args(kernel_args+2*ns, x3, opts);
      evaluate_kernel_vector(kernel_values, kernel_args, opts, 3*ns);
    } else {
      eval_kernel_vec_Horner(ker1,x1,ns,opts);
      eval_kernel_vec_Horner(ker2,x2,ns,opts);
      eval_kernel_vec_Horner(ker3,x3,ns,opts);
    }
    // Combine kernel with complex source value to simplify inner loop
    FLT ker1val[2*MAX_NSPREAD];
    for (int i = 0; i < ns; i++) {
      ker1val[2*i] = re0*ker1[i];
      ker1val[2*i+1] = im0*ker1[i];	
    }    
    // critical inner loop:
    for (int dz=0; dz<ns; ++dz) {
      BIGINT oz = N1*N2*(i3+dz);        // offset due to z
      for (int dy=0; dy<ns; ++dy) {
	BIGINT j = oz + N1*(i2+dy) + i1;
	FLT kerval = ker2[dy]*ker3[dz];
	FLT *trg = du+2*j;
	for (int dx=0; dx<2*ns; ++dx) {
	  trg[dx] += kerval*ker1val[dx];
	}	
      }
    }
  }
}

void add_wrapped_subgrid(BIGINT offset1,BIGINT offset2,BIGINT offset3,
			 BIGINT size1,BIGINT size2,BIGINT size3,BIGINT N1,
			 BIGINT N2,BIGINT N3,FLT *data_uniform, FLT *du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   size1,2,3 give the size of subgrid.
   Works in all dims. Thread-safe since must be called inside omp critical.
   Barnett 3/27/18 made separate routine, tried to speed up inner loop.
*/
{
  std::vector<BIGINT> o2(size2), o3(size3);
  BIGINT y=offset2, z=offset3;    // fill wrapped ptr lists in slower dims y,z...
  for (int i=0; i<size2; ++i) {
    if (y<0) y+=N2;
    if (y>=N2) y-=N2;
    o2[i] = y++;
  }
  for (int i=0; i<size3; ++i) {
    if (z<0) z+=N3;
    if (z>=N3) z-=N3;
    o3[i] = z++;
  }
  BIGINT nlo = (offset1<0) ? -offset1 : 0;          // # wrapping below in x
  BIGINT nhi = (offset1+size1>N1) ? offset1+size1-N1 : 0;    // " above in x
  // this triple loop works in all dims
  for (int dz=0; dz<size3; dz++) {       // use ptr lists in each axis
    BIGINT oz = N1*N2*o3[dz];            // offset due to z (0 in <3D)
    for (int dy=0; dy<size2; dy++) {
      BIGINT oy = oz + N1*o2[dy];        // off due to y & z (0 in 1D)
      FLT *out = data_uniform + 2*oy;
      FLT *in  = du0 + 2*size1*(dy + size2*dz);   // ptr to subgrid array
      BIGINT o = 2*(offset1+N1);         // 1d offset for output
      for (int j=0; j<2*nlo; j++)        // j is really dx/2 (since re,im parts)
	out[j+o] += in[j];
      o = 2*offset1;
      for (int j=2*nlo; j<2*(size1-nhi); j++)
	out[j+o] += in[j];
      o = 2*(offset1-N1);
      for (int j=2*(size1-nhi); j<2*size1; j++)
      	out[j+o] += in[j];
    }
  }
}

void bin_sort_singlethread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
	      double bin_size_x,double bin_size_y,double bin_size_z, int debug)
/* Returns permutation of all nonuniform points with good RAM access,
 * ie less cache misses for spreading, in 1D, 2D, or 3D. Singe-threaded version.
 *
 * This is achieved by binning into cuboids (of given bin_size)
 * then reading out the indices within
 * these boxes in the natural box order (x fastest, y med, z slowest).
 * Finally the permutation is inverted.
 * 
 * Inputs: M - number of input NU points.
 *         kx,ky,kz - length-M arrays of real coords of NU pts, in the valid
 *                    range for RESCALE, which includes [0,N1], [0,N2], [0,N3]
 *                    respectively, if pirange=0; or [-pi,pi] if pirange=1.
 *         N1,N2,N3 - ranges of NU coords (set N2=N3=1 for 1D, N3=1 for 2D)
 *         bin_size_x,y,z - what binning box size to use in each dimension
 *                    (in rescaled coords where ranges are [0,Ni] ).
 *                    For 1D, only bin_size_x is used; for 2D, it and bin_size_y
 * Output:
 *         writes to ret a vector list of indices, each in the range 0,..,M-1.
 *         Thus, ret must have been allocated for M BIGINTs.
 *
 * Notes: I compared RAM usage against declaring an internal vector and passing
 * back; the latter used more RAM and was slower.
 * Avoided the bins array, as in JFM's spreader of 2016,
 * tidied up, early 2017, Barnett.
 *
 * Timings: 3s for M=1e8 NU pts on 1 core of i7; 5s on 1 core of xeon.
 */
{
  bool isky=(N2>1), iskz=(N3>1);  // ky,kz avail? (cannot access if not)
  BIGINT nbins1=N1/bin_size_x+1, nbins2, nbins3;
  nbins2 = isky ? N2/bin_size_y+1 : 1;
  nbins3 = iskz ? N3/bin_size_z+1 : 1;
  BIGINT nbins = nbins1*nbins2*nbins3;

  std::vector<BIGINT> counts(nbins,0);  // count how many pts in each bin
  for (BIGINT i=0; i<M; i++) {
    // find the bin index in however many dims are needed
    BIGINT i1=RESCALE(kx[i],N1,pirange)/bin_size_x, i2=0, i3=0;
    if (isky) i2 = RESCALE(ky[i],N2,pirange)/bin_size_y;
    if (iskz) i3 = RESCALE(kz[i],N3,pirange)/bin_size_z;
    BIGINT bin = i1+nbins1*(i2+nbins2*i3);
    counts[bin]++;
  }
  std::vector<BIGINT> offsets(nbins);   // cumulative sum of bin counts
  offsets[0]=0;     // do: offsets = [0 cumsum(counts(1:end-1)]
  for (BIGINT i=1; i<nbins; i++)
    offsets[i]=offsets[i-1]+counts[i-1];
  
  std::vector<BIGINT> inv(M);           // fill inverse map
  for (BIGINT i=0; i<M; i++) {
    // find the bin index (again! but better than using RAM)
    BIGINT i1=RESCALE(kx[i],N1,pirange)/bin_size_x, i2=0, i3=0;
    if (isky) i2 = RESCALE(ky[i],N2,pirange)/bin_size_y;
    if (iskz) i3 = RESCALE(kz[i],N3,pirange)/bin_size_z;
    BIGINT bin = i1+nbins1*(i2+nbins2*i3);
    BIGINT offset=offsets[bin];
    offsets[bin]++;
    inv[i]=offset;
  }
  // invert the map, writing to output pointer (writing pattern is random)
  for (BIGINT i=0; i<M; i++)
    ret[inv[i]]=i;
}

void bin_sort_multithread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
	      double bin_size_x,double bin_size_y,double bin_size_z, int debug)
/* Mostly-OpenMP'ed version of bin_sort.
   For documentation see: bin_sort_singlethread.
   Caution: when M (# NU pts) << N (# U pts), is SLOWER than single-thread.
   Barnett 2/8/18
   Todo: if debug, print timing breakdowns
 */
{
  bool isky=(N2>1), iskz=(N3>1);  // ky,kz avail? (cannot access if not)
  BIGINT nbins1=N1/bin_size_x+1, nbins2, nbins3;
  nbins2 = isky ? N2/bin_size_y+1 : 1;
  nbins3 = iskz ? N3/bin_size_z+1 : 1;
  BIGINT nbins = nbins1*nbins2*nbins3;
  int nt = MIN(M,MY_OMP_GET_MAX_THREADS());      // printf("\tnt=%d\n",nt);
  std::vector<BIGINT> brk(nt+1); // start NU pt indices per thread

  // distribute the M NU pts to threads once & for all...
  for (int t=0; t<=nt; ++t)
    brk[t] = (BIGINT)(0.5 + M*t/(double)nt);   // start index for t'th chunk
  std::vector<BIGINT> counts(nbins,0);  // counts of how many pts in each bin
  std::vector< std::vector<BIGINT> > ot(nt,counts); // offsets per thread, nt * nbins
  {    // scope for ct, the 2d array of counts in bins for each threads's NU pts
    std::vector< std::vector<BIGINT> > ct(nt,counts);   // nt * nbins, init to 0

#pragma omp parallel
    {
      int t = MY_OMP_GET_THREAD_NUM();
      if (t<nt) {                      // could be nt < actual # threads
	//printf("\tt=%d: [%d,%d]\n",t,jlo[t],jhi[t]);
	for (BIGINT i=brk[t]; i<brk[t+1]; i++) {
	  // find the bin index in however many dims are needed
	  BIGINT i1=RESCALE(kx[i],N1,pirange)/bin_size_x, i2=0, i3=0;
	  if (isky) i2 = RESCALE(ky[i],N2,pirange)/bin_size_y;
	  if (iskz) i3 = RESCALE(kz[i],N3,pirange)/bin_size_z;
	  BIGINT bin = i1+nbins1*(i2+nbins2*i3);
	  ct[t][bin]++;               // no clash btw threads
	}
      }
    }
    for (int t=0; t<nt; ++t)
#pragma omp parallel for schedule(dynamic,10000)     // probably useless
      for (BIGINT b=0; b<nbins; ++b)
	counts[b] += ct[t][b];
    
    std::vector<BIGINT> offsets(nbins);   // cumulative sum of bin counts
    offsets[0]=0;
    // do: offsets = [0 cumsum(counts(1:end-1))].
    // multithread? (do chunks in 2 pass) but not many bins; don't bother...
    for (BIGINT i=1; i<nbins; i++)
      offsets[i]=offsets[i-1]+counts[i-1];
    
    for (BIGINT b=0; b<nbins; ++b)  // now build offsets for each thread & bin:
      ot[0][b] = offsets[b];                       // init
    for (int t=1; t<nt; ++t)
#pragma omp parallel for schedule(dynamic,10000)     // probably useless
      for (BIGINT b=0; b<nbins; ++b)
	ot[t][b] = ot[t-1][b]+ct[t-1][b];        // cumsum along t axis
    
  } // scope frees up ct here
  
  std::vector<BIGINT> inv(M);           // fill inverse map
#pragma omp parallel
  {
    int t = MY_OMP_GET_THREAD_NUM();
    if (t<nt) {                      // could be nt < actual # threads
      for (BIGINT i=brk[t]; i<brk[t+1]; i++) {
	// find the bin index (again! but better than using RAM)
	BIGINT i1=RESCALE(kx[i],N1,pirange)/bin_size_x, i2=0, i3=0;
	if (isky) i2 = RESCALE(ky[i],N2,pirange)/bin_size_y;
	if (iskz) i3 = RESCALE(kz[i],N3,pirange)/bin_size_z;
	BIGINT bin = i1+nbins1*(i2+nbins2*i3);
	inv[i]=ot[t][bin];   // get the offset for this NU pt and thread
	ot[t][bin]++;               // no clash
      }
    }
  }
  // invert the map, writing to output pointer (writing pattern is random)
#pragma omp parallel for schedule(dynamic,10000)
  for (BIGINT i=0; i<M; i++)
    ret[inv[i]]=i;
}


void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,BIGINT &size2,BIGINT &size3,BIGINT M,FLT* kx,FLT* ky,FLT* kz,int ns,int ndims)
/* Writes out the offsets and sizes of the subgrid defined by the
   nonuniform points and the spreading diameter approx ns/2.
   Requires O(M) effort to find the k array bnds. Works in all dims 1,2,3.
   Must return offset 0 and size 1 for each unused dimension.
   Grid has been made tight to the kernel point choice using identical ceil
   operations.  6/16/17
*/
{
  FLT ns2 = (FLT)ns/2;
  // compute the min/max of the k-space locations of the nonuniform points
  FLT min_kx,max_kx;
  arrayrange(M,kx,&min_kx,&max_kx);
  BIGINT a1=std::ceil(min_kx-ns2);
  BIGINT a2=std::ceil(max_kx-ns2)+ns-1;
  offset1=a1;
  size1=a2-a1+1;
  if (ndims>1) {
    FLT min_ky,max_ky;
    arrayrange(M,ky,&min_ky,&max_ky);
    BIGINT b1=std::ceil(min_ky-ns2);
    BIGINT b2=std::ceil(max_ky-ns2)+ns-1;
    offset2=b1;
    size2=b2-b1+1;
  } else {
    offset2=0;
    size2=1;
  }
  if (ndims>2) {
    FLT min_kz,max_kz;
    arrayrange(M,kz,&min_kz,&max_kz);
    BIGINT c1=std::ceil(min_kz-ns2);
    BIGINT c2=std::ceil(max_kz-ns2)+ns-1;
    offset3=c1;
    size3=c2-c1+1;
  } else {
    offset3=0;
    size3=1;
  }
}
