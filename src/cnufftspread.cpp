#include "cnufftspread.h"

#include <stdlib.h>
#include <vector>
#include <math.h>

// declarations of internal functions...
std::vector<long> compute_sort_indices(long M,double *kx, double *ky,
				       double *kz,long N1,long N2,long N3);
void compute_kernel_values(double frac1,double frac2,double frac3,
			   const spread_opts &opts, int *r1, int *r2, int *r3,
			   double *ker, int ndims);
bool set_thread_index_box(long *i1th,long *i2th,long *i3th,long N1,long N2,
			  long N3,int th,int nth, const spread_opts &opts,
			  int ndims);
bool ind_might_affect_interval(long i,long N,long *ith,long nspread);
bool wrapped_range_in_interval(long i,int *R,long *ith,long N,int *r);

int cnufftspread(
        long N1, long N2, long N3, double *data_uniform,
        long M, double *kx, double *ky, double *kz, double *data_nonuniform,
        spread_opts opts)
/* Spreader for 1, 2, or 3 dimensions.
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

   Notes:
   No particular normalization of the spreading kernel is assumed, since this
   will cancel in the NUFFT anyway. Uniform (U) points are centered at coords
   [0,1,...,N1-1] in 1D, analogously in 2D and 3D. They are stored in x
   fastest, y medium, z slowest ordering, up to however many
   dimensions are relevant; note that this is Fortran-style ordering for an
   array f(x,y,z), but C style for f[z][y][x]. This is to match the fortran
   interface of the original CMCL libraries.
   Non-uniform (NU) points kx,ky,kz are real and must be in the range [0,N1]
   in 1D, analogously in 2D and 3D, otherwise an error is returned and no
   calculation is done.
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
                1D, only kx and ky used in 2D). These must lie in the box,
		ie 0<=kx<=N1 etc.
   opts - object controlling spreading method and text output, has fields
   including:
        spread_direction=1, spreads from nonuniform input to uniform output, or
        spread_direction=2, interpolates ("spread transpose") from uniform input
                            to nonuniform output.
	sort_data - (boolean) whether to sort NU points using natural yz-grid
	            ordering.
	debug = 0: no text output, 1: some openmp output, 2: mega output
	           (each NU pt)
        checkerboard = 0: for dir=1, split top dimension only,
	               1: checkerboard top two (not yet implemented)

   Inputs/Outputs:
   data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
   data_nonuniform - input strengths of the sources (dir=1)
                     OR output values at targets (dir=2)
   Ouputs:
   returned value - error status indicator:
      0 : success.
      1 : one or more non-trivial box dimensions is less than 2.nspread.
      2 : nonuniform points outside range [0,Nm] in at least one dimension
          m=1,2,3.
      3 : out of memory for the internal sorting arrays.
      4 : invalid opts.spread_direction

   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
*/
{ 
  // Input checking: cuboid not too small for spreading
  long minN = 2*opts.nspread;
  if (N1<minN || (N2>1 && N2<minN) || (N3>1 && N3<minN)) {
    fprintf(stderr,"error: one or more non-trivial box dims is less than 2.nspread!\n");
    return 1;
  }
  if (opts.spread_direction!=1 && opts.spread_direction!=2) {
    fprintf(stderr,"opts.spread_direction must be 1 or 2!\n");
    return 4;
  }
  int ndims = 1;                 // decide ndims: 1,2 or 3
  if (N2>1) ++ndims;
  if (N3>1) ++ndims;

  // declarations can't be in try block, so use resize below...
  std::vector<double> kx2,ky2,kz2,data_nonuniform2;
  std::vector<long> sort_indices;
  try {    // alloc the big workspaces in a graceful way
    kx2.resize(M); ky2.resize(M); kz2.resize(M); data_nonuniform2.resize(M*2);
    sort_indices.resize(M);
  }
  catch(std::bad_alloc &e) {
    fprintf(stderr,"cnufftspread cannot alloc arrays!\n");
    return 3;
  }
  // MY_OMP_SET_NUM_THREADS(1); // for debug; also may set via shell env var OMP_NUM_THREADS
  
  // Sort NU pts once and for all: sorted answer will be k{xyz}2 and data_nonuniform2
  // We also zero unused coordinates (for 1D or 2D cases) and check bounds:
  CNTime timer; timer.start();
  if (opts.sort_data)
    sort_indices=compute_sort_indices(M,kx,ky,kz,N1,N2,N3); // a good perm of NU pts
  else {
    for (long i=0; i<M; i++)                  // (omp no speed-up here)
      sort_indices[i]=i;                      // the identity permutation!
  }
  bool bnderr = false;
  #pragma omp parallel for schedule(dynamic)
  for (long i=0; i<M; i++) {  // (omp has 20% effect on dir=1 case, so use it)
    long jj=sort_indices[i];
    kx2[i]=kx[jj];
    if (N2==1)                // safely kill not-needed coords
      ky2[i] = 0.0;
    else
      ky2[i]=ky[jj];
    if (N3==1)	
      kz2[i] = 0.0;
    else
      kz2[i]=kz[jj];
    // while we're here, check bounds of NU pts (after coords were killed)...
    if (kx2[i]<0.0 || kx2[i]>N1 || ky2[i]<0.0 || ky2[i]>N2 || kz2[i]<0.0 || kz2[i]>N3)
      bnderr = true;
    if (opts.spread_direction==1) {  // note this also sorts incoming strengths
      data_nonuniform2[i*2]=data_nonuniform[jj*2];      // real
      data_nonuniform2[i*2+1]=data_nonuniform[jj*2+1];  // imag
    }
  }
  double t=timer.elapsedsec();
  if (opts.debug) printf("sort time (sort_data=%d): %.3g s\n",(int)opts.sort_data,t);
  if (bnderr) {
    fprintf(stderr,"error: at least one nonuniform point not in range [0,N1] x ... !\n");
    return 2;
  }

  // set up spreading kernel index bounds in each dim, relative to bottom left corner:
  int ns=opts.nspread;
  double ns2 = (double)ns/2;          // half spread width, used later
  int R1[2]={0,ns-1};                 // we always spread in x
  int R2[2]={0,0}; int R3[2]={0,0};
  if (N2>1) R2[1] = ns-1;             // also spread in y
  if (N3>1) R3[1] = ns-1;             // also spread in z
  if (opts.debug) printf("R box: %d %d %d %d %d %d\n",R1[0],R1[1],R2[0],R2[1],R3[0],R3[1]);
  
  if (opts.spread_direction==1) {  // zero complex output array ready to accumulate...
    timer.restart();
    for (long i=0; i<2*N1*N2*N3; i++) data_uniform[i]=0.0;    // would be ruined by omp!
    if (opts.debug) printf("zeroing output array: %.3g s\n",timer.elapsedsec());
  }

#pragma omp parallel
  {  // omp block : release a cadre of threads.
    int nth = MY_OMP_GET_NUM_THREADS(), th = MY_OMP_GET_THREAD_NUM();
    if (th==0 && opts.debug) printf("spreading dir=%d, %d threads\n", opts.spread_direction,nth);
    
    if (opts.spread_direction==1) { // ==================== direction 1 =======
      
      BIGINT i1th[2],i2th[2],i3th[2];   // get this thread's (fixed) grid index writing box...
      bool thread_has_task = set_thread_index_box(i1th,i2th,i3th,N1,N2,N3,th,nth,opts,ndims);
      if (opts.debug) printf("th=%d: N1=%ld,N2=%ld,N3=%ld,has_task=%d\n",th,N1,N2,N3,(int)thread_has_task);
      if (thread_has_task) {
	if (opts.debug) printf("th=%d ind box: %ld %ld %ld %ld %ld %ld\n",th,i1th[0],i1th[1],i2th[0],i2th[1],i3th[0],i3th[1]);
	BIGINT c = 0;   // debug count how many NU pts each thread does
	
	for (BIGINT i=0; i<M; i++) {  // main loop over NU pts, spread each to U grid
	  // (note every thread does this loop, but only sometimes write to the grid)
	  BIGINT i1=(BIGINT)std::ceil(kx2[i]-ns2); // leftmost x grid index
	  BIGINT i2=(BIGINT)std::ceil(ky2[i]-ns2); // lowest y grid index
	  BIGINT i3=(BIGINT)std::ceil(kz2[i]-ns2); // lowest z grid index
	  int r1[2],r2[2],r3[2]; // lower & upper rel ind bnds restricted to thread's box
	  // set the r1,r2,r3 bounds and decide if NU point i affects this thread's box:
	  bool i_affects_box = wrapped_range_in_interval(i1,R1,i1th,N1,r1) &&
	    wrapped_range_in_interval(i2,R2,i2th,N2,r2) &&
	    wrapped_range_in_interval(i3,R3,i3th,N3,r3);

	  if (i_affects_box) {
	    if (opts.debug>1) printf("th=%d r box: %d %d %d %d %d %d\n",th,r1[0],r1[1],r2[0],r2[1],r3[0],r3[1]);
	    ++c;     // debug
	    // set up indices for each dim ahead of time using by-hand modulo wrapping
	    // periodically up to +-1 period:    
	    BIGINT j1_array[MAX_NSPREAD],j2_array[MAX_NSPREAD],j3_array[MAX_NSPREAD];
	    j2_array[0] = 0; j3_array[0] = 0;             // needed for unused dims
	    double x1=(double)i1-kx2[i], x2, x3;          // real shifts of ker center
	    for (int dx=r1[0]; dx<=r1[1]; dx++) {
	      BIGINT j=i1+dx; if (j<0) j+=N1; if (j>=N1) j-=N1;
	      j1_array[dx-r1[0]]=j;
	    }
	    if (ndims>1) {              // 2d stuff
	      x2=(double)i2-ky2[i];
	      for (int dy=r2[0]; dy<=r2[1]; dy++) {
		BIGINT j=i2+dy; if (j<0) j+=N2; if (j>=N2) j-=N2;
		j2_array[dy-r2[0]]=j;
	      }
	    }
	    if (ndims>2) {              // 2d or 3d stuff
	      x3=(double)i3-kz2[i];
	      for (int dz=r3[0]; dz<=r3[1]; dz++) {
		BIGINT j=i3+dz; if (j<0) j+=N3; if (j>=N3) j-=N3;
		j3_array[dz-r3[0]]=j;
	      }
	    }
	    // from now on in dir=1, we use "small r" instead of "big R" bounds...
	    // Eval only ker vals needed for overall dim and this thread's index box
	    double kernel_values[MAX_NSPREAD*MAX_NSPREAD*MAX_NSPREAD];
	    compute_kernel_values(x1,x2,x3,opts,r1,r2,r3,kernel_values,ndims);
	    double re0=data_nonuniform2[i*2];
	    double im0=data_nonuniform2[i*2+1];
  	    long aa = 0;
	    for (int dz=r3[0]; dz<=r3[1]; dz++) {
	      BIGINT o3=N1*N2*j3_array[dz-r3[0]];  // use precomp index lists in each dim
	      for (int dy=r2[0]; dy<=r2[1]; dy++) {
		BIGINT o2=o3 + N1*j2_array[dy-r2[0]];
		for (int dx=r1[0]; dx<=r1[1]; dx++) {
		  BIGINT jjj=o2 + j1_array[dx-r1[0]];
		  double kern0=kernel_values[aa];     // kernel vals swept in proper order
		  data_uniform[jjj*2]   += re0*kern0; // accumulate complex value to grid
		  data_uniform[jjj*2+1] += im0*kern0;
		  aa++;
		}
	      }
	    }
	  }
	}
	if (opts.debug) printf("th %d did %ld NU pts.\n",th,c);
      }
    } else {                      // ==================== direction 2 ===============
#pragma omp for schedule(dynamic)   // assign threads to NU targ pts, easy
      for (BIGINT i=0; i<M; i++) {  // main loop over NU pts targets, interp each from U
	// set up indices for each dim ahead of time using by-hand modulo wrapping
	// periodically up to +-1 period:
	BIGINT j1_array[MAX_NSPREAD],j2_array[MAX_NSPREAD],j3_array[MAX_NSPREAD];
	j2_array[0] = 0; j3_array[0] = 0;             // needed for unused dims
	BIGINT i1=(BIGINT)std::ceil(kx2[i]-ns2), i2=0, i3=0; // leftmost grid index
	double x1=(double)i1-kx2[i], x2, x3;          // real-valued shifts of ker center
	for (int dx=R1[0]; dx<=R1[1]; dx++) {
	  BIGINT j=i1+dx; if (j<0) j+=N1; if (j>=N1) j-=N1;
	  j1_array[dx-R1[0]]=j;                       // redundant since R1[0]=0 always
	}
	if (ndims>1) {              // 2d stuff
	  i2=(BIGINT)std::ceil(ky2[i]-ns2); // lowest y grid index, or 0 if unused dim
	  x2=(double)i2-ky2[i];
	  for (int dy=R2[0]; dy<=R2[1]; dy++) {
	    BIGINT j=i2+dy; if (j<0) j+=N2; if (j>=N2) j-=N2;
	    j2_array[dy-R2[0]]=j;
	  }
	}
	if (ndims>2) {              // 2d or 3d stuff
	  i3=(BIGINT)std::ceil(kz2[i]-ns2); // lowest z grid index, or 0 if unused dim
	  x3=(double)i3-kz2[i];
	  for (int dz=R3[0]; dz<=R3[1]; dz++) {
	    BIGINT j=i3+dz; if (j<0) j+=N3; if (j>=N3) j-=N3;
	    j3_array[dz-R3[0]]=j;
	  }
	}
	double kernel_values[MAX_NSPREAD*MAX_NSPREAD*MAX_NSPREAD];
	compute_kernel_values(x1,x2,x3,opts,R1,R2,R3,kernel_values,ndims);
	double re0=0.0, im0=0.0;
	int aa = 0;
	for (int dz=R3[0]; dz<=R3[1]; dz++) {
	  BIGINT o3=N1*N2*j3_array[dz-R3[0]];  // use precomputed index lists in each dim
	  for (int dy=R2[0]; dy<=R2[1]; dy++) {
	    BIGINT o2=o3 + N1*j2_array[dy-R2[0]];
	    for (int dx=R1[0]; dx<=R1[1]; dx++) {
	      BIGINT jjj=o2 + j1_array[dx-R1[0]];
	      double kern0=kernel_values[aa];
	      re0 += data_uniform[jjj*2]*kern0;  // interpolate using kernel as weights
	      im0 += data_uniform[jjj*2+1]*kern0;
	      aa++;
	    }
	  }
	}
	data_nonuniform2[i*2]   = re0;     // copy out the accumulated complex value
	data_nonuniform2[i*2+1] = im0;
      }
      // "unsort" values which were dumped to NU output pts
#pragma omp for schedule(dynamic)   // assign threads to NU targ pts
      for (BIGINT i=0; i<M; i++) {
	BIGINT jj=sort_indices[i];
	data_nonuniform[jj*2]=data_nonuniform2[i*2];
	data_nonuniform[jj*2+1]=data_nonuniform2[i*2+1];
      }
    }
  } // omp block
  return 0;
}

bool set_thread_index_box(long *i1th,long *i2th,long *i3th,long N1,long N2,long N3,
			  int th,int nth, const spread_opts &opts, int ndims)
/* Decides how the uniform grid is to be partitioned into cuboids for each thread
 * (for spread_direction=1 only, ie, writing to the grid).
 *
 * Inputs: N1,N2,N3 dimensions of uniform grid (N2=N3=1 for 1d, N3=1 for 2d, otherwise 3d)
 *         nth - number of total threads which must cover the grid.
 *         th - number of the thread we're assigning a cuboid. Must have 0 <= th < nth.
 *         opts - spreading opts structure, only opts.checkerboard used:
 *                 0: slice only top dimension, 1: checkerboard in 2d & 3d (todo)
 *         ndims - 1,2, or 3. (must match N1,N2,N3 choices).
 * Outputs: returned value: true - this thread is given a cuboid
 *                          false - this thread is given no grid points at all
 *                                  (possible only if there's more threads than N).
 *         If returns true, all of the following are meaningful (even in unused dims):
 *         i1th[2] - lower and upper index bounds of cuboid in x, in range 0 to N1-1
 *         i2th[2] - lower and upper index bounds of cuboid in y, in range 0 to N2-1
 *         i3th[2] - lower and upper index bounds of cuboid in z, in range 0 to N3-1
 *
 * todo: see if any speed advantage by matching the compute_sort_indices ordering below.
 * todo: speed test checkerboard division for 3D instead, for nth large and N3 small!
 * Barnett 1/16/17
 */
{
  i1th[0] = 0; i1th[1] = N1-1; // set defaults; for unused dims this gives 0 & 0
  i2th[0] = 0; i2th[1] = N2-1;
  i3th[0] = 0; i3th[1] = N3-1;
  // set Ntop the lowest (in grid order) nontrivial dimension, ie Nd for d = #dims...
  BIGINT ith[2], Ntop = N1, dims=1;
  if (ndims==2) Ntop = N2;
  if (ndims==3) Ntop = N3;
  if (N2==1 || !opts.checkerboard) {  // slice only along one dim
    if (Ntop<nth) {    // we're at a loss to occupy every thread; assign one per grid pt
      ith[0] = ith[1] = th;
      return th<Ntop;
    } else {           // this relies on consistent rounding behavior every time called!
      ith[0] = (BIGINT)(th*(double)Ntop/nth);
      ith[1] = (BIGINT)((th+1)*(double)Ntop/nth - 1);
    }
    // now slice only along the top dim (we keep lines or planes in lower dims)
    if (ndims==1) {
      i1th[0] = ith[0]; i1th[1] = ith[1];
    } else if (ndims==2) {
      i2th[0] = ith[0]; i2th[1] = ith[1];
    } else if (ndims==3) {
      i3th[0] = ith[0]; i3th[1] = ith[1];
    }
    return true;
  } else {
    printf("2d or 3d checkerboard not implemented!\n");
    return false;
  }
}

bool wrapped_range_in_interval(long i,int *R,long *ith,long N,int *r)
  /* returns in r[0], r[1] the lower, upper index limits relative to 1d grid index i
   * that, after N-periodizing, will fall into box with index range ith[0] to ith[1].
   * R[0..1] are the max spreading lower and upper relative index limits.
   * Ie, computes the limits of interval produced by intersecting the set [R[0],R[1]]
   * with the union of all N-periodically shifted copies of [ith[0]-i,ith[1]-i].
   * Output is true if the set of indices is non-empty, false if empty (in which case
   * r[0..1] are undefined). Doesn't have to execute fast. Assumes N <= 2*nspread,
   * and abs(R[0..1]) <= nspread/2, and ith interval either fills [0,N-1] or is smaller
   * in length by at least nspread.
   * This last assumption ensures intersection is a single interval.
   * Barnett 1/16/17
   */
{
  if (ith[1]-ith[0]+1==N) {      // box fills periodic interval: interval is unscathed
    r[0] = R[0]; r[1] = R[1];
    return true;
  }
  long lo=ith[0]-i, hi=ith[1]-i; // ith interval, expressed relative to center index i.
  for (long d=-N;d<=N;d+=N) {   // loop over periodic copies of ith interval
    r[0] = (int)std::max((long)R[0],lo+d);    // clip interval copy to R-spread interval
    r[1] = (int)std::min((long)R[1],hi+d);
    if (r[1]>=r[0]) return true;    // either happens never, or once in which case exit
  }
  return false;
}

std::vector<long> compute_sort_indices(long M,double *kx, double *ky, double *kz,long N1,long N2,long N3)
  /* Returns permutation of the 1, 2 or 3D nonuniform points with good RAM access for the
   * upcoming spreading step.
   *
   * Here "good" means lots of requested blocks of RAM can stay in cache to be reused.
   * Currenty this is achieved by binning into 1-grid-point sized boxes in the yz-plane,
   * with no sorting along x within each box, then reading out the indices within these
   * boxes in the natural box order (y fast, z slow).
   * Finally the permutation map is inverted.
   * 
   * Inputs: M - length of inputs
   *         kx,ky,kz - length-M real numbers in [0,N1], [0,N2], [0,N3]
   *                    respectively.
   * Output: vector list of indices, each in the range 0,..,M-1, which is a good ordering
   *         of the points.
   *
   * Note: apparently in 2D sorts only along y, and in 1D doesn't sort at all (x).
   *
   * todo: fix the 1d case to sort along x dimension.
   * Magland, Dec 2016; Barnett tweaked so doesn't examine ky in 1d, or kz in 1d or 2d.
   */
{
  bool isky=(N2>1), iskz=(N3>1);           // are ky,kz available? cannot access if not!
  std::vector<long> counts(N2*N3);
  for (long j=0; j<N2*N3; j++)
    counts[j]=0;
  for (long i=0; i<M; i++) {
    long i2=isky ? (long)(ky[i]+0.5) : 0;
    if (i2<0) i2=0;
    if (i2>=N2) i2=N2-1;
    
    long i3=iskz ? (long)(kz[i]+0.5) : 0;
    if (i3<0) i3=0;
    if (i3>=N3) i3=N3-1;
    
    counts[i2+N2*i3]++;
  }
  std::vector<long> inds(N2*N3);
  long offset=0;
  for (long j=0; j<N2*N3; j++) {
    inds[j]=offset;
    offset+=counts[j];
  }
  
  std::vector<long> ret_inv(M);
  for (long i=0; i<M; i++) {
    long i2=isky ? (long)(ky[i]+0.5) : 0;
    if (i2<0) i2=0;
    if (i2>=N2) i2=N2-1;
    
    long i3=iskz ? (long)(kz[i]+0.5) : 0;
    if (i3<0) i3=0;
    if (i3>=N3) i3=N3-1;
    
    long jj=inds[i2+N2*i3];
    inds[i2+N2*i3]++;
    ret_inv[i]=jj;
  }
  
  std::vector<long> ret(M);
  for (long i=0; i<M; i++) {
    ret[ret_inv[i]]=i;
  }
  
  return ret;
}

void compute_kernel_values(double x1,double x2,double x3,
			   const spread_opts &opts, int *r1, int *r2, int *r3,
			   double *ker, int ndims)
/* Evaluate spreading kernel values on integer cuboid of grid points
 * shifted from the origin by real vector
 * (x1,x2,x3). The integer ranges are controlled by
 * r1,r2,r3 which each are two-element arrays giving start and end integers
 * in each dimension.  ndims=1,2,3 sets the number of used dimensions.
 * The output ker is values in the cuboid,
 * ordered x (dim1) fast, y (dim2) medium, z (dim3) slow.  ker should be
 * allocated for MAX_NSPREAD^3 doubles.
 *
 * Magland Dec 2016. Restrict to sub-cuboids and doc by Barnett 1/16/17.
 * C-style ext alloc, frac sign, removing prefacs from unused dims - faster.
 * 2/15/17-2/17/17
 */
{
  int s1=r1[1]-r1[0]+1, s2, s3;
  if (ndims==1) {
    int aa=0;            // pointer for writing to output ker array
    for (int i=r1[0]; i<=r1[1]; ++i)
      ker[aa++] = evaluate_kernel(x1+(double)i,opts);
    return;
  }
  // now either 2d or 3d. fill kernel evalation 1d lists for each dim...
  double v1[MAX_NSPREAD],v2[MAX_NSPREAD],v3[MAX_NSPREAD];
  for (int i=r1[0]; i<=r1[1]; ++i)
    v1[i-r1[0]] = evaluate_kernel(x1+(double)i,opts);
  s2=r2[1]-r2[0]+1;
  for (int i=r2[0]; i<=r2[1]; ++i)
    v2[i-r2[0]] = evaluate_kernel(x2+(double)i,opts);
  if (ndims>2) {
    s3=r3[1]-r3[0]+1;
    for (int i=r3[0]; i<=r3[1]; ++i)
      v3[i-r3[0]] = evaluate_kernel(x3+(double)i,opts);
  }
  int aa=0;            // pointer for writing to output ker array
  if (ndims==2) {      // compute the rank-2 outer product of two 1d lists...
    for (int j=0; j<s2; j++) {
      double val2=v2[j];
      for (int i=0; i<s1; i++)
	ker[aa++]=val2*v1[i];
    }
  } else {             // compute the rank-3 outer product of three 1d lists...
    for (int k=0; k<s3; k++) {
      double val3=v3[k];
      for (int j=0; j<s2; j++) {
	double val2=val3*v2[j];
	for (int i=0; i<s1; i++)
	  ker[aa++]=val2*v1[i];
      }
    }
  }
}


// ----------------------------------- ES spreading kernel -------------------

int setup_kernel(spread_opts &opts,double eps,double R)
// must be called before evaluate_kernel used.
// returns error code: 0 success, >0 various problems.
{
  int ier=0;   // status
  double fudgefac = 1.0;   // how much actual errors exceed estimated errors
  int ns = std::ceil(-log10(eps/fudgefac))+1;   // 1 digit per power of ten
  opts.nspread = ns;
  opts.ES_halfwidth=(double)ns/2;            // full support, since no 1/4 power
  opts.ES_c = 4.0/(double)(ns*ns);           // avoids recomputing
  double betaoverns = 2.30;                  // approximate betas for R=2.0
  if (ns==2) betaoverns = 2.20;
  if (ns==3) betaoverns = 2.26;
  if (ns==4) betaoverns = 2.38;
  opts.ES_beta = betaoverns * (double)ns;
  if (eps<=1e-16) {
    fprintf(stderr,"setup_kernel: requested eps is too small (<=1e-16)!\n");
    ier=1;
  }
  if (R<1.9 || R>2.1) {
    fprintf(stderr,"setup_kernel: R is not close to 2.0; may be inaccurate!\n");
    ier=2;
  }
  return ier;
}

double evaluate_kernel(double x, const spread_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:

      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2

   which is the asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   beta ~ 2.3 nspread for upsampling factor R=2.0, and has been previously
   chosen by 1d optimization for the R used.

   Barnett 2/16/17. Removed the factor (1-(2x/n_s)^2)^{-1/4},  2/17/17
*/
{
  if (abs(x)>=opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp(opts.ES_beta * sqrt(1.0 - opts.ES_c*x*x));
}
// ------------------------------------- ES kernel done -------------------
