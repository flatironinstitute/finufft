#include "cnufftspread.h"
#include "../contrib/besseli.h"

#include <stdlib.h>
#include <vector>
#include <math.h>
#include <sys/time.h>

// allow compile-time switch off of openmp (_OPENMP is set by -fopenmp compile flag)
#ifdef _OPENMP
  #include <omp.h>
#else
//  hack to handle all the omp commands we use...
  #define omp_get_num_threads() 1
  #define omp_get_thread_num() 0
  #define omp_set_num_threads()
#endif

// declarations of later functions...
std::vector<long> compute_sort_indices(long M,double *kx, double *ky, double *kz,
				       long N1,long N2,long N3);
double evaluate_kernel(double x,const cnufftspread_opts &opts);
std::vector<double> compute_kernel_values(double frac1,double frac2,double frac3,
					  const cnufftspread_opts &opts, int *r1,
					  int *r2, int *r3);
bool set_thread_index_box(long *i1th,long *i2th,long *i3th,long N1,long N2,long N3,
			  int th,int nth, const cnufftspread_opts &opts);
bool ind_might_affect_interval(long i,long N,long *ith,long nspread);
bool wrapped_range_in_interval(long i,int *R,long *ith,long N,int *r);


int cnufftspread(
        long N1, long N2, long N3, double *data_uniform,
        long M, double *kx, double *ky, double *kz, double *data_nonuniform,
        cnufftspread_opts opts)
/* Main code for spreader for 1, 2, or 3 dimensions. No particular normalization of the
   spreading kernel is assumed, since this is cancelling in the NUFFT anyway.
   Uniform (U) points are centered at coords [0,1,...,N1-1] in 1D, analogously in 2D and
   3D. They are stored in x fastest, y medium, z slowest ordering, up to however many
   dimensions are relevant.
   Non-uniform (NU) points are real and must be in the range [0,N1] in 1D,
   analogously in 2D and 3D, otherwise an error is returned and no calculation is done.

   Inputs:
   N1,N2,N3 - grid sizes in x (fastest), y (medium), z (slowest) respectively.
              If N2==0, 1D spreading is done. If N3==0, 2D spreading. Otherwise, 3D.
   M - number of NU pts.
   kx, ky, kz - length-M real arrays of NU point coordinates (only kz used in 1D,
                only kx and ky used in 2D). These must lie in the box, ie 0<=kx<=N1 etc.
   opts - struct controlling spreading method and text output, has fields including:
        spread_direction=1, spreads from nonuniform input to uniform output, or
        spread_direction=2, interpolates ("spread transpose") from uniform input
                            to nonuniform output.
	sort_data - (boolean) whether to sort NU points using natural yz-grid ordering.
	debug = 0: no text output, 1: some openmp output, 2: mega output (each NU pt)
        checkerboard = 0: for dir=1, split top dimension only, 1: checkerboard top two.

   Inputs/Outputs:
   data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
   data_nonuniform - input strengths of the sources (dir=1)
                     OR output values at targets (dir=2)
   Ouputs:
   returned value - error status indicator:
      0 : ok
      1 : one or more non-trivial box dimensions is less than 2.nspread
      2 : nonuniform points outside range [0,Nm] in at least one dimension m=1,2,3.
      3 : out of memory

   Notes:
   1) it is assumed that 2*opts.nspread < min(N1,N2,N2), so that the kernel only ever
   wraps once when falls below 0 or off the top of a uniform grid dimension.
   2) decided not to periodically fold input data is better, since use type 3!)

   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17.
*/
{ 
  // Input checking: cuboid not too small for spreading
  long minN = 2*opts.nspread;
  if (N1<minN || (N2>1 && N2<minN) || (N3>1 && N3<minN)) {
    printf("error: one or more non-trivial box dimensions is less than 2.nspread!\n");
    return 1;
  }

  std::vector<double> kx2,ky2,kz2,data_nonuniform2;  // declarations can't be in try block
  std::vector<long> sort_indices;
  try {    // alloc the big workspaces in a graceful way
    kx2.resize(M); ky2.resize(M); kz2.resize(M); data_nonuniform2.resize(M*2);
    sort_indices.resize(M);
  }
  catch(std::bad_alloc &e) {
    return 3;
  }
  // omp_set_num_threads(1); // for debug; also can set via environment var
  
  // Sort NU pts once and for all: sorted answer will be k{xyz}2 and data_nonuniform2
  // We also zero unused coordinates (for 1D or 2D cases) and check bounds:
  CNTime timer; timer.start();
  if (opts.sort_data)
    sort_indices=compute_sort_indices(M,kx,ky,kz,N1,N2,N3);  // a good perm of NU pts
  else {
    for (long i=0; i<M; i++)        // (omp no effect here)
      sort_indices[i]=i;                                  // the identity permutation!
  }
  bool bnderr = false;
  #pragma omp parallel for schedule(dynamic)
  for (long i=0; i<M; i++) {        // (omp has 20% effect on dir=1 case, so use it)
    long jj=sort_indices[i];
    kx2[i]=kx[jj];
    if (N2==1)            // safely kill not-needed coords
      ky2[i] = 0.0;
    else
      ky2[i]=ky[jj];
    if (N3==1)	
      kz2[i] = 0.0;
    else
      kz2[i]=kz[jj];
    // while we're here, check bounds of NU pts (after unused coords were killed)...
    if (kx2[i]<0.0 || kx2[i]>N1 || ky2[i]<0.0 || ky2[i]>N2 || kz2[i]<0.0 || kz2[i]>N3)
      bnderr = true;
    if (opts.spread_direction==1) {    // note this also sorts incoming strengths
      data_nonuniform2[i*2]=data_nonuniform[jj*2];      // real
      data_nonuniform2[i*2+1]=data_nonuniform[jj*2+1];  // imag
    }
  }
  double t=timer.elapsedsec();
  if (opts.debug) printf("sort time (sort_data=%d): %.3g s\n",(int)opts.sort_data,t);
  if (bnderr) {
    printf("error: at least one nonuniform point not in range [0,N1] x ... !\n");
    return 2;
  }

  // set up spreading kernel index bounds relative to its center, in each dim:
  int R=opts.nspread;       // shorthand for max box size
  int R1[2]={-R/2,R/2-1};                      // we always spread in x
  int R2[2]={0,0}; int R3[2]={0,0};
  if (N2>1) { R2[0] = -R/2; R2[1] = R/2-1; }   // also spread in y
  if (N3>1) { R3[0] = -R/2; R3[1] = R/2-1; }   // also spread in z
  if (opts.debug) printf("R box: %d %d %d %d %d %d\n",R1[0],R1[1],R2[0],R2[1],R3[0],R3[1]);
  
#pragma omp parallel
  {  // omp block : release a cadre of threads
    int nth = omp_get_num_threads(), th = omp_get_thread_num();
    if (th==0 && opts.debug) printf("spreading dir=%d, %d threads\n", opts.spread_direction,nth);
    
    if (opts.spread_direction==1) { // ==================== direction 1 ==============
      // zero the complex output array ready to accumulate...
#pragma omp for schedule(dynamic)
      for (long i=0; i<2*N1*N2*N3; i++) data_uniform[i]=0.0;
      
      long i1th[2],i2th[2],i3th[2];     // get this thread's (fixed) grid index box...
      if (set_thread_index_box(i1th,i2th,i3th,N1,N2,N3,th,nth,opts)) {  // thread has task
	if (opts.debug) printf("th=%d ind box: %ld %ld %ld %ld %ld %ld\n",th,i1th[0],i1th[1],i2th[0],i2th[1],i3th[0],i3th[1]);
	long c = 0;   // debug count how many NU pts each thread does
	
	for (long i=0; i<M; i++) {  // main loop over NU pts, spread each to U grid
	  // (note every thread does this loop, but only some cases write to the grid)
	  long i1=(long)((kx2[i]+0.5)); // rounds to nearest grid pt in real space
	  long i2=(long)((ky2[i]+0.5)); // safely is 0 if is a coord in an unused dim
	  long i3=(long)((kz2[i]+0.5)); // "
	  int r1[2],r2[2],r3[2]; // lower & upper rel ind bnds restricted to thread's box
	  bool i_affects_box = wrapped_range_in_interval(i1,R1,i1th,N1,r1) &&
	    wrapped_range_in_interval(i2,R2,i2th,N2,r2) &&
	    wrapped_range_in_interval(i3,R3,i3th,N3,r3); // also set the r1,r2,r3 bounds

	  if (i_affects_box) {
	    if (opts.debug>1) printf("th=%d r box: %d %d %d %d %d %d\n",th,r1[0],r1[1],r2[0],r2[1],r3[0],r3[1]);
	    ++c;     // debug
	    double frac1=kx2[i]-i1;
	    double frac2=ky2[i]-i2;       // "
	    double frac3=kz2[i]-i3;       // "
	    // from now on in dir=1, we use "small r" instead of "big R" bounds...
	    // Eval only ker vals needed for overall dim and this thread's index box
	    // (the set of relative indices always fall into a single box)
	    std::vector<double> kernel_values=compute_kernel_values(frac1,frac2,frac3,opts,r1,r2,r3);
	    // set up indices for each dim ahead of time using by-hand modulo wrapping
	    // periodically up to +-1 period:
	    long j1_array[r1[1]-r1[0]+1],j2_array[r2[1]-r2[0]+1],j3_array[r3[1]-r3[0]+1];
	    for (int dx=r1[0]; dx<=r1[1]; dx++) {
	      long j=i1+dx; if (j<0) j+=N1; if (j>=N1) j-=N1; j1_array[dx-r1[0]]=j; }
	    for (int dy=r2[0]; dy<=r2[1]; dy++) {
	      long j=i2+dy; if (j<0) j+=N2; if (j>=N2) j-=N2; j2_array[dy-r2[0]]=j; }
	    for (int dz=r3[0]; dz<=r3[1]; dz++) {
	      long j=i3+dz; if (j<0) j+=N3; if (j>=N3) j-=N3; j3_array[dz-r3[0]]=j; }
	    double re0=data_nonuniform2[i*2];
	    double im0=data_nonuniform2[i*2+1];
  	    long aa = 0;
	    for (int dz=r3[0]; dz<=r3[1]; dz++) {
	      long o3=N1*N2*j3_array[dz-r3[0]];  // use precomputed index lists in each dim
	      for (int dy=r2[0]; dy<=r2[1]; dy++) {
		long o2=o3 + N1*j2_array[dy-r2[0]];
		for (int dx=r1[0]; dx<=r1[1]; dx++) {
		  long jjj=o2 + j1_array[dx-r1[0]];
		  double kern0=kernel_values[aa];     // kernel vals swept in proper order
		  data_uniform[jjj*2]   += re0*kern0; // accumulate complex value to grid
		  data_uniform[jjj*2+1] += im0*kern0;
		  aa++;
		}
	      }
	    }
	  }
	}
	// printf("th=%d did %ld NU pts.\n",th,c); // debug
      }
    } else {                      // ==================== direction 2 ===============
#pragma omp for schedule(dynamic)   // assign threads to NU targ pts, easy
      for (long i=0; i<M; i++) {  // main loop over NU pts targets, interp each from U
	long i1=(long)((kx2[i]+0.5)); // rounds to nearest grid pt in real space
	long i2=(long)((ky2[i]+0.5)); // safely is 0 if is a coord in an unused dim
	long i3=(long)((kz2[i]+0.5)); // "
	double frac1=kx2[i]-i1;
	double frac2=ky2[i]-i2;         // "
	double frac3=kz2[i]-i3;         // "
	std::vector<double> kernel_values=compute_kernel_values(frac1,frac2,frac3,opts,R1,R2,R3);
	// set up indices for each dim ahead of time using by-hand modulo wrapping
	// periodically up to +-1 period:
	long j1_array[R1[1]-R1[0]+1],j2_array[R2[1]-R2[0]+1],j3_array[R3[1]-R3[0]+1];	
	for (int dx=R1[0]; dx<=R1[1]; dx++) {
	  long j=i1+dx; if (j<0) j+=N1; if (j>=N1) j-=N1; j1_array[dx-R1[0]]=j; }
	for (int dy=R2[0]; dy<=R2[1]; dy++) {
	  long j=i2+dy; if (j<0) j+=N2; if (j>=N2) j-=N2; j2_array[dy-R2[0]]=j; }
	for (int dz=R3[0]; dz<=R3[1]; dz++) {
	  long j=i3+dz; if (j<0) j+=N3; if (j>=N3) j-=N3; j3_array[dz-R3[0]]=j; }
	double re0=0.0, im0=0.0;
	long aa = 0;
	for (int dz=R3[0]; dz<=R3[1]; dz++) {
	  long o3=N1*N2*j3_array[dz-R3[0]];  // use precomputed index lists in each dim
	  for (int dy=R2[0]; dy<=R2[1]; dy++) {
	    long o2=o3 + N1*j2_array[dy-R2[0]];
	    for (int dx=R1[0]; dx<=R1[1]; dx++) {
	      long jjj=o2 + j1_array[dx-R1[0]];
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
      for (long i=0; i<M; i++) {
	long jj=sort_indices[i];
	data_nonuniform[jj*2]=data_nonuniform2[i*2];
	data_nonuniform[jj*2+1]=data_nonuniform2[i*2+1];
      }
    }
  } // omp block
  return 0;
}

std::vector<double> compute_kernel_values(double frac1,double frac2,double frac3,
					  const cnufftspread_opts &opts, int *r1,
					  int *r2, int *r3)
/* Evaluate spreading kernel values on a cuboid of grid points shifted from the origin
 * by fractional part frac1,frac2,frac2. This may be a sub-cuboid of the full
 * possible R^3 values, and is controlled by r1,r2,r3 which are two-element arrays
 * giving start and end indices offsets (in -R/2,...,R/2-1) in each dimension.
 * If a dimension m is unused, fracm should be 0.0, and both rm
 * elements should be set to 0. The kernel gets a factor of the 1d kernel at x=0 for
 * each such dim. For speed, the output is just the values in the
 * sub-cuboid, ordered x (dim1) fast, y (dim2) medium, z (dim3) slow.
 * (A full R^3 vector STL would waste time initializing with R^3 zeros.)
 * R must be even.
 *
 * Magland Dec 2016. Restrict to sub-cuboids and doc by Barnett 1/16/17.
 */
{
  int R=opts.nspread;       // all ints, used for indexing speed
  int sh=-R/2;              // index shift
  int s1=r1[1]-r1[0]+1, s2=r2[1]-r2[0]+1, s3=r3[1]-r3[0]+1; // cuboid sizes
  std::vector<double> v1(s1),v2(s2),v3(s3);  // fill 1d lists in each dim...
  for (int i=r1[0]; i<=r1[1]; ++i)
    v1[i-r1[0]] = evaluate_kernel(frac1-(double)i,opts);
  for (int i=r2[0]; i<=r2[1]; ++i)
    v2[i-r2[0]] = evaluate_kernel(frac2-(double)i,opts);
  for (int i=r3[0]; i<=r3[1]; ++i)
    v3[i-r3[0]] = evaluate_kernel(frac3-(double)i,opts);
  // now simply compute the rank-3 outer product of these 1d lists...
  std::vector<double> ret(s1*s2*s3);
  int aa=0;                     // output pointer
  for (int k=0; k<s3; k++) {
    double val3=v3[k];
    for (int j=0; j<s2; j++) {
      double val2=val3*v2[j];
      for (int i=0; i<s1; i++) {
	double val1=val2*v1[i];
	ret[aa]=val1;
	aa++;
      }
    }
  }
  return ret;
}

double evaluate_kernel(double x,const cnufftspread_opts &opts) {
  double t = 2.0*x/opts.KB_W; 
  double tmp1=1.0-t*t;
  if (tmp1<0.0) {
    return 0.0;
  }
  else {
    double y = opts.KB_beta*sqrt(tmp1);
    //return besseli0(y);
    return besseli0_approx(y);       // todo: compare acc
  }
}

bool set_thread_index_box(long *i1th,long *i2th,long *i3th,long N1,long N2,long N3,
			  int th,int nth, const cnufftspread_opts &opts)
/* Decides how the uniform grid is to be partitioned into cuboids for each thread
 * (for spread_direction=1 only, ie, writing to the grid).
 *
 * Inputs: N1,N2,N3 dimensions of uniform grid (N2=N3=1 for 1d, N3=1 for 2d, otherwise 3d)
 *         nth - number of total threads which must cover the grid.
 *         th - number of the thread we're assigning a cuboid. Must have 0 <= th < nth.
 *         opts - spreading opts structure, only opts.checkerboard used:
 *                 0: slice only top dimension, 1: checkerboard in 2d & 3d (todo)
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
  long ith[2], Ntop = N1, dims=1;
  if (N2>1) { Ntop = N2; dims=2; }
  if (N3>1) { Ntop = N3; dims=3; }
  if (N2==1 || !opts.checkerboard) {  // slice only along one dim
    if (Ntop<nth) {    // we're at a loss to occupy every thread; assign one per grid pt
      ith[0] = ith[1] = th;
      return th<Ntop;
    } else {           // this relies on consistent rounding behavior every time called!
      ith[0] = (long)(th*(double)Ntop/nth);
      ith[1] = (long)((th+1)*(double)Ntop/nth - 1);
    }
    // now slice only along the top dim (we keep lines or planes in lower dims)
    if (dims==1) {
      i1th[0] = ith[0]; i1th[1] = ith[1];
    } else if (dims==2) {
      i2th[0] = ith[0]; i2th[1] = ith[1];
    } else if (dims==3) {
      i3th[0] = ith[0]; i3th[1] = ith[1];
    }
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
   * The last assumption ensures intersection is a single interval.
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
  /* Returns permutation of the 3D nonuniform points with optimal RAM access for the
   * upcoming spreading step.
   *
   * "Optimal" means lots of requested blocks of RAM can stay in cache to be reused.
   * Currenty this is achieved by binning into 1-grid-point sized boxes in the yz-plane,
   * with no sorting along x within each box, then reading out the indices within these
   * boxes in the natural box order (y fast, z slow).
   * Finally the permutation map is inverted.
   * 
   * Inputs: M - length of inputs
   *         kx,ky,kz - length-M real numbers in 
   * Output: vector list of indices, each in the range 0,..,M-1, which is a good ordering
   *         of the points.
   *
   * Note: apparently in 2D sorts only along y, and in 1D doesn't sort at all (x).
   * todo: fix the 1d case.
   */
{
  std::vector<long> counts(N2*N3);
  for (long j=0; j<N2*N3; j++)
    counts[j]=0;
  for (long i=0; i<M; i++) {
    long i2=(long)(ky[i]+0.5);
    if (i2<0) i2=0;
    if (i2>=N2) i2=N2-1;
    
    long i3=(long)(kz[i]+0.5);
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
    long i2=(long)(ky[i]+0.5);
    if (i2<0) i2=0;
    if (i2>=N2) i2=N2-1;
    
    long i3=(long)(kz[i]+0.5);
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

void cnufftspread_opts::set_W_and_beta() {  // set derived parameters in Kaiser--Bessel
  this->KB_W = this->nspread * this->KB_fac1;
  double tmp0 = this->KB_W * this->KB_W / 4 - 0.8;
  if (tmp0<0) tmp0=0;   // fix it?
  this->KB_beta = M_PI*sqrt(tmp0) * this->KB_fac2;
}

void set_kb_opts_from_kernel_params(cnufftspread_opts &opts,double *kernel_params) {
/* Directly sets Kaiser-Bessel spreading options.
 *
 * kernel_params is a 4-element double array containing following information:
 *  entry 0: kernel type (1 for kaiser-bessel) - ignored
 *  entry 1: nspread
 *  entry 2: KB_fac1 (eg 1)
 *  entry 3: KB_fac2 (eg 1)
 */
  opts.nspread=kernel_params[1];
  opts.KB_fac1=kernel_params[2];
  opts.KB_fac2=kernel_params[3];
  opts.set_W_and_beta();  
}

void set_kb_opts_from_eps(cnufftspread_opts &opts,double eps)
// Sets KB spreading opts from accuracy eps. It seems from other uses that
// nspread should always be even.
{
  int nspread=12; double fac1=1,fac2=1;  // defaults: todo decide for what tol
  // tests done sequentially to categorize eps...
  if (eps>=1e-2) {
    nspread=4; fac1=0.75; fac2=1.71;
  }
  else if (eps>=1e-4) {
    nspread=6; fac1=0.83; fac2=1.56;
  }
  else if (eps>=1e-6) {
    nspread=8; fac1=0.89; fac2=1.45;
  }
  else if (eps>=1e-8) {
    nspread=10; fac1=0.90; fac2=1.47;
  }
  else if (eps>=1e-10) {
    nspread=12; fac1=0.92; fac2=1.51;
  }
  else if (eps>=1e-12) {
    nspread=14; fac1=0.94; fac2=1.48;
  }
  else {       // eps < 1e-12
    nspread=16; fac1=0.94; fac2=1.46;
  }
  opts.nspread=nspread;
  opts.KB_fac1=fac1;
  opts.KB_fac2=fac2;
  opts.set_W_and_beta();
}

void cnufftspread_type1(int N,double *Y,int M,double *kx,double *ky,double *kz,double *X,double *kernel_params)
// wrapper for matlab access - move this and its .h to matlab/
{
  cnufftspread_opts opts;
  set_kb_opts_from_kernel_params(opts,kernel_params);
  opts.spread_direction=1;
  int ier;
  
  ier = cnufftspread(N,N,N,Y,M,kx,ky,kz,X,opts);
  //    todo: return ier; somehow
}

void evaluate_kernel(int len, double *x, double *values, cnufftspread_opts opts)
{
  for (int i=0; i<len; i++) {
    values[i]=evaluate_kernel(x[i],opts);
  }
}

// ----------------------- helpers for timing...
using namespace std;

void CNTime::start()
{
  gettimeofday(&initial, 0);
}

int CNTime::restart()
{
  int delta = this->elapsed();
  this->start();
  return delta;
}

int CNTime::elapsed()
//  returns answers as integer number of milliseconds
{
  struct timeval now;
  gettimeofday(&now, 0);
  int delta = 1000 * (now.tv_sec - (initial.tv_sec + 1));
  delta += (now.tv_usec + (1000000 - initial.tv_usec)) / 1000;
  return delta;
}

double CNTime::elapsedsec()
//  returns answers as double in sec
{
  return (double)(this->elapsed()/1e3);
}
