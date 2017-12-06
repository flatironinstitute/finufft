#include "cnufftspread.h"
#include <stdlib.h>
#include <vector>
#include <math.h>

// declarations of internal functions...
void fill_kernel_line(FLT x1, const spread_opts& opts, FLT* ker);
void fill_kernel_square(FLT x1, FLT x2, const spread_opts& opts, FLT* ker);
void fill_kernel_cube(FLT x1, FLT x2, FLT x3, const spread_opts& opts, FLT* ker);
void interp_line(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns);
void interp_square(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT i2,BIGINT N1,BIGINT N2,int ns);
void interp_cube(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT i2,BIGINT i3,BIGINT N1,BIGINT N2,BIGINT N3,int ns);
void spread_subproblem_1d(BIGINT N1,FLT *du0,BIGINT M0,FLT *kx0,FLT *dd0,
			  const spread_opts& opts);
void spread_subproblem_2d(BIGINT N1,BIGINT N2,FLT *du0,BIGINT M0,
			  FLT *kx0,FLT *ky0,FLT *dd0,const spread_opts& opts);
void spread_subproblem_3d(BIGINT N1,BIGINT N2,BIGINT N3,FLT *du0,BIGINT M0,
			  FLT *kx0,FLT *ky0,FLT *kz0,FLT *dd0,
			  const spread_opts& opts);
void bin_sort(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
	      double bin_size_x,double bin_size_y,double bin_size_z);
void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,
		 BIGINT &size2,BIGINT &size3,BIGINT M0,FLT* kx0,FLT* ky0,
		 FLT* kz0,int ns, int ndims);


// subproblem object: is simply a collection of indices of nonuniform points
struct Subproblem {
    std::vector<BIGINT> nonuniform_indices;
};


int cnufftspread(
        BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
        BIGINT M, FLT *kx, FLT *ky, FLT *kz, FLT *data_nonuniform,
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
	sort = (boolean) whether to sort NU points using natural yz-grid
	       ordering. Recommended true.
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
      (see utils.h for error codes)
      3 : one or more non-trivial box dimensions is less than 2.nspread.
      4 : nonuniform points outside range [0,Nm] or [-pi,pi] in at least one
          dimension m=1,2,3.
      6 : invalid opts.spread_direction

   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
   error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17
*/
{
  CNTime timer;
  // Input checking: cuboid not too small for spreading
  int minN = 2*opts.nspread;
  if (N1<minN || (N2>1 && N2<minN) || (N3>1 && N3<minN)) {
    fprintf(stderr,"error: one or more non-trivial box dims is less than 2.nspread!\n");
    return ERR_SPREAD_BOX_SMALL;
  }
  if (opts.spread_direction!=1 && opts.spread_direction!=2) {
    fprintf(stderr,"error: opts.spread_direction must be 1 or 2!\n");
    return ERR_SPREAD_DIR;
  }
  int ndims = 1;                 // decide ndims: 1,2 or 3
  if (N2>1) ++ndims;
  if (N3>1) ++ndims;
  int ns=opts.nspread;          // abbrev. for w, kernel width
  FLT ns2 = (FLT)ns/2;          // half spread width, used as stencil shift
  if (opts.debug)
    printf("starting spread %dD (dir=%d. M=%ld; N1=%ld,N2=%ld,N3=%ld; pir=%d), %d threads\n",ndims,opts.spread_direction,M,N1,N2,N3,opts.pirange,MY_OMP_GET_MAX_THREADS());
  
    if (opts.chkbnds) {  // check NU pts are valid (incl +-1 box), exit gracefully
    timer.start();
    for (BIGINT i=0; i<M; ++i) {
      FLT x=RESCALE(kx[i],N1,opts.pirange);  // this includes +-1 box folding
      if (x<0 || x>N1) {
	fprintf(stderr,"NU pt not in valid range (central three periods): kx=%g, N1=%ld (pirange=%d)\n",x,N1,opts.pirange);
	return ERR_SPREAD_PTS_OUT_RANGE;
      }
    }
    if (ndims>1)
      for (BIGINT i=0; i<M; ++i) {
	FLT y=RESCALE(ky[i],N2,opts.pirange);
	if (y<0 || y>N2) {
	  fprintf(stderr,"NU pt not in valid range (central three periods): ky=%g, N2=%ld (pirange=%d)\n",y,N2,opts.pirange);
	  return ERR_SPREAD_PTS_OUT_RANGE;
	}
      }
    if (ndims>2)
      for (BIGINT i=0; i<M; ++i) {
	FLT z=RESCALE(kz[i],N3,opts.pirange);
	if (z<0 || z>N3) {
	  fprintf(stderr,"NU pt not in valid range (central three periods): kz=%g, N3=%ld (pirange=%d)\n",z,N3,opts.pirange);
	  return ERR_SPREAD_PTS_OUT_RANGE;
	}
      }
    if (opts.debug) printf("NU bnds check: %g s\n",timer.elapsedsec());
  }

  timer.start();                 // if needed, sort all the NU pts...
  BIGINT* sort_indices = (BIGINT*)malloc(sizeof(BIGINT)*M);
  if (opts.sort)
    // store good permutation ordering of all NU pts (dim=1,2 or 3)
    bin_sort(sort_indices,M,kx,ky,kz,N1,N2,N3,opts.pirange,16,4,4);
  else
    for (BIGINT i=0; i<M; i++)                // (omp no speed-up here)
      sort_indices[i]=i;                      // the identity permutation
  if (opts.debug)
    printf("sort time (sort=%d): %.3g s\n",(int)opts.sort,timer.elapsedsec());
  

  if (opts.spread_direction==1) { // ========= direction 1 (spreading) =======

    timer.start();
    BIGINT N=N1*N2*N3;           // output array size, zero it
    for (BIGINT i=0; i<2*N; i++)
        data_uniform[i]=0.0;
    if (M==0)                     // no NU pts
      return 0;
    int nthr = MY_OMP_GET_MAX_THREADS();
    std::vector<Subproblem> s;  // create set of subproblems
    int nb;
    
    int nospatialsplit = 1;  // hardcoded for now
    if (nospatialsplit) {   // split sorted inds (advanced2), could double RAM
      nb = MIN(4*nthr,M);
      if (nb*opts.max_subproblem_size<M)
	nb = (M+opts.max_subproblem_size-1)/opts.max_subproblem_size;
      BIGINT subprobsize=(M+nb-1)/nb;
      s.resize(nb);
      BIGINT offset = 0;
      for (int p=0;p<nb;++p) {   // loop over subprobs to fill
	BIGINT M0 = MIN(subprobsize,M-offset);  // size of this subprob
	for (BIGINT i=0;i<M0;++i) {
	  BIGINT j = sort_indices[i+offset];
	  s.at(p).nonuniform_indices.push_back(j);
	}
	offset += M0;
      }
    } else {       // split by spatial boxes then by sorted inds (advanced4)
      // *** currently too slow (30% of total time!)
      double boxesperthr = 4.0;
      nb = ceil(boxesperthr*nthr);  // rough number of boxes
      int w1=N1,w2=N2,w3=N3;            // set up box sizes in any dim...
      if (ndims==1)
	w1 = MAX(1,ceil(N1/nb));                       // so up to ~nb boxes
      else if (ndims==2) {
	w1=MAX(1,ceil(N1/sqrt(nb))); w2=MAX(1,ceil(N2/sqrt(nb)));  // "
      } else {
	w2=MAX(1,ceil(N2/sqrt(nb))); w3=MAX(1,ceil(N3/sqrt(nb)));  // "
      }
      // # subproblem boxes along each dim
      int nb1=ceil((FLT)N1/w1), nb2=ceil((FLT)N2/w2), nb3=ceil((FLT)N3/w3);
      nb = nb1*nb2*nb3;               // update to actual # boxes
      if (opts.debug) printf("%dx%dx%d subproblem boxes size %dx%dx%d\n",nb1,nb2,nb3,w1,w2,w3);
      
      s.resize(nb);  // create nb subproblems
      for (BIGINT i=0; i<M; i++) {    // build subproblem indices in sorted order
	BIGINT j = sort_indices[i];
	int i1=MIN(nb1-1,floor(RESCALE(kx[j],N1,opts.pirange)/w1)); //make sure legal
	int i2=0, i3=0;
	if (N2>1) i2=MIN(nb2-1,floor(RESCALE(ky[j],N2,opts.pirange)/w2)); // "
	if (N3>1) i3=MIN(nb3-1,floor(RESCALE(kz[j],N3,opts.pirange)/w3)); // "
	int si=i1+nb1*(i2+nb2*i3);      // subproblem index
	// append this source pt's index to the appropriate subproblem's index list
	s.at(si).nonuniform_indices.push_back(j);  // hard to parallelize
      }
      
      // split subproblems by index chunks so none exceed opts.max_subproblem_size
      BIGINT num_nonempty_subproblems=0; // for information only (verbose output)
      for (BIGINT i=0; i<nb; i++) {
	std::vector<BIGINT> inds=s.at(i).nonuniform_indices;
	BIGINT num_nonuniform_points=inds.size();
	if (num_nonuniform_points>opts.max_subproblem_size) {
	  BIGINT next=0;
	  for (BIGINT j=0; j+opts.max_subproblem_size<=num_nonuniform_points; j+=opts.max_subproblem_size) {
	    Subproblem X;   // make new subproblem
	    X.nonuniform_indices=std::vector<BIGINT>(inds.begin()+j,inds.begin()+j+opts.max_subproblem_size);  // extracts contiguous set from vector
	    s.push_back(X);
	    num_nonempty_subproblems++; // for info only
	    next=j+opts.max_subproblem_size;
	  }
	  // the remainder of the indices go to the ith subproblem
	  // it is possible that this will now be an empty subproblem (that's okay)
	  s.at(i).nonuniform_indices=std::vector<BIGINT>(inds.begin()+next,inds.end());
	}
	if (s.at(i).nonuniform_indices.size()>0) {
	  num_nonempty_subproblems++;              // for info only
	}
      }
    }
    nb = s.size();
    if (opts.debug) printf("subprobs setup %.3g s (%d subprobs)\n",timer.elapsedsec(),nb);
    
#pragma omp parallel for schedule(dynamic,1)
    for (int isub=0; isub<nb; isub++) { // Main loop through the subproblems
      std::vector<BIGINT> inds = s.at(isub).nonuniform_indices;
      BIGINT M0 = inds.size();   // # NU pts in this subproblem
      if (M0>0) {              // if some NU pts in this subproblem
	// copy the location and data vectors for the nonuniform points
	FLT* kx0=(FLT*)malloc(sizeof(FLT)*M0), *ky0, *kz0;
	if (N2>1)
	  ky0=(FLT*)malloc(sizeof(FLT)*M0);
	if (N3>1)
	  kz0=(FLT*)malloc(sizeof(FLT)*M0);
	FLT* dd0=(FLT*)malloc(sizeof(FLT)*M0*2);    // complex strength data
	for (BIGINT j=0; j<M0; j++) {   // todo: can avoid this copying ? ***
	  BIGINT kk=inds[j];            // get NU pt from subprob index list
	  kx0[j]=RESCALE(kx[kk],N1,opts.pirange);
	  if (N2>1) ky0[j]=RESCALE(ky[kk],N2,opts.pirange);
	  if (N3>1) kz0[j]=RESCALE(kz[kk],N3,opts.pirange);
	  dd0[j*2]=data_nonuniform[kk*2];     // real part
	  dd0[j*2+1]=data_nonuniform[kk*2+1]; // imag part
	}
	// get the subgrid which will include padding by roughly nspread/2
	BIGINT offset1,offset2,offset3,size1,size2,size3; // get_subgrid sets
	get_subgrid(offset1,offset2,offset3,size1,size2,size3,M0,kx0,ky0,kz0,ns,ndims);
	if (opts.debug>1)  // verbose
	  if (ndims==1)
	    printf("subgrid: off %ld\t siz %ld\t #NU %ld\n",offset1,size1,M0);
	  else if (ndims==2)
	    printf("subgrid: off %ld,%ld\t siz %ld,%ld\t #NU %ld\n",offset1,offset2,size1,size2,M0);
	  else
	    printf("subgrid: off %ld,%ld,%ld\t siz %ld,%ld,%ld\t #NU %ld\n",offset1,offset2,offset3,size1,size2,size3,M0);
	for (BIGINT j=0; j<M0; j++) {
	  kx0[j]-=offset1;  // now kx0 coords are relative to corner of subgrid
	  if (N2>1) ky0[j]-=offset2;  // only accessed if 2D or 3D
	  if (N3>1) kz0[j]-=offset3;  // only access if 3D
	}
	// allocate output data for this subgrid
	FLT* du0=(FLT*)malloc(sizeof(FLT)*2*size1*size2*size3); // complex

	// Spread to subgrid without need for bounds checking or wrapping
	if (!(opts.flags & TF_OMIT_SPREADING))
	  if (ndims==1)
	    spread_subproblem_1d(size1,du0,M0,kx0,dd0,opts);
	  else if (ndims==2)
	    spread_subproblem_2d(size1,size2,du0,M0,kx0,ky0,dd0,opts);
	  else
	    spread_subproblem_3d(size1,size2,size3,du0,M0,kx0,ky0,kz0,dd0,opts);
	
	// Add the subgrid to output grid, wrapping (slower). Works in all dims.
	std::vector<BIGINT> o1(size1), o2(size2), o3(size3);  // alloc 1d output ptr lists
	BIGINT x=offset1, y=offset2, z=offset3;  // fill lists with wrapping...
	for (int i=0; i<size1; ++i) {
	  if (x<0) x+=N1;
	  if (x>=N1) x-=N1;
	  o1[i] = x++;
	}
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
#pragma omp critical
	{  // do the adding of subgrid to output; only here threads cannot clash
	  int p=0;  // pointer into subgrid; this triple loop works in all dims
	  if (!(opts.flags & TF_OMIT_WRITE_TO_GRID))
	    for (int dz=0; dz<size3; dz++) {       // use ptr lists in each axis
	      BIGINT oz = N1*N2*o3[dz];            // offset due to z (0 in <3D)
	      for (int dy=0; dy<size2; dy++) {
		BIGINT oy = oz + N1*o2[dy];        // off due to y & z (0 in 1D)
		for (int dx=0; dx<size1; dx++) {
		  BIGINT j = oy + o1[dx];
		  data_uniform[2*j] += du0[2*p];
		  data_uniform[2*j+1] += du0[2*p+1];
		  ++p;                    // advance input ptr through subgrid
		}
	      }
	    }
	} // end critical block
	// free up stuff from this subprob... (that was malloc'ed by hand)
	free(dd0); free(du0);
	free(kx0);
	if (N2>1) free(ky0);
	if (N3>1) free(kz0); 
      }
    }     // end main loop over subprobs
    
  } else {          // ================= direction 2 (interpolation) ===========

#pragma omp parallel for schedule(dynamic,10000) // (dynamic not needed) assign threads to NU targ pts:
    for (BIGINT i=0; i<M; i++) {   // main loop over NU targs, interp each from U
      BIGINT j=sort_indices[i];    // j current index in input NU targ list
    
      // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
      FLT xj=RESCALE(kx[j],N1,opts.pirange);
      BIGINT i1=(BIGINT)std::ceil(xj-ns2); // leftmost grid index
      FLT x1=(FLT)i1-xj;          // real-valued shifts of ker center
      FLT kernel_values[MAX_NSPREAD*MAX_NSPREAD*MAX_NSPREAD]; // static, up to 3D
      
      // eval kernel values patch and use to interpolate from uniform data...
      if (!(opts.flags & TF_OMIT_SPREADING))
	if (ndims==1) {                                          // 1D
	  fill_kernel_line(x1,opts,kernel_values);
	  interp_line(&data_nonuniform[2*j],data_uniform,kernel_values,i1,N1,ns);
	} else if (ndims==2) {                                   // 2D
	  FLT yj=RESCALE(ky[j],N2,opts.pirange);
	  BIGINT i2=(BIGINT)std::ceil(yj-ns2); // min y grid index
	  FLT x2=(FLT)i2-yj;
	  fill_kernel_square(x1,x2,opts,kernel_values);
	  interp_square(&data_nonuniform[2*j],data_uniform,kernel_values,i1,i2,N1,N2,ns);
	} else {                                                 // 3D
	  FLT yj=RESCALE(ky[j],N2,opts.pirange);
	  FLT zj=RESCALE(kz[j],N3,opts.pirange);
	  BIGINT i2=(BIGINT)std::ceil(yj-ns2); // min y grid index
	  BIGINT i3=(BIGINT)std::ceil(zj-ns2); // min z grid index
	  FLT x2=(FLT)i2-yj;
	  FLT x3=(FLT)i3-zj;
	  fill_kernel_cube(x1,x2,x3,opts,kernel_values);
	  interp_cube(&data_nonuniform[2*j],data_uniform,kernel_values,i1,i2,i3,N1,N2,N3,ns);
	}
    }    // end NU targ loop
  }                           // ================= end direction choice ========
  
  free(sort_indices);
  return 0;
}

///////////////////////////////////////////////////////////////////////////


int setup_spreader(spread_opts &opts,FLT eps,FLT R)
// Initializes spreader kernel parameters, including all options in spread_opts.
// See cnufftspread.h for definitions.
// Must be called before evaluate_kernel is used.
// Returns: 0 success, >0 failure (see error codes in utils.h)
{
  opts.spread_direction = 1;    // user should always set to 1 or 2 as desired
  opts.pirange = 1;             // user also should always set
  opts.chkbnds = 1;
  opts.sort = 1;
  opts.max_subproblem_size = (BIGINT)1e5;
  opts.flags = 0;
  opts.debug = 0;
  
  // Set the kernel width w (nspread) and parameters, using eps and R...
  FLT fudgefac = 1.0;   // how much actual errors exceed estimated errors
  int ns = std::ceil(-log10(eps/fudgefac))+1;   // 1 digit per power of ten
  ns = max(2,ns);                            // we don't have ns=1 version yet
  ns = min(ns,MAX_NSPREAD);                  // clip for safety!
  opts.nspread = ns;
  opts.ES_halfwidth=(FLT)ns/2;            // full support, since no 1/4 power
  opts.ES_c = 4.0/(FLT)(ns*ns);           // avoids recomputing
  FLT betaoverns = 2.30;                  // approximate betas for R=2.0
  if (ns==2) betaoverns = 2.20;
  if (ns==3) betaoverns = 2.26;
  if (ns==4) betaoverns = 2.38;
  opts.ES_beta = betaoverns * (FLT)ns;
  if (eps<0.5*EPSILON) {       // arbitrary, but fortran wants 1e-16 to be ok
    fprintf(stderr,"setup_kernel: requested eps is too small (<%.3g)!\n",0.5*EPSILON);
    return ERR_EPS_TOO_SMALL;
  }
  if (R<1.9 || R>2.1)
    fprintf(stderr,"setup_kernel: warning R=%.3g is not close to 2.0; may be inaccurate!\n",(double)R);
  
  return 0;
}

FLT evaluate_kernel(FLT x, const spread_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:

      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2

   which is an asymptotic approximation to the Kaiser--Bessel, itself an
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

FLT evaluate_kernel_noexp(FLT x, const spread_opts &opts)
// Version of the above just for timing purposes!!! Gives wrong answer
{
  if (abs(x)>=opts.ES_halfwidth)
    return 0.0;
  else {
    FLT s = sqrt(1.0 - opts.ES_c*x*x);
    //  return sinh(opts.ES_beta * s)/s; // roughly, inverse K-B kernel of NFFT
        return opts.ES_beta * s;
  }
}

void fill_kernel_line(FLT x1, const spread_opts& opts, FLT* ker)
// Fill ker with kernel values evaluated at x1+[0:ns] in 1D.
{
  int ns=opts.nspread;
  if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL))
    if (!(opts.flags & TF_OMIT_EVALUATE_EXPONENTIAL))
      for (int i = 0; i <= ns; i++)
	ker[i] = evaluate_kernel(x1 + (FLT)i, opts);
    else
      for (int i = 0; i <= ns; i++)
	ker[i] = evaluate_kernel_noexp(x1 + (FLT)i, opts);
  else
    for (int i = 0; i <= ns; i++)
      ker[i] = 1.0;        // dummy
}

void fill_kernel_square(FLT x1, FLT x2, const spread_opts& opts, FLT* ker)
// Fill ker with tensor product of kernel values evaluated at xm+[0:ns] in dims
// m=1,2.
{
  int ns=opts.nspread;
  FLT v1[MAX_NSPREAD], v2[MAX_NSPREAD];
  if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL))
    if (!(opts.flags & TF_OMIT_EVALUATE_EXPONENTIAL))
      for (int i = 0; i < ns; i++) {
	v1[i] = evaluate_kernel(x1 + (FLT)i, opts);
	v2[i] = evaluate_kernel(x2 + (FLT)i, opts);
      }
    else
      for (int i = 0; i < ns; i++) {
	v1[i] = evaluate_kernel_noexp(x1 + (FLT)i, opts);
	v2[i] = evaluate_kernel_noexp(x2 + (FLT)i, opts);
      }
  else
    for (int i = 0; i < ns; i++) { v1[i] = 1.0; v2[i] = 1.0; }  // dummy
  int aa = 0; // pointer for writing to output ker array
  for (int j = 0; j < ns; j++) {
    FLT val2 = v2[j];
    for (int i = 0; i < ns; i++)
      ker[aa++] = val2 * v1[i];
  }
}

void fill_kernel_cube(FLT x1, FLT x2, FLT x3, const spread_opts& opts,FLT* ker)
// Fill ker with tensor product of kernel values evaluated at xm+[0:ns] in dims
// m=1,2,3.
{
    int ns=opts.nspread;
    FLT v1[MAX_NSPREAD], v2[MAX_NSPREAD], v3[MAX_NSPREAD];
    if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL))
      if (!(opts.flags & TF_OMIT_EVALUATE_EXPONENTIAL))
	for (int i = 0; i < ns; i++) {
	  v1[i] = evaluate_kernel(x1 + (FLT)i, opts);
	  v2[i] = evaluate_kernel(x2 + (FLT)i, opts);
	  v3[i] = evaluate_kernel(x3 + (FLT)i, opts);
	}
      else
	for (int i = 0; i < ns; i++) {
	  v1[i] = evaluate_kernel_noexp(x1 + (FLT)i, opts);
	  v2[i] = evaluate_kernel_noexp(x2 + (FLT)i, opts);
	  v3[i] = evaluate_kernel_noexp(x3 + (FLT)i, opts);
	}
    else
      for (int i=0; i<ns; i++) { v1[i]=1.0; v2[i]=1.0; v3[i]=1.0; }  // dummy
    int aa = 0; // pointer for writing to output ker array
    for (int k = 0; k < ns; k++) {
        FLT val3 = v3[k];
        for (int j = 0; j < ns; j++) {
            FLT val2 = val3 * v2[j];
            for (int i = 0; i < ns; i++)
                ker[aa++] = val2 * v1[i];
        }
    }
}

void interp_line(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns)
// 1D interpolate complex values from du array to out, using real weights
// ker[0] through ker[ns-1]. out must be size 2 (real,imag), and du
// of size 2*N1 (alternating real,imag). i1 is the left-most index in [0,N1)
// Periodic wrapping in the du array is applied, assuming N1>=ns.
// dx is index into ker array, j index in complex du (data_uniform) array.
// Barnett 6/15/17
{
  out[0] = 0.0; out[1] = 0.0;
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
}

void interp_square(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT i2,BIGINT N1,BIGINT N2,int ns)
// 2D interpolate complex values from du (uniform grid data) array to out value,
// using ns*ns square of real weights
// in ker. out must be size 2 (real,imag), and du
// of size 2*N1*N2 (alternating real,imag). i1 is the left-most index in [0,N1)
// and i2 the bottom index in [0,N2).
// Periodic wrapping in the du array is applied, assuming N1,N2>=ns.
// dx,dy indices into ker array, j index in complex du array.
// Barnett 6/16/17
{
  out[0] = 0.0; out[1] = 0.0;
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2) {  // no wrapping: avoid ptrs
    int p=0;  // pointer into ker array
    for (int dy=0; dy<ns; dy++) {
      BIGINT j = N1*(i2+dy) + i1;
      for (int dx=0; dx<ns; dx++) {
	FLT k = ker[p++];             // advance the pointer through ker
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
    int p=0;  // pointer into ker array
    for (int dy=0; dy<ns; dy++) {      // use the pts lists
      BIGINT oy = N1*j2[dy];           // offset due to y
      for (int dx=0; dx<ns; dx++) {
	FLT k = ker[p++];              // advance the pointer through ker
	BIGINT j = oy + j1[dx];
	out[0] += du[2*j] * k;
	out[1] += du[2*j+1] * k;
      }
    }
  }
}

void interp_cube(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT i2,BIGINT i3,
		 BIGINT N1,BIGINT N2,BIGINT N3,int ns)
// 3D interpolate complex values from du (uniform grid data) array to out value,
// using ns*ns*ns cube of real weights
// in ker. out must be size 2 (real,imag), and du
// of size 2*N1*N2*N3 (alternating real,imag). i1 is the left-most index in
// [0,N1), i2 the bottom index in [0,N2), i3 lowest in [0,N3).
// Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
// dx,dy,dz indices into ker array, j index in complex du array.
// Barnett 6/16/17
{
  out[0] = 0.0; out[1] = 0.0;
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2 && i3>=0 && i3+ns<=N3) {
    // no wrapping: avoid ptrs
    int p=0;  // pointer into ker array
    for (int dz=0; dz<ns; dz++) {
      BIGINT oz = N1*N2*(i3+dz);        // offset due to z
      for (int dy=0; dy<ns; dy++) {
	BIGINT j = oz + N1*(i2+dy) + i1;
	for (int dx=0; dx<ns; dx++) {
	  FLT k = ker[p++];             // advance the pointer through ker
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
    int p=0;  // pointer into ker array
    for (int dz=0; dz<ns; dz++) {             // use the pts lists
      BIGINT oz = N1*N2*j3[dz];               // offset due to z
      for (int dy=0; dy<ns; dy++) {
	BIGINT oy = oz + N1*j2[dy];           // offset due to y & z
	for (int dx=0; dx<ns; dx++) {
	  FLT k = ker[p++];                   // advance the pointer through ker
	  BIGINT j = oy + j1[dx];
	  out[0] += du[2*j] * k;
	  out[1] += du[2*j+1] * k;
	}
      }
    }
  }
}

void spread_subproblem_1d(BIGINT N1,FLT *du,BIGINT M,
			  FLT *kx,FLT *dd,
			  const spread_opts& opts)
/* spreader from dd (NU) to du (uniform) in 1D without wrapping.
   kx (size M) are NU locations in [0,N1]
   dd (size M complex) are source strengths
   du (size N1) is uniform output array
 */
{
  int ns=opts.nspread;
  FLT ns2 = (FLT)ns/2;          // half spread width
  for (BIGINT i=0;i<2*N1;++i)
    du[i] = 0.0;
  FLT ker[MAX_NSPREAD];
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];
    fill_kernel_line(x1,opts,ker);
    // critical inner loop:
    int j=i1;
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
  FLT ker[MAX_NSPREAD*MAX_NSPREAD];
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];
    FLT x2 = (FLT)i2 - ky[i];
    fill_kernel_square(x1,x2,opts,ker);
    // critical inner loop:
    int p=0;              // ptr to ker array
    for (int dy=0; dy<ns; ++dy) {
      int j = N1*(i2+dy) + i1;
      for (int dx=0; dx<ns; ++dx) {
	FLT k = ker[p++];            // increment ker array ptr
	du[2*j] += re0*k;
	du[2*j+1] += im0*k;
	++j;
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
  FLT ker[MAX_NSPREAD*MAX_NSPREAD*MAX_NSPREAD];
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
    BIGINT i3 = (BIGINT)std::ceil(kz[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];
    FLT x2 = (FLT)i2 - ky[i];
    FLT x3 = (FLT)i3 - kz[i];
    fill_kernel_cube(x1,x2,x3,opts,ker);
    // critical inner loop:
    int p=0;              // ptr to ker array
    for (int dz=0; dz<ns; ++dz) {
      BIGINT oz = N1*N2*(i3+dz);        // offset due to z
      for (int dy=0; dy<ns; ++dy) {
	BIGINT j = oz + N1*(i2+dy) + i1;
	for (int dx=0; dx<ns; ++dx) {
	  FLT k = ker[p++];            // increment ker array ptr
	  du[2*j] += re0*k;
	  du[2*j+1] += im0*k;
	  ++j;
	}
      }
    }
  }
}

void bin_sort(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
	      double bin_size_x,double bin_size_y,double bin_size_z)
/* Returns permutation of all nonuniform points with good RAM access,
 * ie less cache misses for spreading, in 1D, 2D, or 3D.
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
 * Avoided the bins array, as in JFM's spreader of 2016.
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
  offsets[0]=0;
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
  // invert the map, writing to output pointer
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
