/* TWOPISPREAD:
 * Wrappers to the C++ spreading library cnufftspread, that handle d-dim
 * nonuniform (NU) points in [-pi,pi]^d, with separate calls for each d=1,2,3.
 * They call the spreader after rescaling these points to
 * [0,N1] x ... x [0,Nd], which is the domain the spreader wants.
 * Any unused coords of the points are also filled with zeros.
 * Either direction (dir=1,2) is handled in the same call, so that either
 * cj is input and fw output, or vice versa (see cnufftspread docs).
 * Cost of this routine is a copy to workspace of nj*d doubles.
 * As with cnufftspread, opts must have already been set by setup_kernel.
 *
 * For each of these routines:
 *
 * Inputs:
 *  nf1 (and nf2, nf3) - uniform grid size in each dimension
 *  nj - number of NU points
 *  xj (and yj, zj) - length nj array of coordinates of NU points, in [-pi,pi]
 *  opts - spreading opts struct (see cnufftspread.h), sets dir=1 or 2, etc.
 *
 * Inputs/Outputs:  (note twice-length double type arrays not complex used)
 *  fw (size nf1, or nf1*nf2, or nf1*nf2*nf3, complex) - uniform grid array.
 *  cj - complex length-nj array of strengths of or at NU points.
 *
 * Returned value is same as cnufftspread.
 *
 * Greengard 1/13/17 fortran; rewrite in C++, doc, rename, Barnett 1/17/17
 * opts in interface 2/17/17
 */

#include "twopispread.h"

int twopispread1d(BIGINT nf1,double *fw,BIGINT nj,double* xj,double* cj,
		  spread_opts opts)
{
  double *dummy;   // note this should never be read from!
  double *xjscal = (double*)malloc(sizeof(double)*nj);
  double s = nf1/(2*M_PI);
  for (BIGINT i=0;i<nj;++i)
    xjscal[i] = s * (xj[i]+M_PI);
  //printf("nf1=%d, xjscal = %.15g, Re cj = %.15g\n",nf1,xjscal[0],cj[0]);

  int ier = cnufftspread(nf1,1,1,fw,nj,xjscal,dummy,dummy,cj,opts);
  return ier;
}

int twopispread2d(long nf1,long nf2, double *fw,BIGINT nj,double* xj,
		  double *yj,double* cj,spread_opts opts)
{
  double *dummy;
  double *xjscal = (double*)malloc(sizeof(double)*nj);
  double *yjscal = (double*)malloc(sizeof(double)*nj);
  double s1 = nf1/(2*M_PI);
  double s2 = nf2/(2*M_PI);
  for (BIGINT i=0;i<nj;++i) {
    xjscal[i] = s1 * (xj[i]+M_PI);
    yjscal[i] = s2 * (yj[i]+M_PI);
  }
  return cnufftspread(nf1,nf2,1,fw,nj,xjscal,yjscal,dummy,cj,opts);
}

int twopispread3d(long nf1,long nf2,long nf3,double *fw,BIGINT nj,double* xj,
		  double *yj,double* zj,double* cj,spread_opts opts)
{
  double *xjscal = (double*)malloc(sizeof(double)*nj);
  double *yjscal = (double*)malloc(sizeof(double)*nj);
  double *zjscal = (double*)malloc(sizeof(double)*nj);
  double s1 = nf1/(2*M_PI);
  double s2 = nf2/(2*M_PI);
  double s3 = nf3/(2*M_PI);
  for (BIGINT i=0;i<nj;++i) {
    xjscal[i] = s1 * (xj[i]+M_PI);
    yjscal[i] = s2 * (yj[i]+M_PI);
    zjscal[i] = s3 * (zj[i]+M_PI);
  }
  return cnufftspread(nf1,nf2,nf3,fw,nj,xjscal,yjscal,zjscal,cj,opts);
}
