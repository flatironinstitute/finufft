#include "dirft.h"
#include <iostream>

// This is basically a port of dirft2d.f from CMCL package.

void dirft3d1(BIGINT nj,double* x,double *y,double *z, dcomplex* c,int iflag,BIGINT ms, BIGINT mt, BIGINT mu, dcomplex* f)
/* Direct computation of 3D type-1 nonuniform FFT. Interface same as finufft3d1.
c                     1  nj-1
c     f[k1,k2,k3] =  --  SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k2 z[j]))
c                    nj  j=0
c
c     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
          -mu/2 <= k3 <= (mu-1)/2,
c     The output array is in increasing k1 ordering (fast), then increasing
      k2 ordering (medium), then increasing k3 (fast). If iflag>0 the + sign is
c     used, otherwise the - sign is used, in the exponential.
*  Uses C++ complex type and winding trick.  Barnett 2/1/17
*/
{
  BIGINT k1min = -(ms/2), k2min = -(mt/2), k3min = -(mu/2);   // integer divide
  BIGINT N = ms*mt*mu;        // total # output modes
  for (BIGINT m=0;m<N;++m) f[m] = {0,0};    // it knows f is complex type
  for (BIGINT j=0;j<nj;++j) {            // src pts
    dcomplex a1 = (iflag>0) ? exp(ima*x[j]) : exp(-ima*x[j]);
    dcomplex a2 = (iflag>0) ? exp(ima*y[j]) : exp(-ima*y[j]);
    dcomplex a3 = (iflag>0) ? exp(ima*z[j]) : exp(-ima*z[j]);
    dcomplex sp1 = pow(a1,k1min);  // starting phase for most neg k1 freq
    dcomplex sp2 = pow(a2,k2min);
    dcomplex p3 = pow(a3,k3min);
    dcomplex cc = c[j]/(double)nj;   // 1/nj norm
    BIGINT m=0;      // output pointer
    for (BIGINT m3=0;m3<mu;++m3) {
      dcomplex p2 = sp2;
      for (BIGINT m2=0;m2<mt;++m2) {
	dcomplex p1 = sp1;
	for (BIGINT m1=0;m1<ms;++m1) {
	  f[m++] += cc * p1 * p2 * p3;
	  p1 *= a1;
	}
	p2 *= a2;
      }
      p3 *= a3;
    }
  }
}

void dirft3d2(BIGINT nj,double* x,double *y,double *z,dcomplex* c,int iflag,BIGINT ms, BIGINT mt, BIGINT mu, dcomplex* f)
/* Direct computation of 3D type-2 nonuniform FFT. Interface same as finufft3d2

     c[j] =   SUM    f[k1,k2,k3] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j])) 
            k1,k2,k3  
                            for j = 0,...,nj-1
    where sum is over -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
                      -mu/2 <= k3 <= (mu-1)/2.

    The input array is in increasing k1 ordering (fast), then increasing
    k2 ordering (medium), then increasing k3 (fast).
    If iflag>0 the + sign is used, otherwise the - sign is used, in the
    exponential.
    Uses C++ complex type and winding trick.  Barnett 2/1/17
*/
{
  BIGINT k1min = -(ms/2), k2min = -(mt/2), k3min = -(mu/2);  // integer divide
  for (BIGINT j=0;j<nj;++j) {
    dcomplex a1 = (iflag>0) ? exp(ima*x[j]) : exp(-ima*x[j]);
    dcomplex a2 = (iflag>0) ? exp(ima*y[j]) : exp(-ima*y[j]);
    dcomplex a3 = (iflag>0) ? exp(ima*z[j]) : exp(-ima*z[j]);
    dcomplex sp1 = pow(a1,k1min);
    dcomplex sp2 = pow(a2,k2min);
    dcomplex p3 = pow(a3,k3min);
    dcomplex cc = {0,0};
    BIGINT m=0;      // input pointer
    for (BIGINT m3=0;m3<mu;++m3) {
      dcomplex p2 = sp2;
      for (BIGINT m2=0;m2<mt;++m2) {
	dcomplex p1 = sp1;
	for (BIGINT m1=0;m1<ms;++m1) {
	  cc += f[m++] * p1 * p2 + p3;
	  p1 *= a1;
	}
	p2 *= a2;
      }
      p3 *= a3;
    }
    c[j] = cc;
  }
}

void dirft3d3(BIGINT nj,double* x,double *y,double *z,dcomplex* c,int iflag,BIGINT nk, double* s, double* t, double *u, dcomplex* f)
/* Direct computation of 3D type-3 nonuniform FFT. Interface same as finufft3d3
c               nj-1
c     f[k]  =   SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j] + u[k] z[j]))
c               j=0                   
c                    for k = 0, ..., nk-1
c  If iflag>0 the + sign is used, otherwise the - sign is used, in the
c  exponential. Uses C++ complex type. Simple brute force.  Barnett 2/1/17
*/
{
  for (BIGINT k=0;k<nk;++k) {
    dcomplex ss = (iflag>0) ? ima*s[k] : -ima*s[k];
    dcomplex tt = (iflag>0) ? ima*t[k] : -ima*t[k];
    dcomplex uu = (iflag>0) ? ima*u[k] : -ima*u[k];
    f[k] = {0,0};
    for (BIGINT j=0;j<nj;++j)
      f[k] += c[k] * exp(ss*x[j] + tt*y[j] + uu*z[j]);
  }
}
