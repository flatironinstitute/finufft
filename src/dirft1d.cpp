#include "dirft.h"
#include <iostream>

// This is basically a port of dirft1d.f from CMCL package.

void dirft1d1(BIGINT nj,double* x,dcomplex* c,int iflag,BIGINT ms, dcomplex* f)
/* Direct computation of 1D type-1 nonuniform FFT. Interface same as finufft1d1.
c                  1  nj-1
c     f[k1]    =  --  SUM  c[j] exp(+-i k1 x[j]) 
c                 nj  j=0
c
c     for -ms/2 <= k1 <= (ms-1)/2.
c     The output array is in increasing k1 ordering. If iflag>0 the + sign is
c     used, otherwise the - sign is used, in the exponential.
*  Uses C++ complex type and winding trick.  Barnett 1/25/17
*/
{
  BIGINT kmin = -(ms/2);                   // integer divide
  for (BIGINT m=0;m<ms;++m) f[m] = {0,0};  // it knows f is complex type
  for (BIGINT j=0;j<nj;++j) {
    dcomplex a = (iflag>0) ? exp(ima*x[j]) : exp(-ima*x[j]);
    dcomplex p = pow(a,kmin);   // starting phase for most neg freq
    dcomplex cc = c[j]/(double)nj;   // 1/nj norm
    for (BIGINT m=0;m<ms;++m) {
      f[m] += cc * p;
      p *= a;
    }
  }
}

void dirft1d2(BIGINT nj,double* x,dcomplex* c,int iflag,BIGINT ms, dcomplex* f)
/* Direct computation of 1D type-2 nonuniform FFT. Interface same as finufft1d2
c
c     c[j] = SUM   f[k1] exp(+-i k1 x[j]) 
c             k1  
c                            for j = 0,...,nj-1
c     where sum is over -ms/2 <= k1 <= (ms-1)/2.
c     The input array is in increasing k1 ordering. If iflag>0 the + sign is
c     used, otherwise the - sign is used, in the exponential.
*  Uses C++ complex type and winding trick.  Barnett 1/25/17
*/
{
  BIGINT kmin = -(ms/2);                     // integer divide
  for (BIGINT j=0;j<nj;++j) {
    dcomplex a = (iflag>0) ? exp(ima*x[j]) : exp(-ima*x[j]);
    dcomplex p = pow(a,kmin);   // starting phase for most neg freq
    dcomplex cc = {0,0};
    for (BIGINT m=0;m<ms;++m) {
      cc += f[m] * p;
      p *= a;
    }
    c[j] = cc;
  }
}

void dirft1d3(BIGINT nj,double* x,dcomplex* c,int iflag,BIGINT nk, double* s, dcomplex* f)
/* Direct computation of 1D type-3 nonuniform FFT. Interface same as finufft1d3
c              nj-1
c     f[k]  =  SUM   c[j] exp(+-i s[k] x[j]) 
c              j=0                   
c                    for k = 0, ..., nk-1
c  If iflag>0 the + sign is used, otherwise the - sign is used, in the
c  exponential. Uses C++ complex type. Simple brute force.  Barnett 1/25/17
*/
{
  for (BIGINT k=0;k<nk;++k) {
    dcomplex ss = (iflag>0) ? ima*s[k] : -ima*s[k];
    f[k] = {0,0};
    for (BIGINT j=0;j<nj;++j)
      f[k] += c[k] * exp(ss*x[j]);
  }
}
